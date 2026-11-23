//! Unicode recovery for composite fonts that lack a `/ToUnicode` CMap.
//!
//! A Type0 font with `Identity-H`/`Identity-V` encoding addresses glyphs by
//! CID. Without a `/ToUnicode` CMap there is no declared text mapping, and
//! PDFium falls back to reporting the raw character code, so a page of
//! subset-embedded TrueType text comes out as `\u{1}\u{2}\u{3}...`.
//!
//! The text is still recoverable: the CID maps to a glyph id through the
//! CIDFont's `/CIDToGIDMap` (identity or a stream of big-endian u16s), and
//! the embedded TrueType program (`/FontFile2`) usually keeps a `cmap`
//! table that maps Unicode code points to those glyph ids. Inverting that
//! table gives CID -> text. This is what pdf.js and poppler do too.
//!
//! PDFium does not expose font dictionaries, so this module carries a
//! deliberately small PDF object reader: it scans `N G obj` definitions
//! (no xref needed, so it also works on damaged files), unpacks object
//! streams, and inflates Flate streams. It is only consulted for pages whose
//! extracted text is full of control characters, so normal documents never
//! pay for it. Encrypted documents are skipped (their streams are
//! ciphertext).

use std::collections::HashMap;
use std::io::Read;

/// CID -> text for one font.
pub(crate) type CidMap = HashMap<u32, String>;

/// Recovered maps keyed by `/BaseFont` name (as PDFium reports it, e.g.
/// `ABCDEF+Arial`).
#[derive(Debug, Default, Clone)]
pub(crate) struct FontMaps {
    by_name: HashMap<String, CidMap>,
}

impl FontMaps {
    pub(crate) fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Look up the text of character code `code` drawn with font `name`.
    pub(crate) fn lookup(&self, name: &str, code: u32) -> Option<&str> {
        let map = self.by_name.get(name).or_else(|| {
            // PDFium may report the name with or without the subset tag.
            let stripped = strip_subset_tag(name);
            self.by_name
                .iter()
                .find(|(k, _)| strip_subset_tag(k) == stripped)
                .map(|(_, v)| v)
        })?;
        map.get(&code).map(String::as_str)
    }

    pub(crate) fn has_font(&self, name: &str) -> bool {
        let stripped = strip_subset_tag(name);
        self.by_name
            .keys()
            .any(|k| k == name || strip_subset_tag(k) == stripped)
    }
}

fn strip_subset_tag(name: &str) -> &str {
    match name.split_once('+') {
        Some((tag, rest)) if tag.len() == 6 && tag.bytes().all(|b| b.is_ascii_uppercase()) => rest,
        _ => name,
    }
}

/// Is this text what PDFium produces for an unmapped composite font, i.e.
/// does it carry raw low character codes as control characters?
pub(crate) fn looks_like_raw_codes(text: &str) -> bool {
    text.chars()
        .filter(|&c| (c as u32) < 0x20 && c != '\r' && c != '\n')
        .take(2)
        .count()
        >= 2
}

/// Build CID maps for every Identity-encoded Type0 font without a
/// `/ToUnicode` whose embedded TrueType program has a usable `cmap`.
pub(crate) fn recover_font_maps(pdf: &[u8]) -> FontMaps {
    if contains(pdf, b"/Encrypt") {
        return FontMaps::default();
    }
    let objs = Objects::scan(pdf);
    let mut by_name: HashMap<String, CidMap> = HashMap::new();
    let mut ambiguous: Vec<String> = Vec::new();
    for obj in objs.map.values() {
        let Some(dict) = obj.as_dict() else { continue };
        if dict_name(dict, "Subtype") != Some("Type0") || dict_get(dict, "ToUnicode").is_some() {
            continue;
        }
        if !matches!(
            dict_name(dict, "Encoding"),
            Some("Identity-H" | "Identity-V")
        ) {
            continue;
        }
        let Some(base) = dict_name(dict, "BaseFont") else {
            continue;
        };
        let Some(map) = type0_map(&objs, dict) else {
            continue;
        };
        if map.is_empty() {
            continue;
        }
        match by_name.get(base) {
            Some(existing) if existing != &map => ambiguous.push(base.to_string()),
            _ => {
                by_name.insert(base.to_string(), map);
            }
        }
    }
    for name in ambiguous {
        by_name.remove(&name);
    }
    FontMaps { by_name }
}

fn type0_map(objs: &Objects, dict: &[(String, Obj)]) -> Option<CidMap> {
    let desc = match objs.resolve(dict_get(dict, "DescendantFonts")?) {
        Obj::Array(a) => objs.resolve(a.first()?),
        other => other,
    };
    let cid_font = desc.as_dict()?;
    if dict_name(cid_font, "Subtype") != Some("CIDFontType2") {
        return None;
    }
    let fd = objs
        .resolve(dict_get(cid_font, "FontDescriptor")?)
        .as_dict()?;
    let font_file = objs.stream_data(dict_get(fd, "FontFile2")?)?;
    let gid_to_text = truetype_gid_to_text(&font_file)?;

    let cid_to_gid: Option<Vec<u8>> = match dict_get(cid_font, "CIDToGIDMap") {
        None => None,
        Some(o) => match objs.resolve(o) {
            Obj::Name(n) if n == "Identity" => None,
            Obj::Stream(..) => Some(objs.stream_data(o)?),
            _ => None,
        },
    };

    let mut out = CidMap::new();
    match cid_to_gid {
        None => {
            for (gid, text) in &gid_to_text {
                out.insert(u32::from(*gid), text.clone());
            }
        }
        Some(table) => {
            for (cid, pair) in table.as_chunks::<2>().0.iter().enumerate() {
                let gid = u16::from_be_bytes([pair[0], pair[1]]);
                if gid == 0 {
                    continue;
                }
                if let Some(text) = gid_to_text.get(&gid) {
                    out.insert(cid as u32, text.clone());
                }
            }
        }
    }
    Some(out)
}

/// Invert a TrueType `cmap`: glyph id -> the Unicode text it draws. When a
/// glyph is reachable from several code points (e.g. space and no-break
/// space), the smallest non-private-use code point wins.
fn truetype_gid_to_text(font: &[u8]) -> Option<HashMap<u16, String>> {
    let face = ttf_parser::Face::parse(font, 0).ok()?;
    let cmap = face.tables().cmap?;
    let mut best: HashMap<u16, char> = HashMap::new();
    for sub in cmap.subtables {
        if !sub.is_unicode() {
            continue;
        }
        sub.codepoints(|cp| {
            let Some(c) = char::from_u32(cp) else { return };
            if is_private_use(c) || c.is_control() {
                return;
            }
            if let Some(gid) = sub.glyph_index(cp) {
                if gid.0 == 0 {
                    return;
                }
                let e = best.entry(gid.0).or_insert(c);
                if (c as u32) < (*e as u32) {
                    *e = c;
                }
            }
        });
    }
    if best.is_empty() {
        return None;
    }
    Some(best.into_iter().map(|(g, c)| (g, c.to_string())).collect())
}

fn is_private_use(c: char) -> bool {
    matches!(c as u32, 0xE000..=0xF8FF | 0xF0000..=0x10FFFF)
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

// ---------------------------------------------------------------------------
// Minimal PDF object reader.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Obj {
    Null,
    Bool(bool),
    Num(f64),
    Name(String),
    Str(Vec<u8>),
    Array(Vec<Obj>),
    Dict(Vec<(String, Obj)>),
    Ref(u32),
    /// Dictionary plus the byte range of the raw (still filtered) stream
    /// data in the source file.
    Stream(Vec<(String, Obj)>, std::ops::Range<usize>),
}

impl Obj {
    fn as_dict(&self) -> Option<&[(String, Obj)]> {
        match self {
            Obj::Dict(d) | Obj::Stream(d, _) => Some(d),
            _ => None,
        }
    }
}

fn dict_get<'a>(d: &'a [(String, Obj)], key: &str) -> Option<&'a Obj> {
    d.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn dict_name<'a>(d: &'a [(String, Obj)], key: &str) -> Option<&'a str> {
    match dict_get(d, key)? {
        Obj::Name(n) => Some(n),
        _ => None,
    }
}

struct Objects<'a> {
    src: &'a [u8],
    map: HashMap<u32, Obj>,
}

impl<'a> Objects<'a> {
    /// Collect every `N G obj` definition (later definitions win, as with
    /// incremental updates) plus the contents of object streams.
    fn scan(pdf: &'a [u8]) -> Self {
        let mut map: HashMap<u32, Obj> = HashMap::new();
        let mut i = 0;
        while let Some(off) = find(&pdf[i..], b"obj") {
            let at = i + off;
            i = at + 3;
            if pdf.get(at + 3).is_some_and(|&b| !is_delim_or_ws(b)) {
                continue;
            }
            let Some(num) = obj_header_num(pdf, at) else {
                continue;
            };
            let mut lx = Lexer {
                b: pdf,
                pos: at + 3,
            };
            if let Some(o) = lx.object() {
                let o = lx.maybe_stream(o);
                map.insert(num, o);
                i = i.max(lx.pos);
            }
        }
        let mut objs = Objects { src: pdf, map };
        let streams: Vec<Obj> = objs
            .map
            .values()
            .filter(|o| matches!(o, Obj::Stream(d, _) if dict_name(d, "Type") == Some("ObjStm")))
            .cloned()
            .collect();
        for s in streams {
            for (num, o) in unpack_objstm(pdf, &s).unwrap_or_default() {
                objs.map.entry(num).or_insert(o);
            }
        }
        objs
    }

    fn resolve<'b>(&'b self, o: &'b Obj) -> &'b Obj {
        let mut cur = o;
        for _ in 0..8 {
            match cur {
                Obj::Ref(n) => match self.map.get(n) {
                    Some(next) => cur = next,
                    None => return &Obj::Null,
                },
                _ => return cur,
            }
        }
        &Obj::Null
    }

    fn stream_data(&self, o: &Obj) -> Option<Vec<u8>> {
        match self.resolve(o) {
            Obj::Stream(d, r) => decode_stream(self, d, self.src.get(r.clone())?),
            _ => None,
        }
    }
}

fn unpack_objstm(src: &[u8], s: &Obj) -> Option<Vec<(u32, Obj)>> {
    let Obj::Stream(d, r) = s else { return None };
    let empty = Objects {
        src,
        map: HashMap::new(),
    };
    let data = decode_stream(&empty, d, src.get(r.clone())?)?;
    let n = match dict_get(d, "N")? {
        Obj::Num(v) => *v as usize,
        _ => return None,
    };
    let first = match dict_get(d, "First")? {
        Obj::Num(v) => *v as usize,
        _ => return None,
    };
    let mut lx = Lexer { b: &data, pos: 0 };
    let mut header = Vec::with_capacity(n.min(100_000));
    for _ in 0..n {
        let (Some(Obj::Num(num)), Some(Obj::Num(off))) = (lx.object(), lx.object()) else {
            break;
        };
        header.push((num as u32, off as usize));
    }
    let mut out = Vec::with_capacity(header.len());
    for (num, off) in header {
        let mut lx = Lexer {
            b: &data,
            pos: first.checked_add(off)?,
        };
        if let Some(o) = lx.object() {
            out.push((num, o));
        }
    }
    Some(out)
}

fn decode_stream(objs: &Objects, d: &[(String, Obj)], raw: &[u8]) -> Option<Vec<u8>> {
    let filters: Vec<&str> = match dict_get(d, "Filter").map(|f| objs.resolve(f)) {
        None | Some(Obj::Null) => Vec::new(),
        Some(Obj::Name(n)) => vec![n.as_str()],
        Some(Obj::Array(a)) => a
            .iter()
            .filter_map(|f| match objs.resolve(f) {
                Obj::Name(n) => Some(n.as_str()),
                _ => None,
            })
            .collect(),
        _ => return None,
    };
    // Predictors are not supported (rare on font programs / object streams).
    if let Some(Obj::Dict(p)) = dict_get(d, "DecodeParms").map(|p| objs.resolve(p))
        && matches!(dict_get(p, "Predictor"), Some(Obj::Num(v)) if *v > 1.0)
    {
        return None;
    }
    let mut data = raw.to_vec();
    for f in filters {
        match f {
            "FlateDecode" | "Fl" => {
                let mut out = Vec::new();
                let mut z = flate2::read::ZlibDecoder::new(&data[..]);
                if z.read_to_end(&mut out).is_err() && out.is_empty() {
                    return None;
                }
                data = out;
            }
            _ => return None,
        }
    }
    Some(data)
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Given the offset of an `obj` keyword, read back `N G ` and return N.
fn obj_header_num(b: &[u8], at: usize) -> Option<u32> {
    let mut p = at;
    let skip_ws_back = |p: &mut usize| {
        while *p > 0 && is_ws(b[*p - 1]) {
            *p -= 1;
        }
    };
    let digits_back = |p: &mut usize| -> Option<(usize, usize)> {
        let end = *p;
        while *p > 0 && b[*p - 1].is_ascii_digit() {
            *p -= 1;
        }
        (*p < end).then_some((*p, end))
    };
    if p == 0 || !is_ws(b[p - 1]) {
        return None;
    }
    skip_ws_back(&mut p);
    digits_back(&mut p)?; // generation
    if p == 0 || !is_ws(b[p - 1]) {
        return None;
    }
    skip_ws_back(&mut p);
    let (s, e) = digits_back(&mut p)?;
    if p > 0 && !is_delim_or_ws(b[p - 1]) {
        return None;
    }
    std::str::from_utf8(&b[s..e]).ok()?.parse().ok()
}

fn is_ws(b: u8) -> bool {
    matches!(b, 0 | b'\t' | b'\n' | 0x0C | b'\r' | b' ')
}

fn is_delim(b: u8) -> bool {
    matches!(
        b,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

fn is_delim_or_ws(b: u8) -> bool {
    is_ws(b) || is_delim(b)
}

struct Lexer<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Lexer<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if is_ws(c) {
                self.pos += 1;
            } else if c == b'%' {
                while let Some(c) = self.peek() {
                    if c == b'\n' || c == b'\r' {
                        break;
                    }
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn regular_token(&mut self) -> &[u8] {
        let s = self.pos;
        while self.peek().is_some_and(|c| !is_delim_or_ws(c)) {
            self.pos += 1;
        }
        &self.b[s..self.pos]
    }

    fn object(&mut self) -> Option<Obj> {
        self.object_depth(0)
    }

    fn object_depth(&mut self, depth: usize) -> Option<Obj> {
        if depth > 64 {
            return None;
        }
        self.skip_ws();
        match self.peek()? {
            b'/' => {
                self.pos += 1;
                Some(Obj::Name(decode_name(self.regular_token())))
            }
            b'[' => {
                self.pos += 1;
                let mut v = Vec::new();
                loop {
                    self.skip_ws();
                    match self.peek()? {
                        b']' => {
                            self.pos += 1;
                            return Some(Obj::Array(v));
                        }
                        _ => v.push(self.object_depth(depth + 1)?),
                    }
                }
            }
            b'<' if self.b.get(self.pos + 1) == Some(&b'<') => {
                self.pos += 2;
                let mut d = Vec::new();
                loop {
                    self.skip_ws();
                    match self.peek()? {
                        b'>' => {
                            self.pos += 2;
                            return Some(Obj::Dict(d));
                        }
                        b'/' => {
                            self.pos += 1;
                            let k = decode_name(self.regular_token());
                            let v = self.object_depth(depth + 1)?;
                            d.push((k, v));
                        }
                        _ => return None,
                    }
                }
            }
            b'<' => {
                self.pos += 1;
                let mut hex = Vec::new();
                while let Some(c) = self.peek() {
                    self.pos += 1;
                    if c == b'>' {
                        break;
                    }
                    if c.is_ascii_hexdigit() {
                        hex.push(c);
                    }
                }
                if hex.len() % 2 == 1 {
                    hex.push(b'0');
                }
                let bytes = hex
                    .chunks(2)
                    .filter_map(|p| u8::from_str_radix(std::str::from_utf8(p).ok()?, 16).ok())
                    .collect();
                Some(Obj::Str(bytes))
            }
            b'(' => {
                self.pos += 1;
                let mut depth_p = 1usize;
                let mut out = Vec::new();
                while let Some(c) = self.peek() {
                    self.pos += 1;
                    match c {
                        b'\\' => {
                            if let Some(n) = self.peek() {
                                self.pos += 1;
                                out.push(n);
                            }
                        }
                        b'(' => {
                            depth_p += 1;
                            out.push(c);
                        }
                        b')' => {
                            depth_p -= 1;
                            if depth_p == 0 {
                                break;
                            }
                            out.push(c);
                        }
                        _ => out.push(c),
                    }
                }
                Some(Obj::Str(out))
            }
            c if c.is_ascii_digit() || c == b'-' || c == b'+' || c == b'.' => {
                let tok = self.regular_token();
                let v: f64 = std::str::from_utf8(tok).ok()?.parse().ok()?;
                // `N G R` indirect reference?
                if tok.iter().all(u8::is_ascii_digit) {
                    let save = self.pos;
                    self.skip_ws();
                    let generation = self.regular_token();
                    if !generation.is_empty() && generation.iter().all(u8::is_ascii_digit) {
                        self.skip_ws();
                        if self.peek() == Some(b'R')
                            && self.b.get(self.pos + 1).is_none_or(|&c| is_delim_or_ws(c))
                        {
                            self.pos += 1;
                            return Some(Obj::Ref(v as u32));
                        }
                    }
                    self.pos = save;
                }
                Some(Obj::Num(v))
            }
            _ => {
                let tok = self.regular_token();
                match tok {
                    b"true" => Some(Obj::Bool(true)),
                    b"false" => Some(Obj::Bool(false)),
                    b"null" => Some(Obj::Null),
                    _ => None,
                }
            }
        }
    }

    /// If `o` is a dictionary followed by `stream`, attach the stream data.
    fn maybe_stream(&mut self, o: Obj) -> Obj {
        let Obj::Dict(d) = o else { return o };
        let save = self.pos;
        self.skip_ws();
        if !self.b[self.pos..].starts_with(b"stream") {
            self.pos = save;
            return Obj::Dict(d);
        }
        let mut start = self.pos + 6;
        if self.b.get(start) == Some(&b'\r') {
            start += 1;
        }
        if self.b.get(start) == Some(&b'\n') {
            start += 1;
        }
        let declared = match dict_get(&d, "Length") {
            Some(Obj::Num(n)) if *n >= 0.0 => Some(*n as usize),
            _ => None,
        };
        let end = declared
            .and_then(|len| {
                let e = start.checked_add(len)?;
                let tail = self.b.get(e..e.checked_add(12)?.min(self.b.len()))?;
                let mut t = 0;
                while t < tail.len() && is_ws(tail[t]) {
                    t += 1;
                }
                tail[t..].starts_with(b"endstream").then_some(e)
            })
            .or_else(|| {
                let rel = find(&self.b[start..], b"endstream")?;
                let mut e = start + rel;
                if e > start && self.b[e - 1] == b'\n' {
                    e -= 1;
                }
                if e > start && self.b[e - 1] == b'\r' {
                    e -= 1;
                }
                Some(e)
            });
        match end {
            Some(e) => {
                self.pos = e;
                Obj::Stream(d, start..e)
            }
            None => {
                self.pos = save;
                Obj::Dict(d)
            }
        }
    }
}

fn decode_name(raw: &[u8]) -> String {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'#'
            && i + 2 < raw.len()
            && let Ok(v) =
                u8::from_str_radix(std::str::from_utf8(&raw[i + 1..i + 3]).unwrap_or("zz"), 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(raw[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexer_parses_dicts_refs_and_arrays() {
        let src = b"<< /Type /Font /Subtype /Type0 /DescendantFonts [ 7 0 R ] /W [1 [500 600]] /N#20x 3.5 /S (a\\)b) /H <4142> >>";
        let mut lx = Lexer { b: src, pos: 0 };
        let o = lx.object().unwrap();
        let d = o.as_dict().unwrap();
        assert_eq!(dict_name(d, "Subtype"), Some("Type0"));
        assert_eq!(
            dict_get(d, "DescendantFonts"),
            Some(&Obj::Array(vec![Obj::Ref(7)]))
        );
        assert_eq!(dict_get(d, "N x"), Some(&Obj::Num(3.5)));
        assert_eq!(dict_get(d, "S"), Some(&Obj::Str(b"a)b".to_vec())));
        assert_eq!(dict_get(d, "H"), Some(&Obj::Str(b"AB".to_vec())));
    }

    #[test]
    fn scans_objects_streams_and_object_streams() {
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;
        // Object stream holding objects 10 and 11.
        let inner = b"10 0 11 11 << /A 1 >> << /B /X >>";
        let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
        z.write_all(inner).unwrap();
        let packed = z.finish().unwrap();
        let mut pdf = Vec::new();
        pdf.extend_from_slice(
            b"%PDF-1.5\n1 0 obj\n<< /Length 5 >>\nstream\nhello\nendstream\nendobj\n",
        );
        pdf.extend_from_slice(
            format!(
                "2 0 obj\n<< /Type /ObjStm /N 2 /First 11 /Filter /FlateDecode /Length {} >>\nstream\n",
                packed.len()
            )
            .as_bytes(),
        );
        pdf.extend_from_slice(&packed);
        pdf.extend_from_slice(b"\nendstream\nendobj\n");
        let objs = Objects::scan(&pdf);
        assert_eq!(objs.stream_data(&Obj::Ref(1)).unwrap(), b"hello");
        let a = objs.map.get(&10).unwrap().as_dict().unwrap();
        assert_eq!(dict_get(a, "A"), Some(&Obj::Num(1.0)));
        let b = objs.map.get(&11).unwrap().as_dict().unwrap();
        assert_eq!(dict_name(b, "B"), Some("X"));
    }

    #[test]
    fn raw_code_detection() {
        assert!(looks_like_raw_codes("\u{1}\u{2}\u{3}"));
        assert!(!looks_like_raw_codes("Hello world\r\n"));
        assert!(!looks_like_raw_codes("tab\tonce"));
    }

    #[test]
    fn subset_tag_matching() {
        let mut by_name = HashMap::new();
        by_name.insert(
            "ABCDEF+Arial".to_string(),
            CidMap::from([(3, "x".to_string())]),
        );
        let m = FontMaps { by_name };
        assert_eq!(m.lookup("ABCDEF+Arial", 3), Some("x"));
        assert_eq!(m.lookup("Arial", 3), Some("x"));
        assert!(m.has_font("Arial"));
        assert!(!m.has_font("ArialBold"));
        assert_eq!(m.lookup("Arial", 4), None);
    }

    #[test]
    fn encrypted_documents_are_skipped() {
        let pdf = b"%PDF-1.7\n1 0 obj << /Type /Font /Subtype /Type0 >> endobj\ntrailer << /Encrypt 5 0 R >>";
        assert!(recover_font_maps(pdf).is_empty());
    }
}
