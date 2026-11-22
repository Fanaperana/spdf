"""Generate the spdf edge-case corpus deterministically.

Each case produces ``corpus/<name>.pdf`` plus ``corpus/<name>.json`` holding
the ground truth and how the case is scored:

    {"gt": "...expected text in reading order...",
     "kind": "text" | "robust",        # robust = only graceful handling is scored
     "password": "..." (optional),
     "tags": [...]}

Generated PDFs are NOT committed (they embed subsets of system fonts);
run ``python bench/edge/make_corpus.py`` (setup.ps1 does it) to rebuild.
Fonts: Windows system fonts (Arial, Arial Unicode MS, Malgun Gothic,
Calibri, Times New Roman). Override the font dir with SPDF_BENCH_FONTS.
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import TextMode
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus"
FONTS = Path(os.environ.get("SPDF_BENCH_FONTS", r"C:\Windows\Fonts"))
random.seed(1234)

LOREM = (
    "Spatial parsing keeps the geometry of every page intact. Invoices, tax bills "
    "and scientific papers depend on reading order. A good parser must not shuffle "
    "columns, drop numbers, or merge unrelated lines together."
).split()


def sentences(n: int, seed: int) -> list[str]:
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        k = rnd.randint(8, 14)
        words = [rnd.choice(LOREM).strip(".,") for _ in range(k)]
        words[0] = words[0].capitalize()
        out.append(" ".join(words) + ".")
    return out


class Doc(FPDF):
    def __init__(self, orientation="P", fmt="Letter"):
        super().__init__(orientation=orientation, unit="pt", format=fmt)
        self.set_auto_page_break(False)
        self.set_compression(True)
        self.add_font("Arial", "", str(FONTS / "arial.ttf"))
        self.add_font("Arial", "B", str(FONTS / "arialbd.ttf"))
        self.set_font("Arial", size=11)


def save(name: str, pdf_bytes: bytes, gt: str, kind: str = "text", **extra) -> None:
    (OUT / f"{name}.pdf").write_bytes(pdf_bytes)
    meta = {"gt": gt, "kind": kind, **extra}
    (OUT / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def lines_block(pdf: Doc, x: float, y: float, w: float, lines: list[str], lh: float = 14) -> float:
    for ln in lines:
        pdf.set_xy(x, y)
        pdf.cell(w, lh, ln)
        y += lh
    return y


def wrap(pdf: Doc, text: str, width: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if pdf.get_string_width(trial) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


# ----------------------------------------------------------------------------- cases

def case_simple():
    pdf = Doc(); pdf.add_page()
    paras = sentences(6, 1)
    pdf.set_font("Arial", "B", 16)
    pdf.set_xy(72, 72); pdf.cell(400, 20, "Quarterly Report")
    pdf.set_font("Arial", "", 11)
    lines = wrap(pdf, " ".join(paras), 460)
    lines_block(pdf, 72, 110, 460, lines)
    return pdf.output(), "Quarterly Report\n" + "\n".join(lines)


def case_two_column():
    pdf = Doc(); pdf.add_page()
    left = wrap(pdf, " ".join(sentences(7, 2)), 220)
    right = wrap(pdf, " ".join(sentences(7, 3)), 220)
    lines_block(pdf, 60, 80, 220, left)
    lines_block(pdf, 320, 80, 220, right)
    return pdf.output(), "\n".join(left + right)


def case_three_column():
    pdf = Doc(orientation="L"); pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.set_xy(50, 40); pdf.cell(600, 24, "The Daily Column Times")
    pdf.set_font("Arial", "", 10)
    cols = [wrap(pdf, " ".join(sentences(6, 10 + i)), 210) for i in range(3)]
    for i, c in enumerate(cols):
        lines_block(pdf, 50 + i * 240, 90, 210, c, 12)
    return pdf.output(), "The Daily Column Times\n" + "\n".join(sum(cols, []))


def _table(pdf: Doc, x0: float, y0: float, rows: list[list[str]], widths: list[float], border: bool):
    y = y0
    for r in rows:
        x = x0
        for cell, w in zip(r, widths):
            pdf.set_xy(x, y)
            pdf.cell(w, 18, cell, border=1 if border else 0, align="L")
            x += w
        y += 18
    return y


def table_rows():
    rnd = random.Random(7)
    rows = [["Region", "Q1", "Q2", "Q3", "Total"]]
    for name in ["North", "South", "East", "West", "Central", "Overseas"]:
        q = [rnd.randint(100, 9999) for _ in range(3)]
        rows.append([name] + [f"{v:,}" for v in q] + [f"{sum(q):,}"])
    return rows


def case_table_bordered():
    pdf = Doc(); pdf.add_page()
    rows = table_rows()
    pdf.set_xy(72, 60); pdf.cell(300, 16, "Table 1. Sales by region")
    _table(pdf, 72, 90, rows, [110, 80, 80, 80, 90], True)
    return pdf.output(), "Table 1. Sales by region\n" + "\n".join(" ".join(r) for r in rows)


def case_table_borderless():
    pdf = Doc(); pdf.add_page()
    rows = table_rows()
    _table(pdf, 72, 90, rows, [110, 80, 80, 80, 90], False)
    return pdf.output(), "\n".join(" ".join(r) for r in rows)


def case_table_sparse():
    """Table with empty cells and a wrapped multi-line header."""
    pdf = Doc(); pdf.add_page()
    rows = [["Item", "Qty", "Unit price", "Notes"],
            ["Widget A", "4", "12.50", ""],
            ["Widget B", "", "7.25", "backorder"],
            ["Service fee", "1", "40.00", ""],
            ["", "", "Subtotal", "116.25"]]
    _table(pdf, 72, 90, rows, [140, 60, 100, 120], True)
    gt = "\n".join(" ".join(c for c in r if c) for r in rows)
    return pdf.output(), gt


def _rotate(pdf_bytes: bytes, deg: int) -> bytes:
    r = PdfReader(io.BytesIO(pdf_bytes)); w = PdfWriter()
    for p in r.pages:
        p.rotate(deg); w.add_page(p)
    b = io.BytesIO(); w.write(b); return b.getvalue()


def case_rotated(deg):
    def f():
        b, gt = case_simple()
        return _rotate(bytes(b), deg), gt
    return f


def case_rotated_text():
    pdf = Doc(); pdf.add_page()
    body = wrap(pdf, " ".join(sentences(3, 20)), 400)
    lines_block(pdf, 100, 80, 400, body)
    with pdf.rotation(90, x=50, y=700):
        pdf.set_xy(50, 700); pdf.cell(300, 14, "Sidebar label rotated ninety")
    return pdf.output(), "\n".join(body) + "\nSidebar label rotated ninety"


def case_landscape_wide():
    pdf = Doc(orientation="L", fmt="A3"); pdf.add_page()
    lines = wrap(pdf, " ".join(sentences(8, 30)), 1000)
    lines_block(pdf, 60, 80, 1000, lines)
    return pdf.output(), "\n".join(lines)


def _unicode_case(font_file: str, text_lines: list[str], shaping: bool = False, rtl: bool = False):
    pdf = Doc(); pdf.add_page()
    pdf.add_font("U", "", str(FONTS / font_file))
    pdf.set_font("U", size=14)
    if shaping:
        pdf.set_text_shaping(use_shaping_engine=True, direction="rtl" if rtl else "ltr")
    y = 80
    for ln in text_lines:
        pdf.set_xy(72, y)
        pdf.cell(450, 20, ln, align="R" if rtl else "L")
        y += 24
    return pdf.output(), "\n".join(text_lines)


def case_cjk_chinese():
    return _unicode_case("ARIALUNI.TTF", ["空间解析保留页面的几何结构。", "发票和税单依赖正确的阅读顺序。", "第三季度总收入为一千二百万元。"])


def case_japanese():
    return _unicode_case("ARIALUNI.TTF", ["日本語のテキスト抽出テストです。", "カタカナとひらがなと漢字が混在します。"])


def case_korean():
    return _unicode_case("malgun.ttf", ["한국어 텍스트 추출 테스트입니다.", "세금 고지서와 청구서를 읽습니다."])


def case_arabic():
    return _unicode_case("arial.ttf", ["مرحبا بالعالم", "هذا اختبار لاستخراج النص العربي"], shaping=True, rtl=True)


def case_hebrew():
    return _unicode_case("arial.ttf", ["שלום עולם", "זהו מבחן לחילוץ טקסט בעברית"], shaping=True, rtl=True)


def case_diacritics():
    return _unicode_case("arial.ttf", ["Façade naïve coöperate déjà vu résumé", "Tiếng Việt có dấu: Hà Nội, Đà Nẵng", "Zürich Ångström Øresund Łódź"])


def case_ligatures():
    pdf = Doc(); pdf.add_page()
    pdf.add_font("Cal", "", str(FONTS / "calibri.ttf"))
    pdf.set_font("Cal", size=14)
    pdf.set_text_shaping(use_shaping_engine=True, features={"liga": True, "kern": True})
    lines = ["The office staff filed fifty official affidavits.", "Efficient workflow: fjord, afflict, waffle, shuffle."]
    lines_block(pdf, 72, 80, 450, lines, 22)
    return pdf.output(), "\n".join(lines)


def case_tiny_and_huge_text():
    pdf = Doc(); pdf.add_page()
    pdf.set_font("Arial", "B", 40); pdf.set_xy(72, 60); pdf.cell(450, 48, "BIG TITLE")
    pdf.set_font("Arial", "", 11)
    body = wrap(pdf, " ".join(sentences(3, 40)), 450)
    y = lines_block(pdf, 72, 130, 450, body)
    pdf.set_font("Arial", "", 4.5); pdf.set_xy(72, y + 10)
    fine = "Fine print: terms apply, see section 4.2 for the refund policy."
    pdf.cell(450, 6, fine)
    return pdf.output(), "BIG TITLE\n" + "\n".join(body) + "\n" + fine


def case_headers_footers():
    pdf = Doc(); gts = []
    for i in range(3):
        pdf.add_page()
        pdf.set_font("Arial", "", 9); pdf.set_xy(72, 30); pdf.cell(450, 12, "ACME Corp Confidential")
        pdf.set_font("Arial", "", 11)
        body = wrap(pdf, " ".join(sentences(4, 50 + i)), 450)
        lines_block(pdf, 72, 80, 450, body)
        pdf.set_font("Arial", "", 9); pdf.set_xy(280, 740); pdf.cell(60, 12, f"Page {i + 1} of 3")
        gts.append("ACME Corp Confidential\n" + "\n".join(body) + f"\nPage {i + 1} of 3")
    return pdf.output(), "\n".join(gts)


def case_faux_bold():
    pdf = Doc(); pdf.add_page()
    lines = ["Tax Information Statement", "Parcel Number 1836-22-004"]
    y = 80
    for ln in lines:
        for dx in (0, 0.4, 0.8):
            pdf.set_xy(72 + dx, y); pdf.cell(450, 18, ln)
        y += 24
    return pdf.output(), "\n".join(lines)


def case_justified():
    pdf = Doc(); pdf.add_page()
    txt = " ".join(sentences(6, 60))
    pdf.set_xy(72, 80); pdf.multi_cell(300, 14, txt, align="J")
    return pdf.output(), txt


def case_letter_spaced():
    pdf = Doc(); pdf.add_page()
    pdf.set_font("Arial", "B", 20); pdf.set_char_spacing(8)
    pdf.set_xy(72, 80); pdf.cell(450, 26, "ANNUAL SUMMARY")
    pdf.set_char_spacing(0); pdf.set_font("Arial", "", 11)
    body = wrap(pdf, " ".join(sentences(2, 61)), 450)
    lines_block(pdf, 72, 130, 450, body)
    return pdf.output(), "ANNUAL SUMMARY\n" + "\n".join(body)


def case_invisible_ocr_layer():
    """Searchable-scan style: grey box 'image' + invisible (render mode 3) text."""
    pdf = Doc(); pdf.add_page()
    pdf.set_fill_color(235, 235, 235); pdf.rect(60, 60, 480, 200, style="F")
    lines = wrap(pdf, " ".join(sentences(4, 70)), 440)
    pdf.text_mode = TextMode.INVISIBLE
    lines_block(pdf, 72, 80, 440, lines)
    return pdf.output(), "\n".join(lines)


def case_empty_pages():
    pdf = Doc(); pdf.add_page()
    a = wrap(pdf, " ".join(sentences(2, 80)), 450); lines_block(pdf, 72, 80, 450, a)
    pdf.add_page()  # blank
    pdf.add_page()
    b = wrap(pdf, " ".join(sentences(2, 81)), 450); lines_block(pdf, 72, 80, 450, b)
    return pdf.output(), "\n".join(a + b)


def case_mixed_page_sizes():
    pdf = Doc(); gts = []
    for i, (o, f) in enumerate([("P", "A5"), ("L", "Letter"), ("P", "Legal")]):
        pdf.add_page(orientation=o, format=f)
        ln = f"Section {i + 1}: " + sentences(1, 90 + i)[0]
        pdf.set_xy(40, 50); pdf.cell(300, 14, ln); gts.append(ln)
    return pdf.output(), "\n".join(gts)


def case_form_fields():
    """AcroForm with filled values (reportlab)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    b = io.BytesIO(); c = canvas.Canvas(b, pagesize=letter)
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "Applicant name:"); c.drawString(72, 660, "City:"); c.drawString(72, 620, "Amount due:")
    f = c.acroForm
    f.textfield(name="name", value="Jane Q Public", x=200, y=692, width=200, height=20, borderWidth=1)
    f.textfield(name="city", value="Fort Wayne", x=200, y=652, width=200, height=20, borderWidth=1)
    f.textfield(name="amount", value="1,234.56", x=200, y=612, width=200, height=20, borderWidth=1)
    c.showPage(); c.save()
    gt = "Applicant name: Jane Q Public\nCity: Fort Wayne\nAmount due: 1,234.56"
    return b.getvalue(), gt


def _encrypt(pdf_bytes: bytes, user: str, owner: str) -> bytes:
    r = PdfReader(io.BytesIO(pdf_bytes)); w = PdfWriter()
    for p in r.pages:
        w.add_page(p)
    w.encrypt(user_password=user, owner_password=owner, algorithm="AES-256")
    b = io.BytesIO(); w.write(b); return b.getvalue()


def case_encrypted_user_pw():
    b, gt = case_simple()
    return _encrypt(bytes(b), "secret", "ownerpw"), gt, {"password": "secret"}


def case_encrypted_owner_only():
    b, gt = case_two_column()
    return _encrypt(bytes(b), "", "ownerpw"), gt


def case_no_tounicode():
    """Embedded subset TrueType font with the /ToUnicode CMap stripped."""
    b, gt = case_simple()
    r = PdfReader(io.BytesIO(bytes(b))); w = PdfWriter(clone_from=r)
    for page in w.pages:
        fonts = page["/Resources"]["/Font"]
        for k in fonts:
            fo = fonts[k].get_object()
            if "/ToUnicode" in fo:
                del fo[NameObject("/ToUnicode")]
    o = io.BytesIO(); w.write(o)
    return o.getvalue(), gt


def _multi_page_doc(n: int, seed: int):
    pdf = Doc(); gts = []
    for i in range(n):
        pdf.add_page()
        lines = wrap(pdf, f"Page {i + 1}. " + " ".join(sentences(10, seed + i)), 460)
        lines_block(pdf, 72, 72, 460, lines)
        gts.append("\n".join(lines))
    return bytes(pdf.output()), "\n".join(gts)


def case_corrupt_truncated():
    b, gt = _multi_page_doc(4, 100)
    cut = b[: int(len(b) * 0.85)]   # xref + trailer gone, last objects cut
    return cut, gt, {"tags": ["corrupt"]}


def case_corrupt_no_xref():
    b, gt = _multi_page_doc(3, 300)
    i = b.rfind(b"xref")
    return b[:i], gt, {"tags": ["corrupt"]}


def case_corrupt_bad_xref():
    b, gt = _multi_page_doc(3, 200)
    i = b.rfind(b"startxref")
    bad = b[:i] + b"startxref\n999999\n%%EOF\n"
    return bad, gt, {"tags": ["corrupt"]}


def _render_to_image_pdf(pdf_bytes: bytes, skew: float = 0.0, dpi: int = 200, noise: float = 0.0,
                         jpeg_quality: int | None = None, rotate: int = 0) -> bytes:
    """Rasterise every page (like a scanner would) and wrap the images in an image-only PDF."""
    import pypdfium2 as pdfium
    from PIL import Image
    doc = pdfium.PdfDocument(pdf_bytes)
    imgs = []
    for i in range(len(doc)):
        im = doc[i].render(scale=dpi / 72).to_pil().convert("L")
        if skew:
            im = im.rotate(skew, expand=True, fillcolor=255)
        if rotate:
            im = im.rotate(rotate, expand=True, fillcolor=255)
        if noise:
            grain = Image.effect_noise(im.size, 64).point(lambda v: 255 if v > 128 else v * 2)
            im = Image.blend(im, grain, noise)
        if jpeg_quality:
            buf = io.BytesIO(); im.save(buf, format="JPEG", quality=jpeg_quality); buf.seek(0)
            im = Image.open(buf).convert("L")
        imgs.append(im)
    out = io.BytesIO()
    kw = {"quality": jpeg_quality} if jpeg_quality else {}
    imgs[0].save(out, format="PDF", resolution=dpi, save_all=True, append_images=imgs[1:], **kw)
    return out.getvalue()


def case_scanned():
    b, gt = case_simple()
    return _render_to_image_pdf(bytes(b)), gt, {"tags": ["ocr"]}


def case_scanned_skewed():
    b, gt = case_two_column()
    return _render_to_image_pdf(bytes(b), skew=2.5), gt, {"tags": ["ocr"]}


def case_scanned_lowdpi():
    b, gt = case_simple()
    return _render_to_image_pdf(bytes(b), dpi=100), gt, {"tags": ["ocr"]}


def case_scanned_noisy_jpeg():
    b, gt = case_justified()
    return _render_to_image_pdf(bytes(b), dpi=150, noise=0.18, jpeg_quality=35), gt, {"tags": ["ocr"]}


def case_scanned_table():
    b, gt = case_table_bordered()
    return _render_to_image_pdf(bytes(b)), gt, {"tags": ["ocr", "table"]}


def case_scanned_chinese():
    b, gt = case_cjk_chinese()
    return _render_to_image_pdf(bytes(b)), gt, {"tags": ["ocr", "cjk"]}


def case_scanned_multilang():
    b, gt = case_diacritics()
    return _render_to_image_pdf(bytes(b)), gt, {"tags": ["ocr", "latin-ext"]}


def case_scanned_rotated_90():
    b, gt = case_simple()
    return _render_to_image_pdf(bytes(b), rotate=90), gt, {"tags": ["ocr", "rotation"]}


def case_scanned_mixed_pages():
    """Page 1 born-digital, page 2 an image-only scan: OCR must fire only where needed."""
    b1, gt1 = case_headers_footers()
    b2, gt2 = case_table_bordered()
    w = PdfWriter()
    w.append(PdfReader(io.BytesIO(bytes(b1))), pages=[0])
    w.append(PdfReader(io.BytesIO(_render_to_image_pdf(bytes(b2)))))
    o = io.BytesIO(); w.write(o)
    gt1_first = gt1.split("\nACME Corp Confidential")[0]
    return o.getvalue(), gt1_first + "\n" + gt2, {"tags": ["ocr", "mixed"]}


def case_huge():
    b, gt = _multi_page_doc(400, 1000)
    return b, gt, {"tags": ["perf"]}


# ---- robustness-only (scored on graceful handling: no hang, no panic) -------

def case_garbage():
    rnd = random.Random(5)
    return bytes(rnd.getrandbits(8) for _ in range(20000)), "", {"kind": "robust"}


def case_zero_bytes():
    return b"", "", {"kind": "robust"}


def case_html_named_pdf():
    return b"<html><body>not a pdf</body></html>", "", {"kind": "robust"}


CASES = {
    "simple": case_simple,
    "two_column": case_two_column,
    "three_column": case_three_column,
    "table_bordered": case_table_bordered,
    "table_borderless": case_table_borderless,
    "table_sparse": case_table_sparse,
    "rotated_page_90": case_rotated(90),
    "rotated_page_180": case_rotated(180),
    "rotated_page_270": case_rotated(270),
    "rotated_text": case_rotated_text,
    "landscape_wide": case_landscape_wide,
    "cjk_chinese": case_cjk_chinese,
    "japanese": case_japanese,
    "korean": case_korean,
    "arabic_rtl": case_arabic,
    "hebrew_rtl": case_hebrew,
    "diacritics": case_diacritics,
    "ligatures": case_ligatures,
    "tiny_and_huge_text": case_tiny_and_huge_text,
    "headers_footers": case_headers_footers,
    "faux_bold": case_faux_bold,
    "justified": case_justified,
    "letter_spaced": case_letter_spaced,
    "invisible_ocr_layer": case_invisible_ocr_layer,
    "empty_pages": case_empty_pages,
    "mixed_page_sizes": case_mixed_page_sizes,
    "form_fields": case_form_fields,
    "encrypted_user_pw": case_encrypted_user_pw,
    "encrypted_owner_only": case_encrypted_owner_only,
    "no_tounicode": case_no_tounicode,
    "corrupt_truncated": case_corrupt_truncated,
    "corrupt_bad_xref": case_corrupt_bad_xref,
    "corrupt_no_xref": case_corrupt_no_xref,
    "scanned": case_scanned,
    "scanned_skewed": case_scanned_skewed,
    "scanned_lowdpi": case_scanned_lowdpi,
    "scanned_noisy_jpeg": case_scanned_noisy_jpeg,
    "scanned_table": case_scanned_table,
    "scanned_chinese": case_scanned_chinese,
    "scanned_multilang": case_scanned_multilang,
    "scanned_rotated_90": case_scanned_rotated_90,
    "scanned_mixed_pages": case_scanned_mixed_pages,
    "huge_400_pages": case_huge,
    "garbage_bytes": case_garbage,
    "zero_bytes": case_zero_bytes,
    "html_named_pdf": case_html_named_pdf,
}


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(argv[1:])
    failures = 0
    for name, fn in CASES.items():
        if only and name not in only:
            continue
        try:
            res = fn()
            data, gt = bytes(res[0]), res[1]
            extra = res[2] if len(res) > 2 else {}
            kind = extra.pop("kind", "text")
            save(name, data, gt, kind=kind, **extra)
            print(f"ok   {name:24s} {len(data):>9,d} bytes")
        except Exception as exc:  # keep going, report
            failures += 1
            print(f"FAIL {name}: {exc!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
