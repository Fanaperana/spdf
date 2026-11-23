//! Stage-by-stage execution of a generated model's forward pass.
//!
//! `burn-onnx` splits a large graph into a handful of submodules and writes
//! `Model::forward` as a chain of calls to them. The CPU recognisers run
//! that chain *stage-major*: every line batch of a wave goes through stage
//! `k` before any enters stage `k + 1`, so once a stage is done its weights
//! can be dropped from the resident set (see `resident.rs`) instead of all
//! of a model's weights staying resident for the whole pass. For the server
//! recogniser that is the difference between ~75 MB and ~35 MB of weights
//! resident at its peak.
//!
//! The build script calls [`staged_impl`] on each generated model source
//! and appends the result, which adds to the model module:
//!
//! - `pub struct Stages`: the values live between stages, one `Option`
//!   field per variable of `Model::forward`;
//! - `Model::STAGES`, `Model::stage_input(..)`, `Model::run_stage(k, &mut
//!   Stages)` and `Model::stage_output(Stages)`.
//!
//! Running all stages in order is exactly `Model::forward`: the same calls
//! on the same values. A variable is moved out of `Stages` at its last use
//! and cloned (a reference-count bump for Burn tensors) before that, as in
//! the original code. When the forward body has any other shape, the whole
//! forward becomes a single stage, which is always correct.
//!
//! No dependencies: the build script includes this file directly.

#![allow(dead_code)]

/// One call in the stage chain: `let <outs> = <call>(<args>);`.
#[derive(Debug, Clone, PartialEq)]
struct Step {
    /// Receiver path, e.g. `self.submodule3`.
    module: String,
    /// Bound output variables with their types.
    outs: Vec<(String, String)>,
    /// Argument variables, each with whether the source cloned it.
    args: Vec<(String, bool)>,
}

/// A parsed `fn forward(&self, <params>) -> <ret> { <body> }`.
struct Forward<'a> {
    params: Vec<(String, String)>,
    ret: String,
    body: &'a str,
}

/// Index of the bracket closing the one at `open` (any of `(`, `{`, `<`
/// pairs with its partner; `->` is not a closing angle bracket).
fn matching(s: &str, open: usize) -> Option<usize> {
    let b = s.as_bytes();
    let (o, c) = match b[open] {
        b'(' => (b'(', b')'),
        b'{' => (b'{', b'}'),
        b'<' => (b'<', b'>'),
        _ => return None,
    };
    let mut depth = 0usize;
    for (i, &ch) in b.iter().enumerate().skip(open) {
        if ch == o {
            depth += 1;
        } else if ch == c && !(c == b'>' && i > 0 && b[i - 1] == b'-') {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Split at commas outside any `()` / `<>` nesting; trims and drops empty
/// pieces (trailing commas).
fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (mut depth, mut cur) = (0i32, String::new());
    let b = s.as_bytes();
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '<' => depth += 1,
            ')' => depth -= 1,
            '>' if !(i > 0 && b[i - 1] == b'-') => depth -= 1,
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(ch);
    }
    out.push(cur);
    out.into_iter()
        .map(|p| squash(&p))
        .filter(|p| !p.is_empty())
        .collect()
}

/// Collapse whitespace runs to one space and trim.
fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Remove all whitespace.
fn tight(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// The block of `impl <ty> {` that contains `pub fn forward(`.
fn find_forward<'a>(src: &'a str, ty: &str) -> Option<Forward<'a>> {
    let head = format!("impl {ty} {{");
    let mut from = 0;
    while let Some(p) = src[from..].find(&head) {
        let start = from + p;
        let open = start + head.len() - 1;
        let close = matching(src, open)?;
        let block = &src[open..close];
        if let Some(f) = block.find("pub fn forward(") {
            let f = open + f;
            let popen = f + "pub fn forward".len();
            let pclose = matching(src, popen)?;
            let bopen = pclose + src[pclose..].find('{')?;
            let ret = src[pclose + 1..bopen].trim().strip_prefix("->")?.trim();
            let bclose = matching(src, bopen)?;
            let mut params = Vec::new();
            for p in split_top(&src[popen + 1..pclose]) {
                if p == "&self" {
                    continue;
                }
                let (n, t) = p.split_once(':')?;
                params.push((n.trim().to_string(), squash(t)));
            }
            return Some(Forward {
                params,
                ret: squash(ret),
                body: &src[bopen + 1..bclose],
            });
        }
        from = close;
    }
    None
}

/// Field types of `pub struct <ty> { .. }`.
fn struct_fields(src: &str, ty: &str) -> Option<Vec<(String, String)>> {
    let head = format!("pub struct {ty} {{");
    let open = src.find(&head)? + head.len() - 1;
    let close = matching(src, open)?;
    let mut out = Vec::new();
    for f in split_top(&src[open + 1..close]) {
        // Drop attributes such as `#[module(skip)]`.
        let f = f.rsplit(']').next().unwrap_or(&f).trim();
        let (n, t) = f.split_once(':')?;
        out.push((n.trim().to_string(), squash(t)));
    }
    Some(out)
}

/// Parse `Model::forward` into its chain of submodule calls; `None` when
/// the body is anything but `let <pat> = self.<field>.forward(<vars>);`
/// statements followed by one returned variable.
fn chain(src: &str) -> Option<(Forward<'_>, Vec<Step>, String)> {
    let fwd = find_forward(src, "Model")?;
    let fields = struct_fields(src, "Model")?;
    let mut stmts: Vec<&str> = fwd.body.split(';').collect();
    let tail = tight(stmts.pop()?);
    if tail.is_empty() || !tail.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    let mut steps = Vec::new();
    for st in stmts {
        let st = st.trim().strip_prefix("let ")?;
        let (pat, call) = st.split_once('=')?;
        let call = tight(call);
        let rest = call.strip_prefix("self.")?;
        let (field, rest) = rest.split_once(".forward(")?;
        let args = rest.strip_suffix(')')?;
        let ty = &fields.iter().find(|(n, _)| n == field)?.1;
        let ret = find_forward(src, ty)?.ret;
        let pat = tight(pat);
        let (names, types) = match pat.strip_prefix('(') {
            Some(p) => {
                let names: Vec<String> =
                    p.strip_suffix(')')?.split(',').map(String::from).collect();
                let types = split_top(ret.strip_prefix('(')?.strip_suffix(')')?);
                (names, types)
            }
            None => (vec![pat.clone()], vec![ret.clone()]),
        };
        if names.len() != types.len() || names.iter().any(|n| n.is_empty()) {
            return None;
        }
        let mut av = Vec::new();
        for a in args.split(',').filter(|a| !a.is_empty()) {
            let (n, cloned) = match a.strip_suffix(".clone()") {
                Some(n) => (n, true),
                None => (a, false),
            };
            if !n.chars().all(|c| c.is_alphanumeric() || c == '_') {
                return None;
            }
            av.push((n.to_string(), cloned));
        }
        steps.push(Step {
            module: format!("self.{field}"),
            outs: names.into_iter().zip(types).collect(),
            args: av,
        });
    }
    if steps.is_empty() {
        return None;
    }
    Some((fwd, steps, tail))
}

/// Code adding the stage-by-stage API (see the module docs) for the model
/// in `src`, a `burn-onnx`-generated source file; `None` when `src` has no
/// recognisable `Model::forward`.
pub fn staged_impl(src: &str) -> Option<String> {
    let (params, ret, steps, tail) = match chain(src) {
        Some((fwd, steps, tail)) => (fwd.params, fwd.ret, steps, tail),
        None => {
            // One stage: the whole forward pass.
            let fwd = find_forward(src, "Model")?;
            let step = Step {
                module: "self".into(),
                outs: vec![("output".into(), fwd.ret.clone())],
                args: fwd.params.iter().map(|(n, _)| (n.clone(), false)).collect(),
            };
            (fwd.params, fwd.ret, vec![step], "output".to_string())
        }
    };
    // Every variable with its type; inputs first.
    let mut vars: Vec<(String, String)> = params.clone();
    for s in &steps {
        vars.extend(s.outs.iter().cloned());
    }
    let defined = |n: &str| vars.iter().any(|(v, _)| v == n);
    if !defined(&tail) || steps.iter().flat_map(|s| &s.args).any(|(a, _)| !defined(a)) {
        return None;
    }
    let last_use = |name: &str| {
        steps
            .iter()
            .rposition(|s| s.args.iter().any(|(a, _)| a == name))
    };

    let mut o = String::new();
    o.push_str("\n// Stage-by-stage forward pass, generated by spdf-ocr's build script.\n");
    o.push_str("#[derive(Default)]\npub struct Stages {\n");
    for (n, t) in &vars {
        o.push_str(&format!("    {n}: Option<{t}>,\n"));
    }
    o.push_str("}\n\nimpl Model {\n");
    o.push_str(&format!(
        "    pub const STAGES: usize = {};\n\n",
        steps.len()
    ));
    let plist: Vec<String> = params.iter().map(|(n, t)| format!("{n}: {t}")).collect();
    o.push_str(&format!(
        "    pub fn stage_input({}) -> Stages {{\n        Stages {{\n",
        plist.join(", ")
    ));
    for (n, _) in &params {
        o.push_str(&format!("            {n}: Some({n}),\n"));
    }
    o.push_str("            ..Default::default()\n        }\n    }\n\n");
    o.push_str("    pub fn run_stage(&self, k: usize, s: &mut Stages) {\n        match k {\n");
    for (k, st) in steps.iter().enumerate() {
        o.push_str(&format!("            {k} => {{\n"));
        let mut seen: Vec<&str> = Vec::new();
        for (a, _) in &st.args {
            if seen.contains(&a.as_str()) {
                continue;
            }
            seen.push(a);
            let get = if last_use(a) == Some(k) && *a != tail {
                "take()"
            } else {
                "clone()"
            };
            o.push_str(&format!(
                "                let {a} = s.{a}.{get}.expect(\"stage {k}: {a}\");\n"
            ));
        }
        let args: Vec<String> = st
            .args
            .iter()
            .map(|(a, c)| {
                if *c {
                    format!("{a}.clone()")
                } else {
                    a.clone()
                }
            })
            .collect();
        let names: Vec<&str> = st.outs.iter().map(|(n, _)| n.as_str()).collect();
        let pat = if names.len() == 1 {
            names[0].to_string()
        } else {
            format!("({})", names.join(", "))
        };
        o.push_str(&format!(
            "                let {pat} = {}.forward({});\n",
            st.module,
            args.join(", ")
        ));
        for n in names {
            o.push_str(&format!("                s.{n} = Some({n});\n"));
        }
        o.push_str("            }\n");
    }
    o.push_str("            _ => panic!(\"no stage {k}\"),\n        }\n    }\n\n");
    o.push_str(&format!(
        "    pub fn stage_output(s: Stages) -> {ret} {{\n        s.{tail}.expect(\"all stages run\")\n    }}\n}}\n"
    ));
    Some(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shaped like `burn-onnx` output (rustfmt-wrapped calls, a tuple
    /// return, a scalar output, a cloned argument).
    const SRC: &str = r#"
#[derive(Module, Debug)]
pub struct Submodule1 {
    conv2d1: Conv2d,
}
impl Submodule1 {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let a = self.conv2d1.forward(x);
        a
    }
}
impl Submodule2 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self { todo!() }

    pub fn forward(
        &self,
        h: Tensor<4>,
    ) -> (Tensor<3>, i64, Tensor<4>) {
        todo!()
    }
}
impl Submodule3 {
    pub fn forward(&self, a: Tensor<3>, n: i64, b: Tensor<4>, h: Tensor<4>) -> Tensor<3> {
        todo!()
    }
}
#[derive(Module, Debug)]
pub struct Model {
    submodule1: Submodule1,
    submodule2: Submodule2,
    submodule3: Submodule3,
    #[module(skip)]
    device: Device,
}
impl Model {
    pub fn new(device: &Device) -> Self { todo!() }

    #[allow(clippy::let_and_return)]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<3> {
        let h = self.submodule1.forward(x);
        let (a, n, b) = self
            .submodule2
            .forward(h.clone());
        let out = self
            .submodule3
            .forward(a, n, b, h);
        out
    }
}
"#;

    #[test]
    fn parses_the_submodule_chain() {
        let (fwd, steps, tail) = chain(SRC).expect("chain");
        assert_eq!(fwd.params, vec![("x".to_string(), "Tensor<4>".to_string())]);
        assert_eq!(fwd.ret, "Tensor<3>");
        assert_eq!(tail, "out");
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[1].module, "self.submodule2");
        assert_eq!(
            steps[1].outs,
            vec![
                ("a".to_string(), "Tensor<3>".to_string()),
                ("n".to_string(), "i64".to_string()),
                ("b".to_string(), "Tensor<4>".to_string()),
            ]
        );
        assert_eq!(steps[1].args, vec![("h".to_string(), true)]);
        assert_eq!(steps[2].args.len(), 4);
    }

    #[test]
    fn values_move_at_their_last_use_only() {
        let code = staged_impl(SRC).expect("staged");
        assert!(code.contains("pub const STAGES: usize = 3;"));
        assert!(code.contains("h: Option<Tensor<4>>,"));
        assert!(code.contains("n: Option<i64>,"));
        // `h` feeds stages 1 and 2: cloned in 1, moved in 2.
        assert!(code.contains("let h = s.h.clone().expect(\"stage 1: h\");"));
        assert!(code.contains("let h = s.h.take().expect(\"stage 2: h\");"));
        assert!(code.contains("let (a, n, b) = self.submodule2.forward(h.clone());"));
        assert!(code.contains("let x = s.x.take().expect(\"stage 0: x\");"));
        assert!(code.contains("pub fn stage_output(s: Stages) -> Tensor<3> {"));
        assert!(code.contains("s.out.expect(\"all stages run\")"));
    }

    #[test]
    fn other_bodies_become_one_stage() {
        let src = SRC.replace(
            "let h = self.submodule1.forward(x);",
            "let h = self.submodule1.forward(x).relu();",
        );
        let code = staged_impl(&src).expect("fallback");
        assert!(code.contains("pub const STAGES: usize = 1;"));
        assert!(code.contains("let output = self.forward(x);"));
        assert!(code.contains("pub fn stage_output(s: Stages) -> Tensor<3> {"));
    }

    #[test]
    fn helpers_respect_nesting() {
        assert_eq!(
            split_top(" a: Tensor<4, Int>, b: (i64, f32), "),
            vec!["a: Tensor<4, Int>", "b: (i64, f32)"]
        );
        let s = "f(a, g(b)) -> X { }";
        assert_eq!(matching(s, 1), Some(9));
        assert!(staged_impl("no model here").is_none());
    }
}
