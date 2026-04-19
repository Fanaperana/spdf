#!/usr/bin/env bash
# Reproducible OCR/parse benchmark: spdf vs liteparse vs raw tesseract.
#
# Usage:
#   benchmark/run.sh [LITEPARSE_DIR]
#
# Writes per-fixture outputs into benchmark/outputs/{spdf,lite,tesseract}/,
# and a machine-readable benchmark/results/summary.json + human-readable
# benchmark/results/summary.md. The README pulls its numbers from those.
#
# Requires: spdf on $PATH, tesseract on $PATH, and (optionally) a built
# liteparse checkout whose node_modules + dist/ are present. If liteparse
# is missing the comparison columns are skipped but the benchmark still
# runs for spdf.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/.." && pwd)"

LITEPARSE_DIR="${1:-${LITEPARSE_DIR:-}}"
FIXTURES_DIR="${FIXTURES_DIR:-$ROOT/example}"
OUT_DIR="$HERE/outputs"
RES_DIR="$HERE/results"

mkdir -p "$OUT_DIR/spdf" "$OUT_DIR/lite" "$OUT_DIR/tesseract" "$RES_DIR"

# Fixture list = every file in $FIXTURES_DIR with a whitelisted extension,
# minus any filename listed in benchmark/.fixtureignore. Keeps private PDFs
# out of the committed benchmark artifacts.
IGNORE_FILE="$HERE/.fixtureignore"
should_skip() {
    local base="$1"
    [ -f "$IGNORE_FILE" ] || return 1
    grep -Fxq -- "$base" "$IGNORE_FILE"
}
mapfile -t ALL_FIXTURES < <({
    find "$FIXTURES_DIR" -maxdepth 1 -type f \
        \( -iname '*.pdf' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \)
    if [ -d "$FIXTURES_DIR/corpus" ]; then
        find "$FIXTURES_DIR/corpus" -maxdepth 1 -type f \
            \( -iname '*.pdf' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \)
    fi
} | sort)
FIXTURES=()
for f in "${ALL_FIXTURES[@]}"; do
    if should_skip "$(basename "$f")"; then
        echo "skip (listed in .fixtureignore): $f" >&2
        continue
    fi
    FIXTURES+=("$f")
done

if [ "${#FIXTURES[@]}" -eq 0 ]; then
    echo "no fixtures found under $FIXTURES_DIR" >&2
    exit 1
fi

command -v spdf >/dev/null || { echo "spdf not on PATH (run \`make install-ocr\`)" >&2; exit 1; }
command -v tesseract >/dev/null || { echo "tesseract not on PATH" >&2; exit 1; }

have_lite=0
if [ -n "$LITEPARSE_DIR" ] && [ -f "$LITEPARSE_DIR/dist/src/index.js" ]; then
    have_lite=1
fi

time_ms() {
    # Wall-clock of "$@" in milliseconds. Echoes the duration on stdout,
    # leaves the command's own output untouched by redirecting it.
    local out="$1"; shift
    local t0 t1
    t0=$(date +%s%N)
    "$@" >"$out" 2>/dev/null || true
    t1=$(date +%s%N)
    echo $(((t1 - t0) / 1000000))
}

echo "fixtures:"
for f in "${FIXTURES[@]}"; do echo "  - $f"; done
echo

# Per-fixture rows accumulate into these shell arrays.
declare -a ROWS_JSON
declare -a ROWS_MD

for fixture in "${FIXTURES[@]}"; do
    name="$(basename "$fixture")"
    stem="${name%.*}"

    spdf_out="$OUT_DIR/spdf/$stem.txt"
    lite_out="$OUT_DIR/lite/$stem.txt"
    tess_out="$OUT_DIR/tesseract/$stem.txt"

    echo "== $name =="

    spdf_ms=$(time_ms "$spdf_out" spdf parse "$fixture" --ocr-language eng)
    spdf_words=$(wc -w < "$spdf_out" | tr -d ' ')
    echo "  spdf:      ${spdf_ms} ms, ${spdf_words} words"

    lite_ms=0; lite_words=0
    if [ "$have_lite" = "1" ]; then
        lite_ms=$(time_ms "$lite_out" node "$LITEPARSE_DIR/dist/src/index.js" parse "$fixture" --ocr-language en)
        lite_words=$(wc -w < "$lite_out" | tr -d ' ')
        echo "  liteparse: ${lite_ms} ms, ${lite_words} words"
    else
        echo "  liteparse: skipped (pass LITEPARSE_DIR)"
    fi

    # Ground truth: raw tesseract on the rendered image. For PDFs we render
    # via pdftoppm (same -r as spdf's default), then OCR; images go direct.
    case "$name" in
        *.pdf)
            tmpdir=$(mktemp -d)
            pdftoppm -r 150 "$fixture" "$tmpdir/p" -png >/dev/null 2>&1 || true
            : > "$tess_out"
            for png in "$tmpdir"/p-*.png; do
                tesseract "$png" - -l eng >> "$tess_out" 2>/dev/null || true
            done
            rm -rf "$tmpdir"
            ;;
        *)
            tesseract "$fixture" - -l eng > "$tess_out" 2>/dev/null || true
            ;;
    esac
    tess_words=$(wc -w < "$tess_out" | tr -d ' ')
    echo "  tesseract: (ground truth) ${tess_words} words"

    # Compute recall / precision / F1 vs tesseract ground truth using python.
    # The token regex matches the one we use throughout the pipeline.
    metrics=$(python3 - "$spdf_out" "$lite_out" "$tess_out" <<'PY'
import re, sys, json
from collections import Counter
def toks(p):
    try:
        s = open(p).read()
    except FileNotFoundError:
        return Counter()
    return Counter(t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-/_@%'()]*", s))
spdf, lite, base = toks(sys.argv[1]), toks(sys.argv[2]), toks(sys.argv[3])
def prf(c):
    inter = sum((c & base).values())
    denom_out = sum(c.values())
    denom_gt  = sum(base.values())
    p = inter / denom_out if denom_out else 0.0
    r = inter / denom_gt  if denom_gt  else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return dict(tokens=denom_out, precision=p, recall=r, f1=f)
print(json.dumps({"spdf": prf(spdf), "lite": prf(lite) if lite else None}))
PY
)
    # Stash one JSON object per fixture for aggregation later.
    row=$(python3 - "$name" "$spdf_ms" "$lite_ms" "$tess_words" "$have_lite" "$metrics" <<'PY'
import json, sys
name, spdf_ms, lite_ms, tess_words, have_lite, metrics = sys.argv[1:]
m = json.loads(metrics)
row = {
    "fixture":     name,
    "tess_words":  int(tess_words),
    "spdf": {
        "ms":        int(spdf_ms),
        **m["spdf"],
    },
}
if have_lite == "1" and m["lite"] is not None:
    row["lite"] = {
        "ms": int(lite_ms),
        **m["lite"],
    }
print(json.dumps(row))
PY
)
    ROWS_JSON+=("$row")
done

# Aggregate into summary.json and summary.md.
python3 - "$RES_DIR/summary.json" "$RES_DIR/summary.md" "${ROWS_JSON[@]}" <<'PY'
import json, sys, statistics
out_json, out_md = sys.argv[1], sys.argv[2]
rows = [json.loads(r) for r in sys.argv[3:]]
with open(out_json, "w") as fh:
    json.dump({"rows": rows}, fh, indent=2)

def pct(x): return f"{x*100:.1f}%"
def ms(x):  return f"{x} ms"

lines = []
lines.append("# Benchmark — spdf vs liteparse\n")
lines.append("Ground truth: raw `tesseract <image> - -l eng` (PDFs first rendered with `pdftoppm -r 150`).\n")
lines.append("Token regex: `[A-Za-z0-9][A-Za-z0-9.\\-/_@%'()]*`, case-insensitive multiset precision/recall.\n")
lines.append("")
lines.append("## Per-fixture\n")
lines.append("| fixture | engine | wall-clock | tokens | recall | precision | F1 |")
lines.append("|---|---|---:|---:|---:|---:|---:|")
for r in rows:
    s = r["spdf"]
    lines.append(f"| {r['fixture']} | spdf | {ms(s['ms'])} | {s['tokens']} | {pct(s['recall'])} | {pct(s['precision'])} | {pct(s['f1'])} |")
    if "lite" in r:
        l = r["lite"]
        lines.append(f"| {r['fixture']} | liteparse | {ms(l['ms'])} | {l['tokens']} | {pct(l['recall'])} | {pct(l['precision'])} | {pct(l['f1'])} |")
lines.append("")

def avg(key, engine):
    vals = [r[engine][key] for r in rows if engine in r]
    return statistics.mean(vals) if vals else 0.0

lines.append("## Mean over fixtures\n")
lines.append("| engine | mean recall | mean precision | mean F1 | mean wall-clock |")
lines.append("|---|---:|---:|---:|---:|")
lines.append(f"| spdf      | {pct(avg('recall','spdf'))} | {pct(avg('precision','spdf'))} | {pct(avg('f1','spdf'))} | {avg('ms','spdf'):.0f} ms |")
if any("lite" in r for r in rows):
    lines.append(f"| liteparse | {pct(avg('recall','lite'))} | {pct(avg('precision','lite'))} | {pct(avg('f1','lite'))} | {avg('ms','lite'):.0f} ms |")
lines.append("")
with open(out_md, "w") as fh:
    fh.write("\n".join(lines))
PY

echo
echo "wrote $RES_DIR/summary.json + $RES_DIR/summary.md"
