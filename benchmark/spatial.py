#!/usr/bin/env python3
"""Accuracy + spatial-precision benchmark: spdf vs liteparse.

Produces benchmark/results/spatial.json and spatial.md summarising, per
fixture and overall, how well each engine matches a ground-truth set of
per-word bounding boxes.

Ground truth: `tesseract <image> - -l eng tsv` with
  * PSM_AUTO (default on the CLI) for multi-layout pages,
  * `user_defined_dpi` forced to 150 so libtesseract doesn't bail to its
    70-dpi fallback on PNGs with no pHYs chunk.
For PDFs we pre-render page 1 with `pdftoppm -r 150`. Pixel bboxes are
converted to PDF points with `pt = px * 72 / 150`, matching how
spdf-core lays OCR results into text-item space.

Matching algorithm:
  1. Lower-case both token strings and strip pure-punctuation noise, to
     match what the token-accuracy benchmark already considers.
  2. For every ground-truth word we pick the engine word with the same
     token text whose bbox IoU is highest. Ties broken by centroid
     distance.
  3. A match counts as "accurate" if the tokens are equal; as
     "spatially precise" additionally if IoU >= threshold (default 0.5).

We report: precision, recall, F1 (accuracy), mean IoU over matched
pairs, mean centroid error in PDF points, and the fraction of matches
that clear the IoU>=0.5 bar ("localisation precision").
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "example"
RESULTS = ROOT / "benchmark" / "results"
IGNORE = ROOT / "benchmark" / ".fixtureignore"

LITEPARSE_DIR = os.environ.get("LITEPARSE_DIR", "")
DPI = 150
PT_PER_INCH = 72.0
PX_TO_PT = PT_PER_INCH / DPI
IOU_THRESHOLD = 0.5

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-/_@%'()]*")


@dataclass
class Word:
    text: str
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def iou(a: Word, b: Word) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def centroid_dist(a: Word, b: Word) -> float:
    dx = a.cx - b.cx
    dy = a.cy - b.cy
    return (dx * dx + dy * dy) ** 0.5


def normalise(tok: str) -> str:
    t = tok.strip().lower()
    # Mirror spdf-core::is_ocr_punctuation_noise: drop tokens with no alnum.
    if not any(c.isalnum() for c in t):
        return ""
    return t


def load_ignore() -> set[str]:
    if not IGNORE.is_file():
        return set()
    return {line.strip() for line in IGNORE.read_text().splitlines() if line.strip()}


def find_fixtures() -> list[Path]:
    skip = load_ignore()
    exts = {".pdf", ".jpg", ".jpeg", ".png"}
    out: list[Path] = []
    roots = [EXAMPLE]
    corpus = EXAMPLE / "corpus"
    if corpus.is_dir():
        roots.append(corpus)
    for root in roots:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in exts and p.name not in skip:
                out.append(p)
    return out


def render_first_page(fixture: Path, work: Path) -> Path:
    """Return a PNG that represents the fixture at 150 DPI."""
    if fixture.suffix.lower() == ".pdf":
        # pdftoppm outputs <prefix>-1.png for page 1.
        prefix = work / "page"
        subprocess.run(
            ["pdftoppm", "-r", str(DPI), str(fixture), str(prefix), "-png"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return prefix.with_name("page-1.png")
    return fixture  # already a raster image


# Regex for pdftotext -bbox-layout word entries. Example:
#   <word xMin="72.000000" yMin="56.904000" xMax="95.328000" yMax="67.920000">Form</word>
BBOX_WORD_RE = re.compile(
    r'<word\s+xMin="([\d.]+)"\s+yMin="([\d.]+)"\s+'
    r'xMax="([\d.]+)"\s+yMax="([\d.]+)"[^>]*>(.*?)</word>',
    re.IGNORECASE | re.DOTALL,
)


def pdftotext_ground_truth(pdf: Path) -> list[Word]:
    """Run `pdftotext -bbox-layout -f 1 -l 1` and return per-word boxes in PDF points.

    pdftotext emits an XHTML file with one `<word>` element per extracted
    glyph run, coordinates already in PDF points with top-left origin — the
    same convention spdf uses. This serves as the **born-digital oracle**:
    for PDFs with a real text layer, pdftotext is overwhelmingly correct
    and does not hallucinate the way a raster OCR run does on form pages.
    """
    if pdf.suffix.lower() != ".pdf":
        return []
    with tempfile.NamedTemporaryFile(suffix=".xhtml", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        res = subprocess.run(
            [
                "pdftotext",
                "-bbox-layout",
                "-f",
                "1",
                "-l",
                "1",
                str(pdf),
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0 or not out_path.is_file():
            return []
        xhtml = out_path.read_text(errors="replace")
    finally:
        out_path.unlink(missing_ok=True)

    words: list[Word] = []
    for m in BBOX_WORD_RE.finditer(xhtml):
        x1, y1, x2, y2, raw = m.groups()
        text = normalise(html.unescape(raw))
        if not text:
            continue
        try:
            x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
        except ValueError:
            continue
        w = x2f - x1f
        h = y2f - y1f
        if w <= 0 or h <= 0:
            continue
        words.append(Word(text=text, x=x1f, y=y1f, w=w, h=h))
    return words


def tesseract_ground_truth(image: Path) -> list[Word]:
    """Run tesseract and return per-word boxes in PDF points."""
    res = subprocess.run(
        [
            "tesseract",
            str(image),
            "-",
            "-l",
            "eng",
            "-c",
            f"user_defined_dpi={DPI}",
            "tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    words: list[Word] = []
    for line in res.stdout.splitlines()[1:]:  # skip header
        cols = line.split("\t")
        if len(cols) < 12 or cols[0] != "5":
            continue
        try:
            conf = float(cols[10])
        except ValueError:
            continue
        if conf < 30:
            continue
        text = normalise(cols[11])
        if not text:
            continue
        left, top, width, height = (float(cols[i]) for i in (6, 7, 8, 9))
        if width <= 0 or height <= 0:
            continue
        words.append(
            Word(
                text=text,
                x=left * PX_TO_PT,
                y=top * PX_TO_PT,
                w=width * PX_TO_PT,
                h=height * PX_TO_PT,
            )
        )
    return words


def run_spdf_json(fixture: Path) -> list[Word]:
    res = subprocess.run(
        [
            "spdf",
            "parse",
            str(fixture),
            "--format",
            "json",
            "--ocr-language",
            "eng",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return items_from_json(res.stdout)


def run_lite_json(fixture: Path) -> list[Word]:
    if not LITEPARSE_DIR:
        return []
    entry = Path(LITEPARSE_DIR) / "dist" / "src" / "index.js"
    if not entry.is_file():
        return []
    res = subprocess.run(
        [
            "node",
            str(entry),
            "parse",
            str(fixture),
            "--format",
            "json",
            "--ocr-language",
            "en",
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return []
    return items_from_json(res.stdout)


def items_from_json(raw: str) -> list[Word]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # liteparse prefixes some progress lines to stdout; grab the JSON block.
        start = raw.find("{")
        data = json.loads(raw[start:]) if start != -1 else {}
    page = (data.get("pages") or [{}])[0]
    items = page.get("textItems") or page.get("text_items") or []
    words: list[Word] = []
    for it in items:
        text = normalise(it.get("text", ""))
        if not text:
            continue
        w = float(it.get("width") or 0)
        h = float(it.get("height") or 0)
        if w <= 0 or h <= 0:
            continue
        words.append(
            Word(
                text=text,
                x=float(it.get("x") or 0),
                y=float(it.get("y") or 0),
                w=w,
                h=h,
            )
        )
    return words


def evaluate(engine: list[Word], truth: list[Word]) -> dict:
    """Greedy best-IoU match on identical token text.

    Returns precision/recall/F1 plus spatial stats over the matched pairs.
    """
    used_engine: set[int] = set()
    matched_pairs: list[tuple[Word, Word, float]] = []

    for t in truth:
        best_i = -1
        best_iou = 0.0
        best_dist = float("inf")
        for i, e in enumerate(engine):
            if i in used_engine or e.text != t.text:
                continue
            ov = iou(t, e)
            dist = centroid_dist(t, e)
            # Prefer higher IoU; break ties by centroid distance.
            if ov > best_iou + 1e-9 or (abs(ov - best_iou) < 1e-9 and dist < best_dist):
                best_iou = ov
                best_dist = dist
                best_i = i
        if best_i >= 0:
            used_engine.add(best_i)
            matched_pairs.append((t, engine[best_i], best_iou))

    matches = len(matched_pairs)
    recall = matches / len(truth) if truth else 0.0
    precision = matches / len(engine) if engine else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    ious = [iou_ for _, _, iou_ in matched_pairs]
    dists = [centroid_dist(t, e) for t, e, _ in matched_pairs]
    well_localised = sum(1 for v in ious if v >= IOU_THRESHOLD)

    return {
        "truth_words": len(truth),
        "engine_words": len(engine),
        "matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": statistics.mean(ious) if ious else 0.0,
        "median_iou": statistics.median(ious) if ious else 0.0,
        "mean_centroid_err_pt": statistics.mean(dists) if dists else 0.0,
        "iou_ge_threshold_rate": (well_localised / matches) if matches else 0.0,
        "iou_threshold": IOU_THRESHOLD,
    }


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> int:
    if not shutil.which("spdf"):
        print("spdf missing from PATH — run `make install-ocr`", file=sys.stderr)
        return 1
    if not shutil.which("tesseract") or not shutil.which("pdftoppm"):
        print("tesseract + poppler-utils must be on PATH", file=sys.stderr)
        return 1

    fixtures = find_fixtures()
    if not fixtures:
        print(f"no fixtures under {EXAMPLE}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for fixture in fixtures:
            print(f"== {fixture.name} ==")
            image = render_first_page(fixture, work)
            truth = tesseract_ground_truth(image)
            print(f"  ground truth (tesseract): {len(truth)} words")

            # Second oracle: pdftotext -bbox-layout (born-digital only).
            bbox_truth = pdftotext_ground_truth(fixture)
            if bbox_truth:
                print(f"  ground truth (pdftotext): {len(bbox_truth)} words")

            spdf_words = run_spdf_json(fixture)
            spdf_stats = evaluate(spdf_words, truth)
            print(
                f"  spdf:      f1={pct(spdf_stats['f1'])} mean_iou={spdf_stats['mean_iou']:.3f} "
                f"iou>={IOU_THRESHOLD}: {pct(spdf_stats['iou_ge_threshold_rate'])} "
                f"centroid_err={spdf_stats['mean_centroid_err_pt']:.2f}pt"
            )
            spdf_bbox = evaluate(spdf_words, bbox_truth) if bbox_truth else None
            if spdf_bbox:
                print(f"    vs pdftotext: f1={pct(spdf_bbox['f1'])} mean_iou={spdf_bbox['mean_iou']:.3f}")

            lite_words = run_lite_json(fixture)
            lite_stats = evaluate(lite_words, truth) if lite_words else None
            lite_bbox = evaluate(lite_words, bbox_truth) if (lite_words and bbox_truth) else None
            if lite_stats:
                print(
                    f"  liteparse: f1={pct(lite_stats['f1'])} mean_iou={lite_stats['mean_iou']:.3f} "
                    f"iou>={IOU_THRESHOLD}: {pct(lite_stats['iou_ge_threshold_rate'])} "
                    f"centroid_err={lite_stats['mean_centroid_err_pt']:.2f}pt"
                )
                if lite_bbox:
                    print(f"    vs pdftotext: f1={pct(lite_bbox['f1'])} mean_iou={lite_bbox['mean_iou']:.3f}")
            else:
                print("  liteparse: skipped (set LITEPARSE_DIR)")

            row = {"fixture": fixture.name, "spdf": spdf_stats}
            if lite_stats:
                row["lite"] = lite_stats
            if spdf_bbox:
                row["spdf_vs_pdftotext"] = spdf_bbox
            if lite_bbox:
                row["lite_vs_pdftotext"] = lite_bbox
            rows.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "spatial.json").write_text(json.dumps({"rows": rows}, indent=2))

    def mean(key: str, engine: str) -> float:
        vals = [r[engine][key] for r in rows if engine in r]
        return statistics.mean(vals) if vals else 0.0

    def emit_md() -> str:
        lines: list[str] = []
        lines.append("# Spatial benchmark — spdf vs liteparse\n")
        lines.append(
            "Ground truth: raw tesseract (PSM_AUTO, `user_defined_dpi=150`) run "
            "directly on the rendered image. PDFs are rasterised at 150 DPI via "
            "`pdftoppm`. Pixel bboxes converted to PDF points (× 72/150). "
            "Tokens are lowercased and pure-punctuation noise dropped before "
            "matching.\n"
        )
        lines.append("")
        lines.append(
            "Matching: per ground-truth word, pick the engine word with the "
            "same text and maximum IoU (ties by centroid distance). "
            "`iou≥0.5` gives the localisation-precision rate.\n"
        )
        lines.append("")
        lines.append("## Per-fixture\n")
        lines.append(
            "| fixture | engine | tokens | matched | F1 | mean IoU | median IoU | IoU≥0.5 | centroid err |"
        )
        lines.append(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"
        )
        for r in rows:
            for name, key in (("spdf", "spdf"), ("liteparse", "lite")):
                if key not in r:
                    continue
                s = r[key]
                lines.append(
                    f"| {r['fixture']} | {name} | {s['engine_words']} | {s['matches']} | "
                    f"{pct(s['f1'])} | {s['mean_iou']:.3f} | {s['median_iou']:.3f} | "
                    f"{pct(s['iou_ge_threshold_rate'])} | {s['mean_centroid_err_pt']:.2f} pt |"
                )
        lines.append("")
        lines.append("## Mean over fixtures\n")
        lines.append(
            "| engine | F1 | mean IoU | IoU≥0.5 | centroid err |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| spdf      | {pct(mean('f1','spdf'))} | {mean('mean_iou','spdf'):.3f} | "
            f"{pct(mean('iou_ge_threshold_rate','spdf'))} | {mean('mean_centroid_err_pt','spdf'):.2f} pt |"
        )
        if any("lite" in r for r in rows):
            lines.append(
                f"| liteparse | {pct(mean('f1','lite'))} | {mean('mean_iou','lite'):.3f} | "
                f"{pct(mean('iou_ge_threshold_rate','lite'))} | {mean('mean_centroid_err_pt','lite'):.2f} pt |"
            )

        # Second oracle: pdftotext -bbox-layout (born-digital PDFs only).
        if any("spdf_vs_pdftotext" in r for r in rows):
            lines.append("")
            lines.append("## Vs pdftotext oracle (born-digital PDFs)\n")
            lines.append(
                "Ground truth is `pdftotext -bbox-layout` word boxes. Excludes "
                "raster fixtures and PDFs whose text layer pdftotext cannot "
                "read (e.g. CID fonts with no ToUnicode). Higher is better; "
                "this isolates spatial accuracy on the cases where the PDF "
                "actually has a ground truth.\n"
            )
            lines.append(
                "| fixture | engine | F1 | mean IoU | IoU≥0.5 | centroid err |"
            )
            lines.append("|---|---|---:|---:|---:|---:|")
            for r in rows:
                for name, key in (("spdf", "spdf_vs_pdftotext"), ("liteparse", "lite_vs_pdftotext")):
                    if key not in r:
                        continue
                    s = r[key]
                    lines.append(
                        f"| {r['fixture']} | {name} | {pct(s['f1'])} | {s['mean_iou']:.3f} | "
                        f"{pct(s['iou_ge_threshold_rate'])} | {s['mean_centroid_err_pt']:.2f} pt |"
                    )
            lines.append("")
            lines.append("### Mean (pdftotext oracle)\n")
            lines.append("| engine | F1 | mean IoU | IoU≥0.5 | centroid err |")
            lines.append("|---|---:|---:|---:|---:|")
            lines.append(
                f"| spdf      | {pct(mean('f1','spdf_vs_pdftotext'))} | "
                f"{mean('mean_iou','spdf_vs_pdftotext'):.3f} | "
                f"{pct(mean('iou_ge_threshold_rate','spdf_vs_pdftotext'))} | "
                f"{mean('mean_centroid_err_pt','spdf_vs_pdftotext'):.2f} pt |"
            )
            if any("lite_vs_pdftotext" in r for r in rows):
                lines.append(
                    f"| liteparse | {pct(mean('f1','lite_vs_pdftotext'))} | "
                    f"{mean('mean_iou','lite_vs_pdftotext'):.3f} | "
                    f"{pct(mean('iou_ge_threshold_rate','lite_vs_pdftotext'))} | "
                    f"{mean('mean_centroid_err_pt','lite_vs_pdftotext'):.2f} pt |"
                )
        return "\n".join(lines) + "\n"

    (RESULTS / "spatial.md").write_text(emit_md())
    print(f"\nwrote {RESULTS / 'spatial.json'} + {RESULTS / 'spatial.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
