"""Regenerate website/results.json from bench/results/latest.json.

    python website/build_data.py

The page reads only results.json, so the numbers on the site are always the
accepted scoreboard run (same win/tie/loss bands as bench/scoreboard.py).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import scoreboard  # noqa: E402

run = json.loads((ROOT / "bench" / "results" / "latest.json").read_text(encoding="utf-8"))
cmp = scoreboard.compare_engines(run)


def strip(engine: dict) -> dict:
    out = {}
    for suite, vals in engine.items():
        out[suite] = {k: v for k, v in vals.items() if k not in ("per_doc", "cases")}
        if "cases" in vals:
            out[suite]["case_count"] = len(vals["cases"])
    return out


data = {
    "timestamp": run["timestamp"],
    "commit": run["commit"],
    "repeats": run["repeats"],
    "measurement": run["measurement"],
    "versions": run["versions"],
    "engines": {name: strip(e) for name, e in run["engines"].items()},
    "tally": {"wins": cmp["wins"], "ties": cmp["ties"], "losses": cmp["losses"],
              "total": len(cmp["rows"]), "beats_liteparse": cmp["beats_liteparse"]},
    "rows": cmp["rows"],
}
out = Path(__file__).resolve().parent / "results.json"
out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
print(f"wrote {out.name}: {cmp['wins']}W/{cmp['ties']}T/{cmp['losses']}L over {len(cmp['rows'])} metrics")
