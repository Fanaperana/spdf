#!/usr/bin/env python3
import sys, re
from collections import Counter

def tokens(path):
    text = open(path, 'r', encoding='utf-8', errors='replace').read()
    # Split into word tokens, preserving digits/punct runs
    toks = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?|[^\sA-Za-z\d]", text)
    return [t.lower() for t in toks]

def compare(ref_path, cand_path, label):
    ref = tokens(ref_path)
    cand = tokens(cand_path)
    rc = Counter(ref); cc = Counter(cand)
    common = sum((rc & cc).values())
    total = sum(rc.values())
    recall = common / total if total else 0
    prec = common / sum(cc.values()) if cc else 0
    f1 = 2*recall*prec/(recall+prec) if (recall+prec) else 0
    print(f"{label}: recall={recall:.3f} precision={prec:.3f} f1={f1:.3f} "
          f"(ref_tokens={total}, cand_tokens={sum(cc.values())}, common={common})")
    # Top missing tokens
    missing = rc - cc
    extra = cc - rc
    if missing:
        print(f"  missing top: {missing.most_common(10)}")
    if extra:
        print(f"  extra top:   {extra.most_common(10)}")

parity = "/Users/pralambomanarivo/Desktop/REPO/ocr/spdf/tests/parity"
compare(f"{parity}/lit_tax_output.txt", f"{parity}/spdf_tax_output.txt", "TAX")
compare(f"{parity}/lit_prc_output.txt", f"{parity}/spdf_prc_output.txt", "PRC")
