#!/usr/bin/env python3
"""Two small LaTeX tables for Supplementary Section S7 (human validation on the study corpora).

  results/tables/oncoco_human_validation_categories.tex
      precision / recall / F1 of the classifier per signature category and per human corpus,
      next to the share of client units the experts and the classifier assign to it
  results/tables/oncoco_human_validation_buckets.tex
      the three client-side signature buckets of Section 5.2, expert share vs classifier share

Inputs are the per-label agreement tables of oncoco_human_annotation_agreement.py (HH roleplay)
and oncoco_hh_real_annotation_agreement.py (HH real), and oncoco_human_gold_signature_check.py.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "results/tables"

CATS = [("CL-IF-ACP-*-PS-*", "Problem statement"), ("CL-IF-ACP-*-PD-*", "Problem definition"),
        ("CL-IF-ACP-*-Cons-*", "Consent"), ("CL-IF-ACP-*-Rej-*", "Rejection"),
        ("CL-IF-ACP-*-Req-*", "General request"), ("CL-O-*-*-O-*", "Other statements (client)"),
        ("CO-O-*-*-O-*", "Other statements (counselor)"), ("CO-O-*-*-UCO-*", "Inappropriate remark (counselor)")]


def per_label(path):
    return {r["label"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}


def f(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


rp, re_ = per_label(T / "oncoco_human_agreement_per_label.csv"), per_label(T / "oncoco_hh_real_agreement_per_label.csv")
lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r"& \multicolumn{5}{c}{HH roleplay} & \multicolumn{5}{c}{HH real} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
         r"Category & Exp. & Model & P & R & F1 & Exp. & Model & P & R & F1 \\", r"\midrule"]
for code, name in CATS:
    a, b = rp.get(code, {}), re_.get(code, {})
    lines.append(f"{name} & {f(a.get('human_share'),3)} & {f(a.get('model_share'),3)} & {f(a.get('precision'))} & {f(a.get('recall'))} & {f(a.get('f1'))} & "
                 f"{f(b.get('human_share'),3)} & {f(b.get('model_share'),3)} & {f(b.get('precision'))} & {f(b.get('recall'))} & {f(b.get('f1'))} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "oncoco_human_validation_categories.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

rows = list(csv.DictReader(open(T / "oncoco_human_gold_signature_check.csv", encoding="utf-8")))
lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Bucket & Condition & Expert share [95\% CI] & Model share [95\% CI] & Ratio & Recall & Precision \\", r"\midrule"]
for r in rows:
    lines.append(f"{r['bucket']} & {r['condition']} & {f(r['expert_share'],3)} [{f(r['expert_lo'],3)}, {f(r['expert_hi'],3)}] & "
                 f"{f(r['model_share'],3)} [{f(r['model_lo'],3)}, {f(r['model_hi'],3)}] & {f(r['ratio'])} & {f(r['bucket_recall'])} & {f(r['bucket_precision'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "oncoco_human_validation_buckets.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", T / "oncoco_human_validation_categories.tex", "and", T / "oncoco_human_validation_buckets.tex")
