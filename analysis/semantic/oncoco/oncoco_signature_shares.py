#!/usr/bin/env python3
"""Conversation-level client signature shares for the three chat conditions.

Section 5.2 of the paper and the panel-2 captions of the hero figure quote
per-conversation shares -- "at least one General request", "writes at least one
question mark", "contains no Rejection at all" -- that had no generator: they were
computed by hand and then lived as literals in the prose and in make_hero_tikz.py.
A corpus change could not reach them. This script derives them from the same master
JSON every other result reads, so they follow the corpus like everything else.

Shares are over conversations that carry at least one classified client span, which
is the population the client-side analyses run on.

Output: results/tables/oncoco_signature_shares.csv
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[3]
NORM = ROOT / "data/processed/combined/normalized"
OUT = ROOT / "results/tables/oncoco_signature_shares.csv"

CONDS = ["HH_real_chat", "HH_roleplay_chat", "H_LLM_roleplay_chat"]

# Section 5.2 also quotes the negative/positive client-feedback ratio. Disagreement is
# split across two labels in OnCoCo -- an explicit rejection and a negative verdict on a
# concrete recommendation -- so both count as negative.
NEG_FEEDBACK = ["CL-IF-HP-*-NegFR-*", "CL-IF-ACP-*-Rej-*"]
POS_FEEDBACK = ["CL-IF-HP-*-PosF-*", "CL-IF-HP-*-PosFR-*"]
LABEL_MAP = json.load(open(ROOT / "analysis/semantic/oncoco/label_text_map.json"))


def _is(code: str, needle: str) -> bool:
    return needle in LABEL_MAP.get(code, code).lower()


def feedback_ratio(rows, cond, model=None):
    """Negative-to-positive client feedback, over pooled spans."""
    neg = pos = 0
    for r in rows:
        if r.get("source") != cond or r.get("speaker_type") != "Client":
            continue
        if model is not None and r.get("model") != model:
            continue
        for s in r.get("sentence_classification") or []:
            lab = s.get("predicted_label")
            neg += lab in NEG_FEEDBACK
            pos += lab in POS_FEEDBACK
    return neg, pos


def collect(rows, cond, model=None):
    """conversation -> which signatures it shows, over its client spans."""
    conv = defaultdict(lambda: dict(request=False, question=False, rejection=False))
    for r in rows:
        if r.get("source") != cond:
            continue
        if model is not None and r.get("model") != model:
            continue
        if r.get("speaker_type") != "Client":
            continue
        spans = r.get("sentence_classification") or []
        if not spans:
            continue
        key = f"{r.get('source_file')}::{r.get('id')}"
        conv[key]  # a conversation counts as soon as it has one client span, hit or not
        if "?" in (r.get("msg_content") or ""):
            conv[key]["question"] = True
        for s in spans:
            lab = s.get("predicted_label") or ""
            if _is(lab, "general request"):
                conv[key]["request"] = True
            if _is(lab, "rejection"):
                conv[key]["rejection"] = True
    return conv


def main() -> None:
    rows = json.load(open(NORM / "oncoco_classification_all.json"))
    models = sorted({r.get("model") for r in rows
                     if r.get("source") == "H_LLM_roleplay_chat" and r.get("model")})

    groups = [(c, None) for c in CONDS] + [("H_LLM_roleplay_chat", m) for m in models]
    stats = {}
    for cond, model in groups:
        conv = collect(rows, cond, model)
        n = len(conv)
        neg, pos = feedback_ratio(rows, cond, model)
        stats[(cond, model)] = dict(
            n=n,
            request=sum(v["request"] for v in conv.values()),
            question=sum(v["question"] for v in conv.values()),
            no_rejection=sum(not v["rejection"] for v in conv.values()),
            neg_feedback=neg, pos_feedback=pos,
        )

    # Fisher's exact against HH real, the reference for every claim in Section 5.2
    ref = stats[("HH_real_chat", None)]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "model", "n_conversations",
                    "n_with_request", "share_with_request",
                    "n_with_question_mark", "share_with_question_mark",
                    "n_without_rejection", "share_without_rejection",
                    "n_negative_feedback", "n_positive_feedback", "neg_to_pos_feedback",
                    "p_request_vs_hh_real", "p_question_vs_hh_real"])
        for (cond, model), d in stats.items():
            n = d["n"]
            if (cond, model) == ("HH_real_chat", None):
                p_req = p_q = ""
            else:
                p_req = f"{fisher_exact([[d['request'], n - d['request']], [ref['request'], ref['n'] - ref['request']]]).pvalue:.6f}"
                p_q = f"{fisher_exact([[d['question'], n - d['question']], [ref['question'], ref['n'] - ref['question']]]).pvalue:.6f}"
            w.writerow([cond, model or "", n,
                        d["request"], f"{d['request'] / n:.4f}",
                        d["question"], f"{d['question'] / n:.4f}",
                        d["no_rejection"], f"{d['no_rejection'] / n:.4f}",
                        d["neg_feedback"], d["pos_feedback"],
                        f"{d['neg_feedback'] / d['pos_feedback']:.4f}" if d["pos_feedback"] else "",
                        p_req, p_q])
            print(f"{cond:22s} {model or '':14s} n={n:3d}  request={d['request']/n:.3f} "
                  f"question={d['question']/n:.3f} no-rejection={d['no_rejection']/n:.3f} "
                  f"neg/pos={d['neg_feedback'] / d['pos_feedback']:.3f}" if d["pos_feedback"] else "")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
