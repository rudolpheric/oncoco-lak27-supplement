#!/usr/bin/env python3
"""Do the paper's signature buckets survive the classifier's errors?

Section 5.2 does not read single OnCoCo labels. It pools them into three client
buckets: problem disclosure, solution-space engagement and reciprocal requests
(the same rule as oncoco_behavioral_signatures.py, applied to the English label
texts). A confusion inside a bucket, such as Problem statement against Problem
definition, leaves those shares untouched. This script therefore scores the
classifier at bucket level on the two human-annotated conditions, where expert
labels and model predictions sit on identical units.

Reads the unit tables written by the two agreement scripts.
Writes results/tables/oncoco_human_gold_signature_check.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/tables"
LABEL_TEXT = json.loads((Path(__file__).resolve().parent / "label_text_map.json").read_text())
B = 2000


def bucket(code) -> str | None:
    t = LABEL_TEXT.get(code, str(code)).lower()
    if "problem" in t:
        return "Problem disclosure"
    if any(k in t for k in ("recommendation", "resource activation", "implementation")):
        return "Solution-space engagement"
    if "general request" in t:
        return "Reciprocal requests"
    return None


def load() -> dict[str, pd.DataFrame]:
    rp = pd.read_csv(OUT / "oncoco_human_agreement_units.csv")
    re_ = pd.read_csv(OUT / "oncoco_hh_real_agreement_units.csv")
    out = {}
    for name, df in (("HH roleplay", rp), ("HH real", re_)):
        df = df[df.in_corpus].dropna(subset=["gold", "pred"])
        df = df[df.role == "K"].copy()           # the signature is client-side
        df["gold_bucket"] = df.gold.map(bucket)
        df["pred_bucket"] = df.pred.map(bucket)
        out[name] = df
    return out


def share_ci(df: pd.DataFrame, col: str, bkt: str, rng) -> tuple:
    """Bucket share with a conversation-level bootstrap interval."""
    groups = [g for _, g in df.groupby("conversation_id")]
    obs = float((df[col] == bkt).mean())
    boot = []
    for _ in range(B):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        s = pd.concat(pick)
        boot.append(float((s[col] == bkt).mean()))
    return obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    data = load()
    rng = np.random.default_rng(42)
    rows = []
    buckets = ["Problem disclosure", "Solution-space engagement", "Reciprocal requests"]
    for bkt in buckets:
        for cond, df in data.items():
            g = df.gold_bucket == bkt
            p = df.pred_bucket == bkt
            hs, hlo, hhi = share_ci(df, "gold_bucket", bkt, rng)
            ms, mlo, mhi = share_ci(df, "pred_bucket", bkt, rng)
            tp = int((g & p).sum())
            rows.append(dict(
                bucket=bkt, condition=cond, n_client_units=len(df),
                expert_share=hs, expert_lo=hlo, expert_hi=hhi,
                model_share=ms, model_lo=mlo, model_hi=mhi,
                ratio=ms / hs if hs else np.nan,
                bucket_recall=tp / max(int(g.sum()), 1),
                bucket_precision=tp / max(int(p.sum()), 1),
            ))
    t = pd.DataFrame(rows)
    fmt = lambda v: f"{v:.3f}"

    # The signature argument is a between-condition contrast, so score the contrast
    # itself: roleplay share minus real share, once per label source.
    contrasts = []
    for bkt in buckets:
        a, b = data["HH roleplay"], data["HH real"]
        for src, col in (("expert labels", "gold_bucket"), ("model labels", "pred_bucket")):
            d = float((a[col] == bkt).mean()) - float((b[col] == bkt).mean())
            contrasts.append(dict(bucket=bkt, labels=src, roleplay_minus_real=d))
    c = pd.DataFrame(contrasts)

    # Same question for the label distribution as a whole, with the paper's pooling
    # applied: does bucket-level pooling remove the distance attenuation?
    def pooled(df, col):
        return df[col].map(lambda x: bucket(x) or "other").value_counts(normalize=True)
    pool_rows = []
    for src, col in (("expert labels", "gold"), ("model labels", "pred")):
        pa, pb = pooled(data["HH roleplay"], col), pooled(data["HH real"], col)
        idx = sorted(set(pa.index) | set(pb.index))
        tvd = 0.5 * float((pa.reindex(idx, fill_value=0) - pb.reindex(idx, fill_value=0)).abs().sum())
        pool_rows.append(dict(labels=src, tvd_pooled_buckets=tvd))
    p = pd.DataFrame(pool_rows)

    # Alternative cuts of the problem bucket. The paper's rule matches the word
    # "problem" in the label text and therefore keeps Own emotional expression out,
    # although the code tree files it under the same pillar, Analysis and
    # clarification of problems.
    variants = {"PS+PD (paper)": {"CL-IF-ACP-*-PS-*", "CL-IF-ACP-*-PD-*"},
                "PS+PD+OE": {"CL-IF-ACP-*-PS-*", "CL-IF-ACP-*-PD-*", "CL-IF-ACP-*-OE-*"},
                "whole ACP pillar": {l for l in LABEL_TEXT if l.startswith("CL-IF-ACP")}}
    vrows = []
    for name, S in variants.items():
        cell = {}
        for cond, df in data.items():
            g, q = df.gold.isin(S), df.pred.isin(S)
            cell[cond] = (float(g.mean()), float(q.mean()),
                          int((g & q).sum()) / max(int(g.sum()), 1))
        a, b = cell["HH roleplay"], cell["HH real"]
        vrows.append(dict(definition=name,
                          roleplay_expert=a[0], roleplay_model=a[1], roleplay_recall=a[2],
                          real_expert=b[0], real_model=b[1], real_recall=b[2],
                          contrast_expert=a[0] - b[0], contrast_model=a[1] - b[1]))
    v = pd.DataFrame(vrows)
    v.to_csv(OUT / "oncoco_human_gold_problem_bucket_variants.csv", index=False)

    # Why the bucket leaks: unit length. Short units carry no discourse context, and
    # a three-word continuation of a problem narrative reads as agreement on its own.
    bins, names = [0, 3, 6, 12, 25, 10 ** 6], ["1-3", "4-6", "7-12", "13-25", "26+"]
    S = variants["PS+PD (paper)"]
    per = {}
    for cond, df in data.items():
        d = df.copy()
        d["words"] = d.text.astype(str).str.split().str.len()
        g = d[d.gold.isin(S)].copy()
        g["bin"] = pd.cut(g.words, bins, labels=names)
        per[cond] = g.groupby("bin", observed=False).apply(
            lambda x: pd.Series({"n": len(x), "recall": x.pred.isin(S).mean(),
                                 "to_consent_or_other": x.pred.isin(
                                     {"CL-IF-ACP-*-Cons-*", "CL-O-*-*-O-*"}).mean()}))
    ln = pd.concat(per, axis=1)
    ln.columns = [f"{c}_{k}" for c, k in ln.columns]
    ln.to_csv(OUT / "oncoco_human_gold_problem_bucket_by_length.csv")

    # Standardisation: how much of the condition gap in bucket recall is composition
    # (HH real writes in shorter bursts) and how much is the model simply being worse
    # on real-counseling language at every length?
    r_rp = per["HH roleplay"].recall.values
    n_rp = per["HH roleplay"].n.values
    r_re = per["HH real"].recall.values
    n_re = per["HH real"].n.values
    obs_rp = float(np.nansum(r_rp * n_rp) / n_rp.sum())
    obs_re = float(np.nansum(r_re * n_re) / n_re.sum())
    counterfactual = float(np.nansum(r_rp * n_re) / n_re.sum())   # real lengths, roleplay rates
    dec = pd.DataFrame([dict(
        recall_roleplay=obs_rp, recall_real=obs_re, gap=obs_rp - obs_re,
        real_with_roleplay_rates=counterfactual,
        share_from_length_composition=(obs_rp - counterfactual) / (obs_rp - obs_re),
        share_from_rate_difference=(counterfactual - obs_re) / (obs_rp - obs_re))])
    dec.to_csv(OUT / "oncoco_human_gold_problem_bucket_decomposition.csv", index=False)

    print()
    print(v.to_string(index=False, float_format=fmt))
    print()
    print(ln.to_string(float_format=fmt))
    print()
    print(dec.T.to_string(header=False, float_format=fmt))

    print(t.to_string(index=False, float_format=fmt))
    print()
    print(c.to_string(index=False, float_format=fmt))
    print()
    print(p.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
