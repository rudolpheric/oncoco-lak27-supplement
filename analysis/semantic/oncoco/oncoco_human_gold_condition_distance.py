#!/usr/bin/env python3
"""Does the classifier's error change the between-condition distance?

Both human-annotated corpora are counseling conditions the paper compares:
HH-roleplay (student practice) and HH-real (live counseling). For every
human-annotated unit we hold two labels, the expert code and the classifier's
prediction on that same unit. The distance between the two conditions can
therefore be computed twice from identical units, once from human labels and
once from model labels. If the classifier's errors are shared across conditions
they cancel and the two distances agree, which is the assumption the paper's
between-condition readings rest on.

The pipeline's own measurement (SaT spans, model labels) is reported alongside
as the third row, since that is what the paper actually computes.

Inputs are the unit tables written by
    oncoco_human_annotation_agreement.py --reclassify
    oncoco_hh_real_annotation_agreement.py --reclassify

Output: results/tables/oncoco_human_gold_condition_distance.csv
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/tables"
CLS = ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"
B = 2000


def js(p, q) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / (b[a > 0] + 1e-12))))
    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def counts(df: pd.DataFrame, col: str, index: dict) -> np.ndarray:
    """Per-conversation label count matrix."""
    m = np.zeros((df.conversation_id.nunique(), len(index)))
    for i, (_, g) in enumerate(df.groupby("conversation_id")):
        for lab, n in g[col].value_counts().items():
            if lab in index:
                m[i, index[lab]] += n
    return m


def distance(a: pd.DataFrame, b: pd.DataFrame, col: str, rng) -> tuple:
    labels = sorted(set(a[col].dropna()) | set(b[col].dropna()))
    idx = {l: i for i, l in enumerate(labels)}
    A, Bm = counts(a, col, idx), counts(b, col, idx)
    d = js(A.sum(0), Bm.sum(0))
    boot = np.array([js(A[rng.integers(0, A.shape[0], A.shape[0])].sum(0),
                        Bm[rng.integers(0, Bm.shape[0], Bm.shape[0])].sum(0))
                     for _ in range(B)])
    return d, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def pipeline_spans() -> pd.DataFrame:
    rows = json.loads(CLS.read_text(encoding="utf-8"))
    recs = []
    for r in rows:
        cond = r.get("source")
        if cond not in ("HH_real_chat", "HH_roleplay_chat"):
            continue
        role = {"Counsellor": "B", "Client": "K"}.get(r.get("speaker_type"))
        for s in r.get("sentence_classification") or []:
            recs.append(dict(condition=cond, role=role,
                             conversation_id=f'{r["source_file"]}#{r["id"]}',
                             pred=s["predicted_label"]))
    return pd.DataFrame(recs)


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, rng) -> tuple:
    """Bootstrap the difference expert-label JSD minus model-label JSD.

    The same conversation resample feeds both label sources, so the sampling
    noise the two share cancels and the interval speaks to the difference only.
    """
    labels = sorted(set(a.gold) | set(b.gold) | set(a.pred) | set(b.pred))
    idx = {l: i for i, l in enumerate(labels)}
    Ag, Bg = counts(a, "gold", idx), counts(b, "gold", idx)
    Ap, Bp = counts(a, "pred", idx), counts(b, "pred", idx)
    obs = js(Ag.sum(0), Bg.sum(0)) - js(Ap.sum(0), Bp.sum(0))
    boot = []
    for _ in range(B):
        i = rng.integers(0, Ag.shape[0], Ag.shape[0])
        j = rng.integers(0, Bg.shape[0], Bg.shape[0])
        boot.append(js(Ag[i].sum(0), Bg[j].sum(0)) - js(Ap[i].sum(0), Bp[j].sum(0)))
    boot = np.array(boot)
    return obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    rp = pd.read_csv(OUT / "oncoco_human_agreement_units.csv")
    re_ = pd.read_csv(OUT / "oncoco_hh_real_agreement_units.csv")
    rp = rp[rp.in_corpus].dropna(subset=["gold", "pred"])
    re_ = re_[re_.in_corpus].dropna(subset=["gold", "pred"])
    spans = pipeline_spans()

    rng = np.random.default_rng(42)
    rows = []
    for role, name in ((None, "all"), ("B", "counselor"), ("K", "client")):
        a = rp if role is None else rp[rp.role == role]
        b = re_ if role is None else re_[re_.role == role]
        sa = spans[spans.condition == "HH_roleplay_chat"]
        sb = spans[spans.condition == "HH_real_chat"]
        if role is not None:
            sa, sb = sa[sa.role == role], sb[sb.role == role]
        for label, col, x, y in (("human units, expert labels", "gold", a, b),
                                 ("human units, model labels", "pred", a, b),
                                 ("pipeline SaT spans, model labels", "pred", sa, sb)):
            d, lo, hi = distance(x, y, col, rng)
            rows.append(dict(side=name, labels=label, jsd=d, ci_lo=lo, ci_hi=hi,
                             n_roleplay=len(x), n_real=len(y)))
    # Within-corpus contrasts. Both sides come from one annotation team and one
    # codebook, so a gap between the expert-label and model-label distance here
    # cannot be an artifact of comparing two annotation cultures.
    within = []
    re_["region"] = re_.doc.str.split("_").str[0]
    for x, y in (("KA", "OÖ"), ("KA", "W"), ("OÖ", "W")):
        a, b = re_[re_.region == x], re_[re_.region == y]
        for label, col in (("expert labels", "gold"), ("model labels", "pred")):
            d, lo, hi = distance(a, b, col, rng)
            within.append(dict(contrast=f"HH-real {x} vs {y}", labels=label, jsd=d,
                               ci_lo=lo, ci_hi=hi, n_a=len(a), n_b=len(b)))
    rp["half"] = rp.conversation_id.astype(int) % 2
    for corpus, df in (("HH-roleplay", rp), ("HH-real", re_)):
        key = "conversation_id"
        convs = sorted(df[key].unique())
        h = set(convs[::2])
        a, b = df[df[key].isin(h)], df[~df[key].isin(h)]
        for label, col in (("expert labels", "gold"), ("model labels", "pred")):
            d, lo, hi = distance(a, b, col, rng)
            within.append(dict(contrast=f"{corpus} split half", labels=label, jsd=d,
                               ci_lo=lo, ci_hi=hi, n_a=len(a), n_b=len(b)))
    deltas = []
    for role, name in ((None, "all"), ("B", "counselor"), ("K", "client")):
        a = rp if role is None else rp[rp.role == role]
        b = re_ if role is None else re_[re_.role == role]
        d, lo, hi = paired_delta(a, b, rng)
        deltas.append(dict(contrast="HH-roleplay vs HH-real", side=name,
                           delta_expert_minus_model=d, ci_lo=lo, ci_hi=hi))
    for x, y in (("KA", "OÖ"), ("KA", "W"), ("OÖ", "W")):
        d, lo, hi = paired_delta(re_[re_.region == x], re_[re_.region == y], rng)
        deltas.append(dict(contrast=f"HH-real {x} vs {y}", side="all",
                           delta_expert_minus_model=d, ci_lo=lo, ci_hi=hi))
    dl = pd.DataFrame(deltas)
    dl.to_csv(OUT / "oncoco_human_gold_distance_attenuation.csv", index=False)

    w = pd.DataFrame(within)
    w.to_csv(OUT / "oncoco_human_gold_within_corpus_distance.csv", index=False)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "oncoco_human_gold_condition_distance.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(w.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(dl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
