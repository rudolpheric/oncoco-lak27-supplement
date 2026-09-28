#!/usr/bin/env python3
"""One validation table for the OnCoCo classifier against both human annotations.

Pulls the per-unit results the two agreement scripts write and reports, per
condition and speaker role: agreement, Cohen's kappa with a conversation-level
bootstrap interval, macro and weighted F1, the majority-class baseline, and
accuracy at the coarser levels of the code (role, pillar, function group).

Writes results/tables/oncoco_human_validation.csv and .tex
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/tables"
B = 2000


def kappa(gold, pred) -> float:
    labels = sorted(set(gold) | set(pred))
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(gold, pred):
        m[idx[x], idx[y]] += 1
    n = len(gold)
    if n == 0:
        return float("nan")
    po = np.trace(m) / n
    pe = float((m.sum(0) / n) @ (m.sum(1) / n))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def f1_scores(gold: pd.Series, pred: pd.Series) -> tuple[float, float]:
    labels = sorted(set(gold) | set(pred))
    f1s, sup = [], []
    for lab in labels:
        tp = int(((gold == lab) & (pred == lab)).sum())
        fp = int(((gold != lab) & (pred == lab)).sum())
        fn = int(((gold == lab) & (pred != lab)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        sup.append(tp + fn)
    f1s, sup = np.array(f1s), np.array(sup)
    present = sup > 0                      # macro over the categories the humans used
    macro = float(f1s[present].mean()) if present.any() else float("nan")
    weighted = float((f1s * sup).sum() / sup.sum()) if sup.sum() else float("nan")
    return macro, weighted


def tier(code, depth) -> str:
    return "-".join(str(code).split("-")[:depth])


def row(name: str, df: pd.DataFrame, rng) -> dict:
    gold, pred = df.gold, df.pred
    groups = [g for _, g in df.groupby("conversation_id")]
    boot = []
    for _ in range(B):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boot.append(kappa(s.gold, s.pred))
    macro, weighted = f1_scores(gold, pred)
    return dict(
        block=name, units=len(df), conversations=df.conversation_id.nunique(),
        categories_used=int(gold.nunique()),
        accuracy=float((gold == pred).mean()),
        kappa=kappa(gold, pred),
        kappa_lo=float(np.nanpercentile(boot, 2.5)),
        kappa_hi=float(np.nanpercentile(boot, 97.5)),
        macro_f1=macro, weighted_f1=weighted,
        majority_baseline=float(gold.value_counts(normalize=True).max()),
        acc_pillar=float((gold.map(lambda c: tier(c, 2)) == pred.map(lambda c: tier(c, 2))).mean()),
        acc_group=float((gold.map(lambda c: tier(c, 3)) == pred.map(lambda c: tier(c, 3))).mean()),
        wrong_role=float((gold.map(lambda c: tier(c, 1)) != pred.map(lambda c: tier(c, 1))).mean()),
        median_words=float(df.text.astype(str).str.split().str.len().median()),
    )


def main() -> None:
    rp = pd.read_csv(OUT / "oncoco_human_agreement_units.csv")
    re_ = pd.read_csv(OUT / "oncoco_hh_real_agreement_units.csv")
    rp = rp[rp.in_corpus].dropna(subset=["gold", "pred"])
    re_ = re_[re_.in_corpus].dropna(subset=["gold", "pred"])

    rng = np.random.default_rng(42)
    rows = []
    for cond, df in (("HH roleplay", rp), ("HH real", re_)):
        rows.append(row(f"{cond} — all units", df, rng))
        for r, label in (("B", "counselor"), ("K", "client")):
            rows.append(row(f"{cond} — {label}", df[df.role == r], rng))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "oncoco_human_validation.csv", index=False)

    cols = [("block", "Condition and role", "{}"), ("units", "Units", "{:,.0f}"),
            ("categories_used", "Cat.", "{:.0f}"), ("median_words", "Words", "{:.0f}"),
            ("accuracy", "Acc.", "{:.3f}"), ("majority_baseline", "Base", "{:.3f}"),
            ("kappa", "$\\kappa$", "{:.3f}"), ("macro_f1", "Macro F1", "{:.3f}"),
            ("weighted_f1", "Wtd. F1", "{:.3f}"), ("acc_pillar", "Pillar", "{:.3f}"),
            ("acc_group", "Group", "{:.3f}")]
    caption = (
        "Agreement of the OnCoCo classifier with the expert annotation of the two "
        "human chat corpora. Both label sets sit on identical units: the classifier "
        "was re-run on the human segments, with the role prefix the pipeline uses, so "
        "only the labels differ and not the segmentation. "
        "\\emph{Units} counts annotated segments, \\emph{Cat.} the number of distinct "
        "OnCoCo categories the human annotators used in that block (the classifier can "
        "choose among 68, of which 66 have a counterpart in the German scheme), and "
        "\\emph{Words} their median length in whitespace-separated tokens. "
        "\\emph{Acc.} is unit-level agreement and \\emph{Base} the share of the largest "
        "human category, the accuracy a constant prediction would reach. "
        "$\\kappa$ is Cohen's kappa with a 95\\% conversation-level bootstrap interval "
        "(2{,}000 draws). \\emph{Macro F1} averages over the categories the humans used, so "
        "rare ones weigh as much as frequent ones, while \\emph{Wtd. F1} weighs by support. "
        "\\emph{Pillar} and \\emph{Group} are accuracies after truncating both codes to their "
        "second and third field, that is to the broad field of action and to the "
        "function group within it."
    )
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", "\\hline",
             " & ".join(h for _, h, _ in cols) + " \\\\", "\\hline"]
    for _, r in t.iterrows():
        cells = []
        for key, _, f in cols:
            cells.append(f.format(r[key]) if key != "block" else str(r[key]).replace("—", "--"))
        cells[6] += f" \\footnotesize[{r.kappa_lo:.3f}, {r.kappa_hi:.3f}]"
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}",
              "\\caption{" + caption + "}",
              "\\label{Stab:human-validation}", "\\end{table}"]
    (OUT / "oncoco_human_validation.tex").write_text("\n".join(lines), encoding="utf-8")

    # the CSV carries no caption, so ship the column glossary beside it
    glossary = [
        ("block", "condition and speaker role"),
        ("units", "annotated segments in the block"),
        ("conversations", "conversations they come from"),
        ("categories_used", "distinct OnCoCo categories the human annotators used "
                            "(the classifier may choose among 68, 66 of which exist in the German scheme)"),
        ("median_words", "median unit length in whitespace-separated tokens"),
        ("accuracy", "unit-level agreement between expert code and model prediction"),
        ("majority_baseline", "share of the largest human category, what a constant prediction would reach"),
        ("kappa", "Cohen's kappa"),
        ("kappa_lo/kappa_hi", "95% conversation-level bootstrap interval, 2000 draws"),
        ("macro_f1", "F1 averaged over the categories the humans used, unweighted"),
        ("weighted_f1", "F1 averaged over the same categories, weighted by support"),
        ("acc_pillar", "accuracy after truncating both codes to the second field (broad field of action)"),
        ("acc_group", "accuracy after truncating both codes to the third field (function group)"),
        ("wrong_role", "share of predictions taken from the other speaker's inventory"),
    ]
    (OUT / "oncoco_human_validation_columns.txt").write_text(
        "Columns of oncoco_human_validation.csv\n\n"
        + "\n".join(f"{k:20s} {v}" for k, v in glossary) + "\n", encoding="utf-8")

    show = t[["block", "units", "conversations", "categories_used", "median_words",
              "accuracy", "majority_baseline", "kappa", "kappa_lo", "kappa_hi",
              "macro_f1", "weighted_f1", "acc_pillar", "acc_group", "wrong_role"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
