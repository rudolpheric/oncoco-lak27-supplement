#!/usr/bin/env python3
"""Assess within-condition variability and persistence of label distributions."""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Within-condition variability analysis for OnCoCo labels.")
    parser.add_argument(
        "--data-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="Path to OnCoCo sentence classification JSON.",
    )
    parser.add_argument(
        "--label-map",
        default=str(PROJECT_ROOT / "analysis" / "semantic" / "oncoco" / "label_text_map.json"),
        help="Mapping from label codes to human-readable names.",
    )
    parser.add_argument(
        "--out-jsd",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_within_condition_jsd.csv"),
        help="Output: within-condition JSD summary.",
    )
    parser.add_argument(
        "--out-jsd-per-conv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_within_condition_jsd_per_conversation.csv"),
        help="Output: per-conversation JSD values.",
    )
    parser.add_argument(
        "--out-weighted",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_weighted_vs_unweighted_jsd.csv"),
        help="Output: weighted vs unweighted distribution JSD.",
    )
    parser.add_argument(
        "--out-persistence",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_persistence_top_labels.csv"),
        help="Output: persistence of top label shifts by comparison.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top labels per comparison/speaker.",
    )
    parser.add_argument(
        "--out-figure-client",
        default=str(PROJECT_ROOT / "results" / "figures" / "oncoco" / "within_jsd_client.png"),
        help="Output: violin plot for client JSDs.",
    )
    parser.add_argument(
        "--out-figure-counselor",
        default=str(PROJECT_ROOT / "results" / "figures" / "oncoco" / "within_jsd_counselor.png"),
        help="Output: violin plot for counselor JSDs.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12)))

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return sqrt(max(jsd, 0.0))


def label_to_text(code: str, label_map: Dict[str, str]) -> str:
    name = label_map.get(code)
    if not name:
        base = "Unmapped category"
        if "-O-" in code:
            base = "Other statements"
        elif "-Mod-" in code:
            base = "Moderation"
        elif "-FA-" in code:
            base = "Formalities at the beginning"
        suffix = ""
        if code.endswith("PI-*"):
            suffix = " (PI)"
        elif code.endswith("ST-*"):
            suffix = " (ST)"
        speaker = "Counselor" if code.startswith("CO-") else "Client" if code.startswith("CL-") else ""
        return f"{speaker}: {base}{suffix}" if speaker else f"{base}{suffix}"
    speaker = "Counselor" if code.startswith("CO-") else "Client" if code.startswith("CL-") else ""
    return f"{speaker}: {name}" if speaker else name


def prepare_sentence_df(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        condition = r.get("source")
        speaker_type = r.get("speaker_type")
        # (source_file, id): ids repeat across raw exports, and merging two conversations
        # into one would both understate the conversation count and average their
        # distributions before the within-condition JSD is taken.
        conv_id = f"{r.get('source_file', '')}::{r.get('msg_learn_counselling_id') or r.get('id')}"
        for sent in r.get("sentence_classification", []) or []:
            label = sent.get("predicted_label")
            if not (condition and speaker_type and label and conv_id is not None):
                continue
            records.append(
                {
                    "condition": condition,
                    "speaker_type": speaker_type,
                    "conv_id": str(conv_id),
                    "label": label,
                }
            )
    df = pd.DataFrame.from_records(records)
    return df.dropna(subset=["condition", "speaker_type", "conv_id", "label"])


def conversation_distributions(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["condition", "speaker_type", "conv_id", "label"]).size().reset_index(name="count")
    totals = counts.groupby(["condition", "speaker_type", "conv_id"]).agg(total=("count", "sum")).reset_index()
    merged = counts.merge(totals, on=["condition", "speaker_type", "conv_id"], how="left")
    merged["proportion"] = merged["count"] / merged["total"]
    return merged


def condition_weighted_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["condition", "speaker_type", "label"]).size().reset_index(name="count")
    totals = counts.groupby(["condition", "speaker_type"]).agg(total=("count", "sum")).reset_index()
    merged = counts.merge(totals, on=["condition", "speaker_type"], how="left")
    merged["proportion"] = merged["count"] / merged["total"]
    return merged


def condition_unweighted_distribution(conv_dist: pd.DataFrame) -> pd.DataFrame:
    # conv_dist carries no zero rows, so a plain groupby-mean averages each label only over
    # the conversations that happen to use it. That inflates every proportion and leaves the
    # reference vector summing well above 1. Densify per (condition, speaker) so conversations
    # that never use a label contribute an explicit 0.0.
    frames = []
    for (condition, speaker), group in conv_dist.groupby(["condition", "speaker_type"]):
        dense = group.pivot_table(index="conv_id", columns="label", values="proportion",
                                  aggfunc="mean", fill_value=0.0)
        means = dense.mean(axis=0)
        frames.append(pd.DataFrame({
            "condition": condition,
            "speaker_type": speaker,
            "label": means.index.to_numpy(),
            "proportion": means.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def align_dist(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    labels = sorted(set(a["label"]).union(set(b["label"])))
    amap = dict(zip(a["label"], a["proportion"]))
    bmap = dict(zip(b["label"], b["proportion"]))
    p = np.array([amap.get(l, 0.0) for l in labels])
    q = np.array([bmap.get(l, 0.0) for l in labels])
    return p, q


def summary_within_jsd(conv_dist: pd.DataFrame, cond_mean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    per_conv_rows = []
    for (condition, speaker), group in conv_dist.groupby(["condition", "speaker_type"]):
        mean_dist = cond_mean[(cond_mean["condition"] == condition) & (cond_mean["speaker_type"] == speaker)]
        conv_jsd = []
        for conv_id, sub in group.groupby("conv_id"):
            p, q = align_dist(sub, mean_dist)
            jsd_val = js_distance(p, q)
            conv_jsd.append(jsd_val)
            per_conv_rows.append(
                {
                    "condition": condition,
                    "speaker_type": speaker,
                    "conv_id": conv_id,
                    "jsd": jsd_val,
                }
            )
        if not conv_jsd:
            continue
        conv_jsd = np.array(conv_jsd)
        q1 = np.quantile(conv_jsd, 0.25)
        q3 = np.quantile(conv_jsd, 0.75)
        iqr = q3 - q1
        outlier_thresh = q3 + 1.5 * iqr
        rows.append(
            {
                "condition": condition,
                "speaker_type": speaker,
                "n_conversations": len(conv_jsd),
                "mean_jsd": float(conv_jsd.mean()),
                "median_jsd": float(np.median(conv_jsd)),
                "p90_jsd": float(np.quantile(conv_jsd, 0.90)),
                "max_jsd": float(conv_jsd.max()),
                "outlier_thresh": float(outlier_thresh),
                "outlier_count": int((conv_jsd > outlier_thresh).sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(per_conv_rows)


def weighted_vs_unweighted_jsd(weighted: pd.DataFrame, unweighted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, speaker), group in weighted.groupby(["condition", "speaker_type"]):
        w = group
        u = unweighted[(unweighted["condition"] == condition) & (unweighted["speaker_type"] == speaker)]
        p, q = align_dist(w, u)
        rows.append({"condition": condition, "speaker_type": speaker, "jsd": js_distance(p, q)})
    return pd.DataFrame(rows)


def auc_probability(a: np.ndarray, b: np.ndarray) -> float:
    """Probability that a random element of a is greater than random element of b."""
    n_a = len(a)
    n_b = len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    # rank-sum based U
    combined = np.concatenate([a, b])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    ranks_a = ranks[:n_a]
    u = ranks_a.sum() - n_a * (n_a + 1) / 2
    return float(u / (n_a * n_b))


def _label_series(group: pd.DataFrame, label: str) -> np.ndarray:
    """Per-conversation proportion of `label` across every conversation in `group`.

    Conversations that never use the label contribute 0.0 instead of being dropped.
    Selecting on the label first silently removes them, which shifts the median towards
    the conversations that happen to use the label and distorts the AUC.
    """
    convs = pd.Index(group["conv_id"].unique())
    present = group[group["label"] == label].groupby("conv_id")["proportion"].mean()
    return present.reindex(convs, fill_value=0.0).to_numpy()


def compute_persistence(
    conv_dist: pd.DataFrame,
    comparisons: List[Tuple[str, str]],
    top_labels: Dict[Tuple[str, str, str], List[str]],
    label_map: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for (cond_a, cond_b) in comparisons:
        for speaker in sorted(conv_dist["speaker_type"].unique()):
            for label in top_labels.get((cond_a, cond_b, speaker), []):
                a = conv_dist[(conv_dist["condition"] == cond_a) & (conv_dist["speaker_type"] == speaker)]
                b = conv_dist[(conv_dist["condition"] == cond_b) & (conv_dist["speaker_type"] == speaker)]
                a_vals = _label_series(a, label)
                b_vals = _label_series(b, label)
                if len(a_vals) == 0 or len(b_vals) == 0:
                    continue
                med_a = float(np.median(a_vals))
                med_b = float(np.median(b_vals))
                auc = auc_probability(a_vals, b_vals)
                rows.append(
                    {
                        "comparison": f"{cond_a} vs {cond_b}",
                        "speaker_type": speaker,
                        "label": label,
                        "label_text": label_to_text(label, label_map),
                        "median_a": med_a,
                        "median_b": med_b,
                        "median_diff": med_a - med_b,
                        "auc": auc,
                        "p_a_gt_b_median": float(np.mean(a_vals > med_b)),
                        "p_b_lt_a_median": float(np.mean(b_vals < med_a)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.data_json))
    df = prepare_sentence_df(rows)
    conv_dist = conversation_distributions(df)
    weighted = condition_weighted_distribution(df)
    unweighted = condition_unweighted_distribution(conv_dist)

    within_jsd, per_conv = summary_within_jsd(conv_dist, unweighted)
    Path(args.out_jsd).parent.mkdir(parents=True, exist_ok=True)
    within_jsd.to_csv(args.out_jsd, index=False)
    Path(args.out_jsd_per_conv).parent.mkdir(parents=True, exist_ok=True)
    per_conv.to_csv(args.out_jsd_per_conv, index=False)

    wvuw = weighted_vs_unweighted_jsd(weighted, unweighted)
    Path(args.out_weighted).parent.mkdir(parents=True, exist_ok=True)
    wvuw.to_csv(args.out_weighted, index=False)

    # Top labels from RQ1 comparisons
    rq1_path = PROJECT_ROOT / "results" / "tables" / "oncoco_rq1_label_effects.csv"
    rq1 = pd.read_csv(rq1_path)
    comparisons = [
        ("HH_roleplay_chat", "H_LLM_roleplay_chat"),
        ("HH_real_chat", "HH_roleplay_chat"),
        ("HH_real_mail", "HH_roleplay_mail"),
    ]
    top_labels: Dict[Tuple[str, str, str], List[str]] = {}
    for cond_a, cond_b in comparisons:
        comp = f"{cond_a} vs {cond_b}"
        sub = rq1[rq1["comparison"] == comp].copy()
        for speaker in sub["speaker_type"].unique():
            sub_s = sub[sub["speaker_type"] == speaker].copy()
            if speaker == "Client":
                sub_s = sub_s[sub_s["label"].str.startswith("CL-")]
            elif speaker == "Counsellor":
                sub_s = sub_s[sub_s["label"].str.startswith("CO-")]
            sub_s["abs_diff"] = sub_s["diff"].abs()
            top = sub_s.sort_values("abs_diff", ascending=False).head(args.top_k)
            top_labels[(cond_a, cond_b, speaker)] = top["label"].tolist()

    label_map = json.loads(Path(args.label_map).read_text(encoding="utf-8")) if Path(args.label_map).exists() else {}
    persistence = compute_persistence(conv_dist, comparisons, top_labels, label_map)
    Path(args.out_persistence).parent.mkdir(parents=True, exist_ok=True)
    persistence.to_csv(args.out_persistence, index=False)

    # Violin plots for within-condition JSD distributions
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        for speaker, out_path in [
            ("Client", args.out_figure_client),
            ("Counsellor", args.out_figure_counselor),
        ]:
            sub = per_conv[per_conv["speaker_type"] == speaker].copy()
            if sub.empty:
                continue
            plt.figure(figsize=(9, 4))
            sns.violinplot(
                data=sub,
                x="condition",
                y="jsd",
                inner="quartile",
                cut=0,
                linewidth=1,
            )
            plt.xticks(rotation=25, ha="right")
            plt.ylabel("JSD to condition mean")
            plt.xlabel("Condition")
            plt.title(f"Within-condition variability (JSD) - {speaker}")
            plt.tight_layout()
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=200)
            plt.close()
    except Exception:
        pass

    print(f"Wrote {args.out_jsd}")
    print(f"Wrote {args.out_jsd_per_conv}")
    print(f"Wrote {args.out_weighted}")
    print(f"Wrote {args.out_persistence}")


if __name__ == "__main__":
    main()
