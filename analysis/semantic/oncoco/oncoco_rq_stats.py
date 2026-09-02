#!/usr/bin/env python3
"""
Compute RQ-specific stats (distribution distances, effect sizes, model realness)
from OnCoCo-labeled sentence data.
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "oncoco"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def explode_sentences(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        for sent in r.get("sentence_classification", []) or []:
            model = r.get("model") or "unknown"
            records.append(
                {
                    "condition": r.get("source"),
                    "speaker_type": r.get("speaker_type"),
                    "label": sent.get("predicted_label"),
                    "model": model,
                    "modality": r.get("modality"),
                }
            )
    df = pd.DataFrame.from_records(records)
    df = df.dropna(subset=["condition", "speaker_type", "label"])
    return df


def label_distribution(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    counts = df.groupby(group_cols + ["label"]).size().reset_index(name="count")
    totals = df.groupby(group_cols).size().reset_index(name="total")
    merged = counts.merge(totals, on=group_cols, how="left")
    merged["proportion"] = merged["count"] / merged["total"]
    return merged


def align_distributions(dist_a: pd.DataFrame, dist_b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = sorted(set(dist_a["label"]).union(set(dist_b["label"])))
    p = np.zeros(len(labels))
    q = np.zeros(len(labels))
    a_map = dict(zip(dist_a["label"], dist_a["proportion"]))
    b_map = dict(zip(dist_b["label"], dist_b["proportion"]))
    for i, lab in enumerate(labels):
        p[i] = a_map.get(lab, 0.0)
        q[i] = b_map.get(lab, 0.0)
    return p, q, labels


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon distance
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12)))
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return sqrt(max(jsd, 0.0))


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))


def l2_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.sum((p - q) ** 2)))


def cohen_h(p1: float, p2: float) -> float:
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def compute_pairwise(dist: pd.DataFrame, cond_a: str, cond_b: str, speaker: str) -> Dict[str, object]:
    a = dist[(dist["condition"] == cond_a) & (dist["speaker_type"] == speaker)]
    b = dist[(dist["condition"] == cond_b) & (dist["speaker_type"] == speaker)]
    # An absent condition used to yield two all-zero vectors, which js_distance turns into a
    # perfectly respectable 0.000. A missing condition must not look like a perfect match.
    if a.empty or b.empty:
        raise SystemExit(
            f"no rows for {cond_a if a.empty else cond_b} / {speaker} in the label distribution; "
            "the artifact does not carry this condition, so this comparison cannot be computed.")
    p, q, labels = align_distributions(a, b)
    return {
        "comparison": f"{cond_a} vs {cond_b}",
        "speaker_type": speaker,
        "jsd": js_distance(p, q),
        "l1": l1_distance(p, q),
        "l2": l2_distance(p, q),
    }


def compute_label_effects(dist: pd.DataFrame, cond_a: str, cond_b: str, speaker: str) -> pd.DataFrame:
    a = dist[(dist["condition"] == cond_a) & (dist["speaker_type"] == speaker)]
    b = dist[(dist["condition"] == cond_b) & (dist["speaker_type"] == speaker)]
    p, q, labels = align_distributions(a, b)
    effects = pd.DataFrame({
        "label": labels,
        "p_a": p,
        "p_b": q,
        "diff": p - q,
        "cohen_h": [cohen_h(pa, pb) for pa, pb in zip(p, q)],
    })
    effects["comparison"] = f"{cond_a} vs {cond_b}"
    effects["speaker_type"] = speaker
    return effects


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    rows = load_rows(DATA_PATH)
    df = explode_sentences(rows)

    dist = label_distribution(df, ["condition", "speaker_type"])
    # Persist it: the label-distribution tables of the supplement are built from this file, and
    # for a long time it had no writer, so it silently kept the numbers of an earlier corpus.
    dist.to_csv(TABLES_DIR / "oncoco_label_distribution.csv", index=False)

    # RQ1 comparisons
    comparisons = [
        ("HH_roleplay_chat", "H_LLM_roleplay_chat"),
        ("HH_real_chat", "HH_roleplay_chat"),
        ("HH_real_chat", "H_LLM_roleplay_chat"),
        ("HH_real_mail", "HH_roleplay_mail"),
    ]

    distance_rows = []
    effects_rows = []
    for cond_a, cond_b in comparisons:
        for speaker in sorted(dist["speaker_type"].unique()):
            distance_rows.append(compute_pairwise(dist, cond_a, cond_b, speaker))
            effects_rows.append(compute_label_effects(dist, cond_a, cond_b, speaker))

    pd.DataFrame(distance_rows).to_csv(TABLES_DIR / "oncoco_rq1_distances.csv", index=False)
    pd.concat(effects_rows, ignore_index=True).to_csv(TABLES_DIR / "oncoco_rq1_label_effects.csv", index=False)

    # RQ1b model realness (chat only)
    chat_df = df[df["condition"] == "H_LLM_roleplay_chat"].copy()
    chat_dist = label_distribution(chat_df, ["model", "speaker_type"])
    baseline_condition = "HH_real_chat" if (dist["condition"] == "HH_real_chat").any() else "HH_roleplay_chat"
    baseline = dist[dist["condition"] == baseline_condition]
    model_rows = []
    for model in sorted(chat_dist["model"].dropna().unique()):
        for speaker in sorted(chat_dist["speaker_type"].unique()):
            a = chat_dist[(chat_dist["model"] == model) & (chat_dist["speaker_type"] == speaker)]
            b = baseline[baseline["speaker_type"] == speaker]
            p, q, _ = align_distributions(a, b)
            model_rows.append(
                {
                    "model": model,
                    "speaker_type": speaker,
                    "jsd": js_distance(p, q),
                    "l1": l1_distance(p, q),
                    "l2": l2_distance(p, q),
                }
            )

    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(TABLES_DIR / "oncoco_rq1b_model_realness.csv", index=False)

    # Plot model realness (JSD)
    if not model_df.empty:
        plt.figure(figsize=(6.8, 5.4))
        sns.barplot(data=model_df, x="model", y="jsd", hue="speaker_type")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"LLM Realness (distance to {baseline_condition.replace('_', ' ')}) – JSD")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "model_realness_chat_jsd.png", dpi=200)
        plt.close()

    # RQ5 CAIA effect (mail, counsellor)
    caia_dist_rows = []
    caia_effects = []
    for speaker in ["Counsellor"]:
        caia_dist_rows.append(
            compute_pairwise(dist, "HH_CAIA_mail", "HH_roleplay_mail", speaker)
        )
        caia_effects.append(
            compute_label_effects(dist, "HH_CAIA_mail", "HH_roleplay_mail", speaker)
        )

    pd.DataFrame(caia_dist_rows).to_csv(TABLES_DIR / "oncoco_rq5_caia_distances.csv", index=False)
    caia_effects_df = pd.concat(caia_effects, ignore_index=True)
    caia_effects_df.to_csv(TABLES_DIR / "oncoco_rq5_caia_label_effects.csv", index=False)

    # Plot top absolute effects for CAIA vs HH roleplay (counsellor)
    if not caia_effects_df.empty:
        plot_df = caia_effects_df.copy()
        plot_df["abs_diff"] = plot_df["diff"].abs()
        plot_df = plot_df.sort_values("abs_diff", ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=plot_df, x="diff", y="label")
        plt.title("Top CAIA vs HH roleplay label shifts (Counsellor)")
        plt.xlabel("Difference in proportion (CAIA - HH roleplay)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "rq5_caia_top_label_shifts.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
