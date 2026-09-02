#!/usr/bin/env python3
"""Analyze H-LLM evolution against HH roleplay and HH real chat baselines."""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H-LLM evolution analysis.")
    parser.add_argument(
        "--data-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="Path to OnCoCo sentence classification JSON.",
    )
    parser.add_argument(
        "--out-model-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_hllm_model_realness_both_baselines.csv"),
        help="Output CSV for model-level realness to both baselines.",
    )
    parser.add_argument(
        "--out-monthly-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_hllm_monthly_realness.csv"),
        help="Output CSV for monthly realness trajectories.",
    )
    parser.add_argument(
        "--out-figure",
        default=str(PROJECT_ROOT / "results" / "figures" / "oncoco" / "hllm_evolution_monthly_jsd.png"),
        help="Output figure for monthly realness trajectories.",
    )
    parser.add_argument(
        "--min-monthly-sentences",
        type=int,
        default=500,
        help="Minimum sentence count per (month,speaker) to include in monthly trend.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    text = str(value)
    if "." in text and ":" in text and "T" not in text:
        dt = pd.to_datetime(text, errors="coerce", utc=True, dayfirst=True)
    else:
        dt = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(dt):
        dt = pd.to_datetime(text, errors="coerce", utc=True, dayfirst=True)
    if pd.isna(dt):
        return None
    return dt


def explode_sentences(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        condition = r.get("source")
        speaker_type = r.get("speaker_type")
        model = r.get("model") or "unknown"
        timestamp = r.get("msg_created_at") or r.get("timestamp")
        month = None
        if timestamp:
            dt = parse_timestamp(timestamp)
            if dt is not None:
                month = dt.tz_convert(None).to_period("M").strftime("%Y-%m")
        for sent in r.get("sentence_classification", []) or []:
            label = sent.get("predicted_label")
            if not (condition and speaker_type and label):
                continue
            records.append(
                {
                    "condition": condition,
                    "speaker_type": speaker_type,
                    "model": model,
                    "month": month,
                    "label": label,
                }
            )
    return pd.DataFrame.from_records(records)


def label_distribution(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    counts = df.groupby(group_cols + ["label"]).size().reset_index(name="count")
    totals = counts.groupby(group_cols)["count"].sum().reset_index(name="total")
    merged = counts.merge(totals, on=group_cols, how="left")
    merged["proportion"] = merged["count"] / merged["total"]
    return merged


def align_distributions(dist_a: pd.DataFrame, dist_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = sorted(set(dist_a["label"]).union(set(dist_b["label"])))
    a_map = dict(zip(dist_a["label"], dist_a["proportion"]))
    b_map = dict(zip(dist_b["label"], dist_b["proportion"]))
    p = np.array([a_map.get(label, 0.0) for label in labels])
    q = np.array([b_map.get(label, 0.0) for label in labels])
    return p, q


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12)))

    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def compute_model_realness(df: pd.DataFrame) -> pd.DataFrame:
    baselines = ["HH_real_chat", "HH_roleplay_chat"]
    baseline_dist = label_distribution(df[df["condition"].isin(baselines)], ["condition", "speaker_type"])
    hllm = df[df["condition"] == "H_LLM_roleplay_chat"].copy()
    model_dist = label_distribution(hllm, ["model", "speaker_type"])

    # first seen timestamp proxy by month in the sentence-level data
    model_first_seen = (
        hllm.dropna(subset=["month"])
        .groupby("model")["month"]
        .min()
        .to_dict()
    )

    rows = []
    for model in sorted(model_dist["model"].unique()):
        for speaker in sorted(model_dist["speaker_type"].unique()):
            dist_model = model_dist[(model_dist["model"] == model) & (model_dist["speaker_type"] == speaker)]
            if dist_model.empty:
                continue
            out_row = {
                "model": model,
                "speaker_type": speaker,
                "first_seen_month": model_first_seen.get(model, ""),
            }
            for baseline in baselines:
                dist_base = baseline_dist[
                    (baseline_dist["condition"] == baseline) & (baseline_dist["speaker_type"] == speaker)
                ]
                if dist_base.empty:
                    continue
                p, q = align_distributions(dist_model, dist_base)
                out_row[f"jsd_to_{baseline}"] = js_distance(p, q)
            rows.append(out_row)

    out = pd.DataFrame(rows)
    return out.sort_values(["speaker_type", "jsd_to_HH_real_chat", "model"])


def compute_monthly_realness(df: pd.DataFrame, min_monthly_sentences: int) -> pd.DataFrame:
    baselines = ["HH_real_chat", "HH_roleplay_chat"]
    baseline_dist = label_distribution(df[df["condition"].isin(baselines)], ["condition", "speaker_type"])
    hllm = df[(df["condition"] == "H_LLM_roleplay_chat") & df["month"].notna()].copy()
    monthly_dist = label_distribution(hllm, ["month", "speaker_type"])

    rows = []
    for (month, speaker), dist_m in monthly_dist.groupby(["month", "speaker_type"]):
        n_sentences = int(dist_m["count"].sum())
        if n_sentences < min_monthly_sentences:
            continue
        for baseline in baselines:
            dist_b = baseline_dist[
                (baseline_dist["condition"] == baseline) & (baseline_dist["speaker_type"] == speaker)
            ]
            if dist_b.empty:
                continue
            p, q = align_distributions(dist_m, dist_b)
            rows.append(
                {
                    "month": month,
                    "speaker_type": speaker,
                    "baseline": baseline,
                    "n_sentences": n_sentences,
                    "jsd": js_distance(p, q),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["speaker_type", "baseline", "month"])


def plot_monthly_realness(monthly: pd.DataFrame, out_path: Path) -> None:
    if monthly.empty:
        return
    pretty_baseline = {
        "HH_real_chat": "HH real chat",
        "HH_roleplay_chat": "HH roleplay chat",
    }
    pretty_speaker = {"Client": "Client", "Counsellor": "Counselor"}
    plot_df = monthly.copy()
    plot_df["baseline"] = plot_df["baseline"].map(pretty_baseline).fillna(plot_df["baseline"])
    plot_df["speaker_type"] = plot_df["speaker_type"].map(pretty_speaker).fillna(plot_df["speaker_type"])

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, speaker in zip(axes, ["Client", "Counselor"]):
        sub = plot_df[plot_df["speaker_type"] == speaker]
        sns.lineplot(
            data=sub,
            x="month",
            y="jsd",
            hue="baseline",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"{speaker} (H--LLM monthly)")
        ax.set_xlabel("Month")
        ax.set_ylabel("JSD")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Baseline", fontsize="small")
    fig.suptitle("H--LLM realness over time (monthly JSD to human baselines)", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.data_json))
    df = explode_sentences(rows)

    model_realness = compute_model_realness(df)
    out_model = Path(args.out_model_csv)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model_realness.to_csv(out_model, index=False)

    monthly_realness = compute_monthly_realness(df, args.min_monthly_sentences)
    out_month = Path(args.out_monthly_csv)
    out_month.parent.mkdir(parents=True, exist_ok=True)
    monthly_realness.to_csv(out_month, index=False)

    plot_monthly_realness(monthly_realness, Path(args.out_figure))

    print(f"Wrote {out_model}")
    print(f"Wrote {out_month}")
    print(f"Wrote {args.out_figure}")


if __name__ == "__main__":
    main()
