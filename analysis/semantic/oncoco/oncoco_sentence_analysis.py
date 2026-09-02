#!/usr/bin/env python3
"""
Sentence-level analysis for OnCoCo-labeled data.
Creates label distribution tables and temporal decile plots by condition and role.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "oncoco"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def explode_sentences(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        sentences = r.get("sentence_classification", []) or []
        for sent in sentences:
            records.append(
                {
                    "conversation_id": r.get("id"),
                    "condition": r.get("source"),
                    "modality": r.get("modality"),
                    "dataset": r.get("dataset"),
                    "model": r.get("model"),
                    "speaker_type": r.get("speaker_type"),
                    "msg_message_number": int(r.get("msg_message_number") or 0),
                    "sentence_index": int(sent.get("sentence_index") or 0),
                    "label": sent.get("predicted_label"),
                }
            )
    df = pd.DataFrame.from_records(records)
    df = df.dropna(subset=["label", "speaker_type", "condition"])
    return df


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["condition", "speaker_type", "label"]).size().reset_index(name="count")
    )
    totals = (
        df.groupby(["condition", "speaker_type"]).size().reset_index(name="total")
    )
    merged = counts.merge(totals, on=["condition", "speaker_type"], how="left")
    merged["proportion"] = merged["count"] / merged["total"]
    return merged


def plot_label_distribution(dist: pd.DataFrame, role: str, top_n: int = 10) -> None:
    role_df = dist[dist["speaker_type"] == role].copy()
    top_labels = (
        role_df.groupby("label")["count"].sum().sort_values(ascending=False).head(top_n).index
    )
    plot_df = role_df[role_df["label"].isin(top_labels)]
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="label",
        y="proportion",
        hue="condition",
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} labels by condition ({role})")
    plt.tight_layout()
    out = FIGURES_DIR / f"label_distribution_{role.lower()}.png"
    plt.savefig(out, dpi=200)
    plt.close()


def add_deciles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["conversation_id", "speaker_type", "msg_message_number", "sentence_index"])
    df["sentence_order"] = df.groupby(["conversation_id", "speaker_type"]).cumcount()
    counts = df.groupby(["conversation_id", "speaker_type"])["sentence_order"].max().reset_index()
    counts["n_sentences"] = counts["sentence_order"] + 1
    df = df.merge(counts[["conversation_id", "speaker_type", "n_sentences"]], on=["conversation_id", "speaker_type"], how="left")
    df["rel_pos"] = df.apply(
        lambda row: 0.0 if row["n_sentences"] <= 1 else row["sentence_order"] / (row["n_sentences"] - 1),
        axis=1,
    )
    df["decile"] = (df["rel_pos"] * 10).astype(int).clip(0, 9)
    return df


def temporal_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["condition", "speaker_type", "decile", "label"]).size().reset_index(name="count")
    )
    totals = (
        df.groupby(["condition", "speaker_type", "decile"]).size().reset_index(name="total")
    )
    merged = counts.merge(totals, on=["condition", "speaker_type", "decile"], how="left")
    merged["proportion"] = merged["count"] / merged["total"]
    return merged


def plot_temporal(dist: pd.DataFrame, role: str, condition: str, top_n: int = 5) -> None:
    role_df = dist[(dist["speaker_type"] == role) & (dist["condition"] == condition)].copy()
    top_labels = (
        role_df.groupby("label")["count"].sum().sort_values(ascending=False).head(top_n).index
    )
    plot_df = role_df[role_df["label"].isin(top_labels)]
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="decile",
        y="proportion",
        hue="label",
        marker="o",
    )
    plt.title(f"Temporal label distribution ({role}, {condition})")
    plt.xlabel("Conversation decile")
    plt.ylabel("Proportion")
    plt.xticks(range(0, 10))
    plt.tight_layout()
    safe_condition = condition.replace("/", "_")
    out = FIGURES_DIR / f"temporal_{role.lower()}_{safe_condition}.png"
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    rows = load_rows(DATA_PATH)
    df = explode_sentences(rows)

    dist = label_distribution(df)
    dist.to_csv(TABLES_DIR / "oncoco_label_distribution.csv", index=False)

    for role in sorted(dist["speaker_type"].unique()):
        plot_label_distribution(dist, role)

    df_dec = add_deciles(df)
    temporal = temporal_distribution(df_dec)
    temporal.to_csv(TABLES_DIR / "oncoco_temporal_distribution.csv", index=False)

    for role in sorted(temporal["speaker_type"].unique()):
        for condition in sorted(temporal["condition"].unique()):
            plot_temporal(temporal, role, condition)

    summary = SUMMARY_DIR / "oncoco_sentence_analysis.md"
    summary.write_text(
        "# OnCoCo sentence analysis\n\n"
        f"Rows (sentences): {len(df)}\n\n"
        "Outputs:\n"
        "- results/tables/oncoco_label_distribution.csv\n"
        "- results/tables/oncoco_temporal_distribution.csv\n"
        "- results/figures/oncoco/label_distribution_*.png\n"
        "- results/figures/oncoco/temporal_*_*.png\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
