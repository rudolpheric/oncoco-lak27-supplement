#!/usr/bin/env python3
"""Test mechanism hypothesis: client uncertainty vs counselor procedural behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversation-level mechanism analysis.")
    parser.add_argument(
        "--data-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="Path to OnCoCo sentence classification JSON.",
    )
    parser.add_argument(
        "--out-conv-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_mechanism_conversation_metrics.csv"),
        help="Conversation-level output CSV.",
    )
    parser.add_argument(
        "--out-summary-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_mechanism_summary.csv"),
        help="Condition-level summary CSV.",
    )
    parser.add_argument(
        "--out-contrast-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_mechanism_contrasts.csv"),
        help="Pairwise condition contrasts CSV.",
    )
    parser.add_argument(
        "--out-summary-tex",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_mechanism_summary.tex"),
        help="Condition-level summary TeX table.",
    )
    parser.add_argument(
        "--out-contrast-tex",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_mechanism_contrasts.tex"),
        help="Contrast TeX table.",
    )
    parser.add_argument(
        "--out-figure",
        default=str(PROJECT_ROOT / "results" / "figures" / "oncoco" / "mechanism_entropy_moderation_chat.png"),
        help="Output scatter figure for chat conditions.",
    )
    parser.add_argument(
        "--min-spans-per-role",
        type=int,
        default=8,
        help="Minimum number of spans for both client and counselor per conversation.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def entropy_from_counts(counts: pd.Series) -> float:
    probs = counts / max(float(counts.sum()), 1.0)
    probs = probs[probs > 0]
    if probs.empty:
        return float("nan")
    return float(-(probs * np.log2(probs)).sum())


def build_sentence_df(rows: List[Dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        cond = r.get("source")
        conv_id = r.get("id")
        speaker = r.get("speaker_type")
        if cond is None or conv_id is None or speaker is None:
            continue
        # (source_file, id): ids repeat across raw exports, and a merged conversation would
        # pool two clients' label distributions into one entropy value.
        conv_id = f"{r.get('source_file', '')}::{conv_id}"
        for s in r.get("sentence_classification", []) or []:
            label = s.get("predicted_label")
            if not label:
                continue
            recs.append(
                {
                    "condition": cond,
                    "conversation_id": str(conv_id),
                    "speaker_type": speaker,
                    "label": label,
                }
            )
    return pd.DataFrame.from_records(recs)


def compute_conversation_metrics(df: pd.DataFrame, min_spans_per_role: int) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["condition", "conversation_id"])
    for (condition, conv_id), sub in grouped:
        client = sub[sub["speaker_type"] == "Client"]
        counselor = sub[sub["speaker_type"].isin(["Counsellor", "Counselor"])]
        n_client = int(len(client))
        n_counselor = int(len(counselor))
        if n_client < min_spans_per_role or n_counselor < min_spans_per_role:
            continue

        client_entropy = entropy_from_counts(client["label"].value_counts())
        counselor_counts = counselor["label"].value_counts()
        counselor_total = max(float(counselor_counts.sum()), 1.0)
        moderation = float(counselor_counts.get("CO-Mod-*-*-*-*", 0.0) / counselor_total)
        complex_reflection = float(counselor_counts.get("CO-IF-AC-RE-RCR-*", 0.0) / counselor_total)
        simple_reflection = float(counselor_counts.get("CO-IF-AC-RF-SRx-*", 0.0) / counselor_total)

        rows.append(
            {
                "condition": condition,
                "conversation_id": conv_id,
                "n_client_spans": n_client,
                "n_counselor_spans": n_counselor,
                "client_entropy": client_entropy,
                "counselor_moderation_share": moderation,
                "counselor_complex_reflection_share": complex_reflection,
                "counselor_simple_reflection_share": simple_reflection,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_condition(conv_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, sub in conv_df.groupby("condition"):
        rho_mod, p_mod = spearmanr(sub["client_entropy"], sub["counselor_moderation_share"])
        rho_refl, p_refl = spearmanr(sub["client_entropy"], sub["counselor_complex_reflection_share"])
        rows.append(
            {
                "condition": condition,
                "n_conversations": int(len(sub)),
                "client_entropy_mean": float(sub["client_entropy"].mean()),
                "counselor_moderation_mean": float(sub["counselor_moderation_share"].mean()),
                "counselor_complex_reflection_mean": float(sub["counselor_complex_reflection_share"].mean()),
                "rho_entropy_vs_moderation": float(rho_mod),
                "p_entropy_vs_moderation": float(p_mod),
                "rho_entropy_vs_complex_reflection": float(rho_refl),
                "p_entropy_vs_complex_reflection": float(p_refl),
            }
        )
    return pd.DataFrame(rows).sort_values("condition")


def compute_contrasts(conv_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("HH_roleplay_chat", "H_LLM_roleplay_chat", "chat roleplay vs H--LLM"),
        ("HH_roleplay_mail", "HH_CAIA_mail", "mail roleplay vs CAIA"),
    ]
    rows = []
    for a, b, name in pairs:
        sub_a = conv_df[conv_df["condition"] == a]
        sub_b = conv_df[conv_df["condition"] == b]
        if sub_a.empty or sub_b.empty:
            continue
        for metric in [
            "client_entropy",
            "counselor_moderation_share",
            "counselor_complex_reflection_share",
        ]:
            stat, p = mannwhitneyu(sub_a[metric], sub_b[metric], alternative="two-sided")
            rows.append(
                {
                    "contrast": name,
                    "condition_a": a,
                    "condition_b": b,
                    "metric": metric,
                    "mean_a": float(sub_a[metric].mean()),
                    "mean_b": float(sub_b[metric].mean()),
                    "delta_b_minus_a": float(sub_b[metric].mean() - sub_a[metric].mean()),
                    "mannwhitney_u": float(stat),
                    "p_value": float(p),
                }
            )
    return pd.DataFrame(rows)


def write_tex_summary(summary: pd.DataFrame, out_path: Path) -> None:
    name_map = {
        "HH_roleplay_chat": "HH roleplay chat",
        "HH_real_chat": "HH real chat",
        "H_LLM_roleplay_chat": "H--LLM roleplay chat",
        "HH_roleplay_mail": "HH roleplay mail",
        "HH_real_mail": "HH real mail",
        "HH_CAIA_mail": "HH+assistant mail",
        "LLM_LLM_mail": "LLM--LLM mail",
    }
    lines = [
        r"\begin{tabular}{p{0.32\linewidth}rrrr}",
        r"\toprule",
        # raw string: "$\\rho$" in a normal string reaches LaTeX as a line break plus "rho"
        r"Condition & $n$ conv. & Entropy & Moderation & $\rho$(Ent,Mod) \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{name_map.get(row['condition'], row['condition'])} & "
            f"{int(row['n_conversations'])} & "
            f"{row['client_entropy_mean']:.2f} & "
            f"{row['counselor_moderation_mean']:.3f} & "
            f"{row['rho_entropy_vs_moderation']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_contrast(contrast_df: pd.DataFrame, out_path: Path) -> None:
    pretty_metric = {
        "client_entropy": "Client entropy",
        "counselor_moderation_share": "Counselor moderation",
        "counselor_complex_reflection_share": "Counselor complex reflection",
    }
    lines = [
        "\\begin{tabular}{p{0.30\\linewidth}p{0.27\\linewidth}rr}",
        "\\toprule",
        "Contrast & Metric & $\\Delta$ (B-A) & $p$ \\\\",
        "\\midrule",
    ]
    for _, row in contrast_df.iterrows():
        p = row["p_value"]
        p_str = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(
            f"{row['contrast']} & "
            f"{pretty_metric.get(row['metric'], row['metric'])} & "
            f"{row['delta_b_minus_a']:.3f} & "
            f"{p_str} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_chat_scatter(conv_df: pd.DataFrame, out_path: Path) -> None:
    keep = conv_df[conv_df["condition"].isin(["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat"])].copy()
    if keep.empty:
        return
    name_map = {
        "HH_roleplay_chat": "HH roleplay chat",
        "HH_real_chat": "HH real chat",
        "H_LLM_roleplay_chat": "H--LLM roleplay chat",
    }
    keep["condition_pretty"] = keep["condition"].map(name_map)
    sns.set_theme(style="whitegrid")
    g = sns.lmplot(
        data=keep,
        x="client_entropy",
        y="counselor_moderation_share",
        hue="condition_pretty",
        height=4.2,
        aspect=1.5,
        scatter_kws={"alpha": 0.55, "s": 32},
        line_kws={"linewidth": 2.0},
        ci=None,
    )
    g.set_axis_labels("Client label entropy (conversation-level)", "Counselor moderation share")
    g.fig.suptitle("Mechanism probe: client uncertainty vs counselor moderation (chat)", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(g.fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.data_json))
    sent_df = build_sentence_df(rows)
    conv_df = compute_conversation_metrics(sent_df, min_spans_per_role=args.min_spans_per_role)
    summary = summarize_by_condition(conv_df)
    contrasts = compute_contrasts(conv_df)

    out_conv = Path(args.out_conv_csv)
    out_summary = Path(args.out_summary_csv)
    out_contrast = Path(args.out_contrast_csv)
    out_conv.parent.mkdir(parents=True, exist_ok=True)
    conv_df.to_csv(out_conv, index=False)
    summary.to_csv(out_summary, index=False)
    contrasts.to_csv(out_contrast, index=False)

    write_tex_summary(summary, Path(args.out_summary_tex))
    write_tex_contrast(contrasts, Path(args.out_contrast_tex))
    plot_chat_scatter(conv_df, Path(args.out_figure))

    print(f"Wrote {out_conv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_contrast}")
    print(f"Wrote {args.out_summary_tex}")
    print(f"Wrote {args.out_contrast_tex}")
    print(f"Wrote {args.out_figure}")


if __name__ == "__main__":
    main()
