#!/usr/bin/env python3
"""Create a compact triadic chat comparison figure for the main RQ1 result."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot triadic chat JSD comparison.")
    parser.add_argument(
        "--rq1-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_rq1_distances.csv"),
        help="CSV with RQ1 pairwise JSD distances.",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "results" / "figures" / "oncoco" / "main_contribution_triad_jsd.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def parse_comparison(text: str) -> Tuple[str, str]:
    parts = [p.strip() for p in str(text).split(" vs ")]
    if len(parts) != 2:
        raise ValueError(f"Could not parse comparison: {text}")
    return parts[0], parts[1]


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def load_triad_jsd(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    df = pd.read_csv(path)
    needed = {
        canonical_pair("HH_roleplay_chat", "HH_real_chat"),
        canonical_pair("HH_roleplay_chat", "H_LLM_roleplay_chat"),
        canonical_pair("HH_real_chat", "H_LLM_roleplay_chat"),
    }
    out: Dict[Tuple[str, str], Dict[str, float]] = {}

    for _, row in df.iterrows():
        a, b = parse_comparison(row["comparison"])
        key = canonical_pair(a, b)
        if key not in needed:
            continue
        speaker = str(row["speaker_type"]).strip()
        jsd = float(row["jsd"])
        out.setdefault(key, {})
        if speaker == "Counsellor":
            out[key]["Counselor"] = jsd
        else:
            out[key][speaker] = jsd

    missing = [k for k in needed if k not in out or "Client" not in out[k] or "Counselor" not in out[k]]
    if missing:
        raise ValueError(f"Missing triad distances in {path}: {missing}")
    return out


def plot_triad(jsd_map: Dict[Tuple[str, str], Dict[str, float]], out_path: Path) -> None:
    coords = {
        "HH_roleplay_chat": np.array([-1.05, -0.75]),
        "HH_real_chat": np.array([1.05, -0.75]),
        "H_LLM_roleplay_chat": np.array([0.0, 0.95]),
    }
    labels = {
        "HH_roleplay_chat": "HH roleplay\nchat",
        "HH_real_chat": "HH real\nchat",
        "H_LLM_roleplay_chat": "H--LLM\nroleplay chat",
    }
    colors = {
        "HH_roleplay_chat": "#4E79A7",
        "HH_real_chat": "#59A14F",
        "H_LLM_roleplay_chat": "#E15759",
    }

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    pairs = [
        ("HH_roleplay_chat", "HH_real_chat"),
        ("HH_roleplay_chat", "H_LLM_roleplay_chat"),
        ("HH_real_chat", "H_LLM_roleplay_chat"),
    ]
    edge_offsets = {
        canonical_pair("HH_roleplay_chat", "HH_real_chat"): np.array([0.0, -0.08]),
        canonical_pair("HH_roleplay_chat", "H_LLM_roleplay_chat"): np.array([-0.06, 0.04]),
        canonical_pair("HH_real_chat", "H_LLM_roleplay_chat"): np.array([0.06, 0.04]),
    }

    avg_vals = []
    for a, b in pairs:
        key = canonical_pair(a, b)
        vals = jsd_map[key]
        avg_vals.append((vals["Client"] + vals["Counselor"]) / 2.0)
    vmin, vmax = min(avg_vals), max(avg_vals)
    denom = max(vmax - vmin, 1e-9)

    for a, b in pairs:
        key = canonical_pair(a, b)
        vals = jsd_map[key]
        p1, p2 = coords[a], coords[b]
        avg = (vals["Client"] + vals["Counselor"]) / 2.0
        norm = (avg - vmin) / denom
        line_color = plt.cm.YlOrRd(0.25 + 0.7 * norm)
        line_width = 2.0 + 7.0 * norm
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=line_color, linewidth=line_width, alpha=0.85, zorder=1)

        mid = 0.5 * (p1 + p2) + edge_offsets[key]
        label = f"Client {vals['Client']:.3f}\nCounselor {vals['Counselor']:.3f}"
        ax.text(
            mid[0],
            mid[1],
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#777777", "alpha": 0.92},
            zorder=3,
        )

    for node, xy in coords.items():
        ax.scatter([xy[0]], [xy[1]], s=3200, c=colors[node], edgecolors="white", linewidths=2.0, zorder=2)
        ax.text(xy[0], xy[1], labels[node], color="white", fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)

    ax.set_title("Main Contribution: Triadic Chat Comparison (JSD)", fontsize=12, pad=12)
    ax.text(
        0.5,
        -0.06,
        "Edge labels report pairwise divergence by speaker role (lower = more similar).",
        fontsize=9,
        ha="center",
        va="top",
        transform=ax.transAxes,
    )
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.15, 1.2)
    ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    jsd_map = load_triad_jsd(Path(args.rq1_csv))
    plot_triad(jsd_map, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
