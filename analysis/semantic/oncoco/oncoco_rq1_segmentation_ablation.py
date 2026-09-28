#!/usr/bin/env python3
"""Recompute key RQ1 distances under SAT vs regex segmentation."""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ1 segmentation ablation (SAT vs regex).")
    parser.add_argument(
        "--sat-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="SAT-based sentence classification JSON.",
    )
    parser.add_argument(
        "--regex-json",
        # CORRECTED artifact; the original split German text on the letter "n" (see
        # analysis/semantic/classification/classification_all.py).
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all_regex_v2.json"),
        help="Regex-based sentence classification JSON.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_rq1_segmentation_ablation.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-tex",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_rq1_segmentation_ablation.tex"),
        help="Output compact TeX table path.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def explode_sentences(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        condition = r.get("source")
        speaker = r.get("speaker_type")
        for sent in r.get("sentence_classification", []) or []:
            label = sent.get("predicted_label")
            if condition and speaker and label:
                records.append(
                    {
                        "condition": condition,
                        "speaker_type": speaker,
                        "label": label,
                    }
                )
    return pd.DataFrame.from_records(records)


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["condition", "speaker_type", "label"]).size().reset_index(name="count")
    totals = counts.groupby(["condition", "speaker_type"])["count"].sum().reset_index(name="total")
    merged = counts.merge(totals, on=["condition", "speaker_type"], how="left")
    merged["proportion"] = merged["count"] / merged["total"]
    return merged


def align_distributions(dist_a: pd.DataFrame, dist_b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    labels = sorted(set(dist_a["label"]).union(set(dist_b["label"])))
    a_map = dict(zip(dist_a["label"], dist_a["proportion"]))
    b_map = dict(zip(dist_b["label"], dist_b["proportion"]))
    p = np.array([a_map.get(label, 0.0) for label in labels], dtype=float)
    q = np.array([b_map.get(label, 0.0) for label in labels], dtype=float)
    return p, q


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))

    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def compute_pairwise(dist: pd.DataFrame, cond_a: str, cond_b: str, speaker: str) -> float:
    a = dist[(dist["condition"] == cond_a) & (dist["speaker_type"] == speaker)]
    b = dist[(dist["condition"] == cond_b) & (dist["speaker_type"] == speaker)]
    if a.empty or b.empty:
        return float("nan")
    p, q = align_distributions(a, b)
    return js_distance(p, q)


def main() -> None:
    args = parse_args()
    sat = label_distribution(explode_sentences(load_rows(Path(args.sat_json))))
    regex = label_distribution(explode_sentences(load_rows(Path(args.regex_json))))

    comparisons = [
        ("HH_roleplay_chat", "H_LLM_roleplay_chat"),
        ("HH_real_chat", "HH_roleplay_chat"),
        ("HH_real_chat", "H_LLM_roleplay_chat"),
        ("HH_real_mail", "HH_roleplay_mail"),
        ("HH_roleplay_chat", "HH_roleplay_mail"),
        ("HH_real_chat", "HH_real_mail"),
    ]
    speakers = ["Client", "Counsellor"]
    rows = []

    for cond_a, cond_b in comparisons:
        for speaker in speakers:
            sat_jsd = compute_pairwise(sat, cond_a, cond_b, speaker)
            regex_jsd = compute_pairwise(regex, cond_a, cond_b, speaker)
            rows.append(
                {
                    "comparison": f"{cond_a} vs {cond_b}",
                    "speaker_type": "Counselor" if speaker == "Counsellor" else speaker,
                    "sat_jsd": sat_jsd,
                    "regex_jsd": regex_jsd,
                    "abs_delta": abs(sat_jsd - regex_jsd),
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # Compact TeX table for paper
    keep = out_df.copy()
    keep["comparison"] = keep["comparison"].replace(
        {
            "HH_roleplay_chat vs H_LLM_roleplay_chat": "HH roleplay chat vs H--LLM chat",
            "HH_real_chat vs HH_roleplay_chat": "HH real chat vs HH roleplay chat",
            "HH_real_chat vs H_LLM_roleplay_chat": "HH real chat vs H--LLM chat",
            "HH_real_mail vs HH_roleplay_mail": "HH real mail vs HH roleplay mail",
            "HH_roleplay_chat vs HH_roleplay_mail": "HH roleplay chat vs HH roleplay mail",
            "HH_real_chat vs HH_real_mail": "HH real chat vs HH real mail",
        }
    )
    wide = keep.pivot(index="comparison", columns="speaker_type", values=["sat_jsd", "regex_jsd"]).reset_index()
    wide.columns = [
        "comparison",
        "sat_client",
        "sat_counselor",
        "regex_client",
        "regex_counselor",
    ]
    wide = wide[
        [
            "comparison",
            "sat_client",
            "regex_client",
            "sat_counselor",
            "regex_counselor",
        ]
    ]

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Comparison & C-SAT & C-regex & CO-SAT & CO-regex \\\\",
        "\\midrule",
    ]
    for _, r in wide.iterrows():
        lines.append(
            f"{r['comparison']} & {r['sat_client']:.3f} & {r['regex_client']:.3f} & "
            f"{r['sat_counselor']:.3f} & {r['regex_counselor']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])

    out_tex = Path(args.out_tex)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    main()
