#!/usr/bin/env python3
"""Compute segmentation-sensitive label shifts per condition (SAT vs regex)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def explode_sentences(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        condition = r.get("source")
        speaker_type = r.get("speaker_type")
        for sent in r.get("sentence_classification", []) or []:
            label = sent.get("predicted_label")
            if not (condition and speaker_type and label):
                continue
            records.append(
                {
                    "condition": condition,
                    "speaker_type": speaker_type,
                    "label": label,
                }
            )
    df = pd.DataFrame.from_records(records)
    return df.dropna(subset=["condition", "speaker_type", "label"])


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["condition", "speaker_type", "label"]).size().reset_index(name="count")
    totals = df.groupby(["condition", "speaker_type"]).size().reset_index(name="total")
    merged = counts.merge(totals, on=["condition", "speaker_type"], how="left")
    merged["proportion"] = merged["count"] / merged["total"]
    return merged


def load_label_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def label_to_text(code: str, label_map: Dict[str, str]) -> str:
    name = label_map.get(code)
    speaker = "Counselor" if code.startswith("CO-") else "Client" if code.startswith("CL-") else ""
    if name:
        return f"{speaker}: {name}" if speaker else name
    # fallback: avoid raw code, but keep a readable hint
    if "-O-" in code:
        base = "Other statements"
    elif "-Mod-" in code:
        base = "Moderation"
    elif "-FA-" in code:
        base = "Formalities at the beginning"
    else:
        base = "Unmapped category"
    suffix = ""
    if code.endswith("PI-*"):
        suffix = " (PI)"
    elif code.endswith("ST-*"):
        suffix = " (ST)"
    return f"{speaker}: {base}{suffix}" if speaker else f"{base}{suffix}"


def compute_label_shifts(dist_sat: pd.DataFrame, dist_regex: pd.DataFrame, label_map: Dict[str, str], top_k: int) -> pd.DataFrame:
    rows = []
    conditions = sorted(set(dist_sat["condition"]).union(dist_regex["condition"]))
    speakers = sorted(set(dist_sat["speaker_type"]).union(dist_regex["speaker_type"]))
    for condition in conditions:
        for speaker in speakers:
            sat = dist_sat[(dist_sat["condition"] == condition) & (dist_sat["speaker_type"] == speaker)]
            reg = dist_regex[(dist_regex["condition"] == condition) & (dist_regex["speaker_type"] == speaker)]
            if sat.empty or reg.empty:
                continue
            labels = sorted(set(sat["label"]).union(set(reg["label"])))
            sat_map = dict(zip(sat["label"], sat["proportion"]))
            reg_map = dict(zip(reg["label"], reg["proportion"]))
            data = []
            for label in labels:
                p_sat = float(sat_map.get(label, 0.0))
                p_reg = float(reg_map.get(label, 0.0))
                diff = p_sat - p_reg
                data.append(
                    {
                        "condition": condition,
                        "speaker_type": speaker,
                        "label": label,
                        "label_text": label_to_text(label, label_map),
                        "p_sat": p_sat,
                        "p_regex": p_reg,
                        "diff": diff,
                        "abs_diff": abs(diff),
                    }
                )
            df = pd.DataFrame(data).sort_values("abs_diff", ascending=False).head(top_k)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def write_summary(md_path: Path, shifts: pd.DataFrame) -> None:
    lines = ["# RQ2 Segmentation-Sensitive Labels (SAT vs regex)", ""]
    if shifts.empty:
        lines.append("No shifts computed (missing data).")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return
    for condition in sorted(shifts["condition"].unique()):
        lines.append(f"## {condition}")
        subset = shifts[shifts["condition"] == condition]
        for speaker in sorted(subset["speaker_type"].unique()):
            lines.append(f"- {speaker}:")
            top = subset[subset["speaker_type"] == speaker].sort_values("abs_diff", ascending=False)
            for _, row in top.iterrows():
                lines.append(
                    f"  - {row['label_text']} (SAT–regex: {row['diff']:+.3f})"
                )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute top segmentation-sensitive labels per condition.")
    parser.add_argument(
        "--sat-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="Path to SAT segmentation classification JSON.",
    )
    parser.add_argument(
        "--regex-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all_regex_v2.json"),
        help="Path to regex segmentation classification JSON. Defaults to the CORRECTED "
             "artifact (_v2); the original oncoco_classification_all_regex.json was produced "
             "with a character class that split German text on the letter 'n' and must not be "
             "used as a segmentation baseline.",
    )
    parser.add_argument(
        "--label-map",
        default=str(PROJECT_ROOT / "analysis" / "semantic" / "oncoco" / "label_text_map.json"),
        help="JSON mapping of OnCoCo label codes to names.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k labels by absolute shift.")
    parser.add_argument(
        "--out-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_rq2_segmentation_label_shifts.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        default=str(PROJECT_ROOT / "results" / "summaries" / "oncoco_rq2_label_shifts.md"),
        help="Output markdown summary path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sat_rows = load_rows(Path(args.sat_json))
    regex_rows = load_rows(Path(args.regex_json))
    label_map = load_label_map(Path(args.label_map))

    sat_df = explode_sentences(sat_rows)
    regex_df = explode_sentences(regex_rows)
    sat_dist = label_distribution(sat_df)
    regex_dist = label_distribution(regex_df)

    shifts = compute_label_shifts(sat_dist, regex_dist, label_map, args.top_k)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    shifts.to_csv(out_csv, index=False)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_summary(out_md, shifts)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
