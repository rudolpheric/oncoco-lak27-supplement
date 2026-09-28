#!/usr/bin/env python3
"""Bootstrap the RQ1 proximity gap across units of analysis.

For each bootstrap sample, conversations are resampled with replacement
within each chat condition; the statistic is the proximity gap
    delta' = JSD(HH_real, H_LLM) - JSD(HH_real, HH_roleplay)
computed on the pooled label distributions. Both terms are distances to real
counseling, so delta' compares the simulated client with a human roleplay
partner on the same yardstick. delta' > 0 means the simulated client is
FARTHER from real counseling than human roleplay is (the bad direction);
delta' <= 0 is what a simulated client should reach to be at least as
realistic as roleplay.

(Until 2026-09 the statistic was JSD(HH_real, H_LLM) - JSD(HH_roleplay, H_LLM),
i.e. "is H-LLM closer to roleplay than to real". That reading is kept in the
text as a description of the distance triangle but is no longer the metric.)

Four units are available, and the ordering should be read across all of them:

  SaT              spans from SaT-6l + a LoRA adapter (the paper's primary unit)
  regex            spans from the corrected sentence regex
  regex-published  spans from the regex as originally shipped -- KEPT ONLY AS A RECORD.
                   Its character class was r"[^.!?\\n]+", where the raw-string "\\n" is
                   backslash + the letter "n", so it split German text on every "n"
                   (~16 spans/message, mean 10 characters). Numbers from this unit are
                   artifacts and must never be reported as a segmentation comparison.
  message          whole messages, i.e. no segmentation at all. NOTE: despite the name
                   "message classification", OnCoCo 1.0 is annotated and trained on
                   utterances (typically one sentence or less), so this unit is the most
                   out-of-distribution one for the classifier, not the least.

The resampling unit is the conversation under every unit of analysis, which is what
keeps the four columns comparable.
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONDITIONS = ["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat"]
SPEAKERS = ["Client", "Counsellor"]


NORM = PROJECT_ROOT / "data" / "processed" / "combined" / "normalized"

# name -> (artifact, unit). "span" reads sentence_classification, "message" reads
# message_classification, which every artifact already carries.
UNITS = {
    "SaT": (NORM / "oncoco_classification_all.json", "span"),
    "regex": (NORM / "oncoco_classification_all_regex_v2.json", "span"),
    "regex-published": (NORM / "oncoco_classification_all_regex.json", "span"),
    "message": (NORM / "oncoco_classification_all.json", "message"),
}
DEFAULT_UNITS = "SaT,regex,message"

# Expected conversation counts; a silent key collision or a data drop must fail loudly.
EXPECTED_CONVERSATIONS = {"HH_roleplay_chat": 68, "HH_real_chat": 53, "H_LLM_roleplay_chat": 414}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the RQ1 proximity gap across units.")
    parser.add_argument(
        "--units",
        default=DEFAULT_UNITS,
        help=f"comma-separated subset of {sorted(UNITS)} (default: {DEFAULT_UNITS})",
    )
    parser.add_argument(
        "--out-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_seg_ordering_bootstrap.csv"),
    )
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_label_lists(path: Path, unit: str = "span") -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """condition -> speaker -> conversation_key -> list of labels.

    The conversation key is (source_file, id): ids repeat across raw exports, so keying on
    id alone would silently merge conversations from different files.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for r in rows:
        condition = r.get("source")
        speaker = r.get("speaker_type")
        conv = r.get("id")
        if condition not in CONDITIONS or speaker not in SPEAKERS or not conv:
            continue
        if unit == "message":
            lab = (r.get("message_classification") or {}).get("predicted_label")
            labels = [lab] if lab else []
        else:
            labels = [s.get("predicted_label") for s in r.get("sentence_classification", []) or []]
            labels = [l for l in labels if l]
        if not labels:
            continue
        key = f"{r.get('source_file', '')}::{conv}"
        out.setdefault(condition, {}).setdefault(speaker, {}).setdefault(key, []).extend(labels)
    return out


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))

    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def pooled_counts(conv_labels: Dict[str, List[str]], conv_ids: List[str], label_index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(label_index), dtype=float)
    for cid in conv_ids:
        for lab in conv_labels[cid]:
            vec[label_index[lab]] += 1.0
    return vec


def conversation_matrix(conv_labels: Dict[str, List[str]], conv_ids: List[str], label_index: Dict[str, int]) -> np.ndarray:
    """One row of label counts per conversation, so a resample is a row draw plus a column sum."""
    mat = np.zeros((len(conv_ids), len(label_index)), dtype=float)
    for i, cid in enumerate(conv_ids):
        for lab in conv_labels[cid]:
            mat[i, label_index[lab]] += 1.0
    return mat


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    results = []

    requested = [u.strip() for u in args.units.split(",") if u.strip()]
    unknown = [u for u in requested if u not in UNITS]
    if unknown:
        raise SystemExit(f"unknown unit(s) {unknown}; available: {sorted(UNITS)}")

    for seg_name in requested:
        path, unit = UNITS[seg_name]
        if not path.exists():
            raise SystemExit(f"unit {seg_name!r}: missing artifact {path}")
        data = load_label_lists(path, unit)
        for cond, expected in EXPECTED_CONVERSATIONS.items():
            seen = len({k for sp in SPEAKERS for k in data.get(cond, {}).get(sp, {})})
            if seen != expected:
                raise SystemExit(
                    f"unit {seg_name!r}: {cond} has {seen} conversations, expected {expected}")
        for speaker in SPEAKERS:
            conv_labels = {c: data[c][speaker] for c in CONDITIONS}
            all_labels = sorted({l for c in CONDITIONS for labs in conv_labels[c].values() for l in labs})
            label_index = {l: i for i, l in enumerate(all_labels)}
            conv_ids = {c: sorted(conv_labels[c]) for c in CONDITIONS}

            # Point estimate on the full data
            mats = {c: conversation_matrix(conv_labels[c], conv_ids[c], label_index) for c in CONDITIONS}
            full = {c: mats[c].sum(axis=0) for c in CONDITIONS}
            jsd_llm = js_distance(full["HH_real_chat"], full["H_LLM_roleplay_chat"])
            jsd_rp = js_distance(full["HH_real_chat"], full["HH_roleplay_chat"])
            point = jsd_llm - jsd_rp

            deltas = np.empty(args.n_boot)
            for b in range(args.n_boot):
                sample = {}
                for c in CONDITIONS:
                    n = mats[c].shape[0]
                    sample[c] = mats[c][rng.integers(0, n, n)].sum(axis=0)
                deltas[b] = js_distance(sample["HH_real_chat"], sample["H_LLM_roleplay_chat"]) - js_distance(
                    sample["HH_real_chat"], sample["HH_roleplay_chat"]
                )

            lo, hi = np.percentile(deltas, [2.5, 97.5])
            results.append(
                {
                    "unit": seg_name,
                    "analysis_unit": UNITS[seg_name][1],
                    "speaker": "Counselor" if speaker == "Counsellor" else speaker,
                    # 6 decimals, not 4: the tables format these to 3 places, and rounding twice
                    # turned 0.374541 into 0.374 while the text said 0.375.
                    "jsd_real_vs_hllm": round(jsd_llm, 6),
                    "jsd_real_vs_roleplay": round(jsd_rp, 6),
                    "delta_point": round(point, 6),
                    "ci_lo": round(lo, 6),
                    "ci_hi": round(hi, 6),
                    "frac_delta_positive": round(float((deltas > 0).mean()), 4),
                    "n_boot": args.n_boot,
                }
            )
            print(results[-1])

    out = pd.DataFrame(results)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
