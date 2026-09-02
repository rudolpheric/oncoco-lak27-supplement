#!/usr/bin/env python3
"""Noise-band, size-matched null, Cramer's V and FDR-controlled transition tests.

Implements the GEMCo validation methodology (Steigerwald et al., 2026) for the
OnCoCo reanalysis paper:

1. Split-half noise band: the reference condition's conversations are split
   into two random halves B times; the JSD between halves forms the
   within-reference noise distribution (mean = noise floor, P95 = upper edge).
   Each between-condition JSD is placed on this band as an empirical
   percentile.
2. Size-matched null: both groups are resampled with replacement from the
   reference at the actual comparison sizes (n_comp, n_ref), removing the
   sample-size confound of the 50/50 split-half band.
3. Cramer's V: standardised effect size from the 2 x k contingency of pooled
   span counts (r=2, so V = sqrt(chi2/n)).
4. Transition-cell tests with FDR control: per off-diagonal transition cell,
   the observed row-normalised probability difference is compared against the
   reference's own split-half differences (two-sided empirical p), with
   Benjamini-Hochberg control at q=0.10 across all eligible cells.
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPEAKERS = ["Client", "Counsellor"]

# (comparison condition, reference condition, optional model filter)
COMPARISONS = [
    ("HH_roleplay_chat", "HH_real_chat", None),
    ("H_LLM_roleplay_chat", "HH_real_chat", None),
    ("H_LLM_roleplay_chat", "HH_real_chat", "GPT_OSS_120B"),
    ("H_LLM_roleplay_chat", "HH_real_chat", "LLama3_3_70B"),
    ("H_LLM_roleplay_chat", "HH_real_chat", "Mixtral"),
    ("HH_CAIA_mail", "HH_roleplay_mail", None),
]

# Transition tests: (comparison, reference) on interleaved role-aware sequences
TRANSITION_COMPARISONS = [
    ("H_LLM_roleplay_chat", "HH_real_chat"),
    ("HH_roleplay_chat", "HH_real_chat"),
    ("HH_CAIA_mail", "HH_roleplay_mail"),
]

ROLE_ORDER = [
    "CL:Opening", "CL:Empathy", "CL:Clarify", "CL:Objectives", "CL:Motivation",
    "CL:Resources", "CL:Help", "CL:Closing", "CL:Other",
    "CO:Opening", "CO:Moderation", "CO:Clarify", "CO:Objectives", "CO:Motivation",
    "CO:Resources", "CO:Help", "CO:Closing", "CO:Other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
    )
    parser.add_argument(
        "--out-band-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_noise_band.csv"),
    )
    parser.add_argument(
        "--out-band-tex",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_noise_band.tex"),
    )
    parser.add_argument(
        "--out-fdr-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_transition_fdr.csv"),
    )
    parser.add_argument(
        "--unit",
        choices=["span", "message"],
        default="span",
        help="Unit the label distributions are built from. 'span' reads sentence_classification "
             "(default, reproduces the published tables byte-for-byte); 'message' reads "
             "message_classification, i.e. no segmentation at all. The OnCoCo classifier was "
             "trained on utterances (one sentence or less), not on whole messages. Non-default units write suffixed outputs and skip "
             "the transition/FDR section, which is a span-sequence analysis.",
    )
    parser.add_argument(
        "--out-suffix",
        default="",
        help="Suffix for the output filenames. Required whenever --data-json is not the primary "
             "artifact, otherwise a variant run would overwrite the tables the paper reads. "
             "Defaults to the unit name for --unit message.",
    )
    parser.add_argument("--n-splits", type=int, default=1000, help="Split-half resamples for the noise band.")
    parser.add_argument("--n-null", type=int, default=1000, help="Resamples for the size-matched null.")
    parser.add_argument("--n-trans", type=int, default=2000, help="Split-half resamples for transition nulls.")
    parser.add_argument("--min-count", type=int, default=10, help="Minimum combined transitions per tested cell.")
    parser.add_argument("--fdr-q", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))

    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def cramers_v(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Cramer's V for the 2 x k contingency of two pooled count vectors."""
    obs = np.vstack([p_counts, q_counts])
    keep = obs.sum(axis=0) > 0
    obs = obs[:, keep]
    n = obs.sum()
    if n == 0 or obs.shape[1] < 2:
        return float("nan")
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    expected = row @ col / n
    chi2 = float(np.sum((obs - expected) ** 2 / expected))
    return sqrt(chi2 / n)


def role_category(label: str) -> str:
    if label.startswith("CO-"):
        if label.startswith("CO-FA"):
            return "CO:Opening"
        if label.startswith("CO-Mod"):
            return "CO:Moderation"
        if label.startswith("CO-FC"):
            return "CO:Closing"
        if label.startswith("CO-O"):
            return "CO:Other"
        if label.startswith("CO-IF-AC"):
            return "CO:Clarify"
        if label.startswith("CO-IF-AO"):
            return "CO:Objectives"
        if label.startswith("CO-IF-Mot"):
            return "CO:Motivation"
        if label.startswith("CO-IF-RA"):
            return "CO:Resources"
        if label.startswith("CO-IF-HP"):
            return "CO:Help"
        return "CO:Other"
    if label.startswith("CL-"):
        if label.startswith("CL-FB"):
            return "CL:Opening"
        if label.startswith("CL-E"):
            return "CL:Empathy"
        if label.startswith("CL-FC"):
            return "CL:Closing"
        if label.startswith("CL-O"):
            return "CL:Other"
        if label.startswith("CL-IF-ACP"):
            return "CL:Clarify"
        if label.startswith("CL-IF-AO"):
            return "CL:Objectives"
        if label.startswith("CL-IF-Mot"):
            return "CL:Motivation"
        if label.startswith("CL-IF-RA"):
            return "CL:Resources"
        if label.startswith("CL-IF-HP"):
            return "CL:Help"
        return "CL:Other"
    return "CL:Other"


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def span_labels_by_conversation(
    rows: List[Dict], condition: str, speaker: str, model: str | None = None,
    unit: str = "span",
) -> Dict[str, List[str]]:
    """conversation_key -> labels. `unit` selects spans or whole messages.

    The key is (source_file, id): ids repeat across raw exports, so keying on id alone
    would silently merge conversations from different files.
    """
    out: Dict[str, List[str]] = {}
    for r in rows:
        if r.get("source") != condition or r.get("speaker_type") != speaker:
            continue
        if model is not None and r.get("model") != model:
            continue
        conv = f"{r.get('source_file', '')}::{r.get('id')}"
        if unit == "message":
            lab = (r.get("message_classification") or {}).get("predicted_label")
            labels = [lab] if lab else []
        else:
            labels = [s.get("predicted_label") for s in r.get("sentence_classification", []) or []]
            labels = [l for l in labels if l]
        if labels:
            out.setdefault(conv, []).extend(labels)
    return out


def count_matrix(conv_labels: Dict[str, List[str]], label_index: Dict[str, int]) -> Tuple[List[str], np.ndarray]:
    """Per-conversation count vectors, so resampling is a matrix row-sum."""
    ids = sorted(conv_labels)
    mat = np.zeros((len(ids), len(label_index)), dtype=float)
    for i, cid in enumerate(ids):
        for lab in conv_labels[cid]:
            mat[i, label_index[lab]] += 1.0
    return ids, mat


def split_half_band(ref_mat: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    n = ref_mat.shape[0]
    half = n // 2
    vals = np.empty(n_splits)
    for b in range(n_splits):
        perm = rng.permutation(n)
        vals[b] = js_distance(ref_mat[perm[:half]].sum(axis=0), ref_mat[perm[half:]].sum(axis=0))
    return vals


def size_matched_null(
    ref_mat: np.ndarray, n_comp: int, n_ref: int, n_null: int, rng: np.random.Generator
) -> np.ndarray:
    n = ref_mat.shape[0]
    vals = np.empty(n_null)
    for b in range(n_null):
        a = ref_mat[rng.integers(0, n, size=n_comp)].sum(axis=0)
        c = ref_mat[rng.integers(0, n, size=n_ref)].sum(axis=0)
        vals[b] = js_distance(a, c)
    return vals


def noise_band_analysis(rows: List[Dict], args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    records = []
    for comp_cond, ref_cond, model in COMPARISONS:
        for speaker in SPEAKERS:
            unit = getattr(args, "unit", "span")
            comp = span_labels_by_conversation(rows, comp_cond, speaker, model, unit)
            ref = span_labels_by_conversation(rows, ref_cond, speaker, None, unit)
            if not comp or not ref:
                # e.g. a chat-only artifact carries no mail conditions
                print(f"skip {comp_cond} vs {ref_cond} ({speaker}): "
                      f"{len(comp)} comparison and {len(ref)} reference conversations")
                continue
            all_labels = sorted({l for labs in list(comp.values()) + list(ref.values()) for l in labs})
            label_index = {l: i for i, l in enumerate(all_labels)}
            _, comp_mat = count_matrix(comp, label_index)
            _, ref_mat = count_matrix(ref, label_index)

            comp_counts = comp_mat.sum(axis=0)
            ref_counts = ref_mat.sum(axis=0)
            jsd_rp = js_distance(comp_counts, ref_counts)
            v = cramers_v(comp_counts, ref_counts)

            band = split_half_band(ref_mat, args.n_splits, rng)
            pctl = float((band < jsd_rp).mean()) * 100.0

            nm = size_matched_null(ref_mat, comp_mat.shape[0], ref_mat.shape[0], args.n_null, rng)
            nm_p95 = float(np.percentile(nm, 95))

            records.append(
                {
                    "comparison": comp_cond + (f" [{model}]" if model else ""),
                    "reference": ref_cond,
                    "speaker": "Counselor" if speaker == "Counsellor" else speaker,
                    "n_comp_conv": comp_mat.shape[0],
                    "n_ref_conv": ref_mat.shape[0],
                    "jsd_rp": round(jsd_rp, 4),
                    "band_mean": round(float(band.mean()), 4),
                    "band_p95": round(float(np.percentile(band, 95)), 4),
                    "band_pctl_of_jsd": round(pctl, 1),
                    "ratio_to_floor": round(jsd_rp / float(band.mean()), 1),
                    "nm_p95": round(nm_p95, 4),
                    "exceeds_nm_p95": bool(jsd_rp > nm_p95),
                    "cramers_v": round(v, 3),
                }
            )
            print(records[-1])
    return pd.DataFrame(records)


def sequences_by_conversation(rows: List[Dict], condition: str) -> Dict[str, List[str]]:
    """Role-aware macro-category sequence per conversation, span order (as in oncoco_sequence_analysis).

    Keyed on (source_file, id) for the same reason as span_labels_by_conversation: ids repeat
    across raw exports, and keying on the id alone would merge two conversations into one
    sequence and invent the transition between them.
    """
    by_conv: Dict[str, List[Dict]] = {}
    for r in rows:
        if r.get("source") != condition:
            continue
        conv = f"{r.get('source_file', '')}::{r.get('msg_learn_counselling_id') or r.get('id')}"
        by_conv.setdefault(conv, []).append(r)

    out: Dict[str, List[str]] = {}
    for conv, messages in by_conv.items():
        def msg_key(m):
            num = m.get("msg_message_number")
            try:
                return (int(num), m.get("msg_created_at") or "")
            except Exception:
                return (10 ** 9, m.get("msg_created_at") or "")

        seq: List[str] = []
        for msg in sorted(messages, key=msg_key):
            sentences = msg.get("sentence_classification") or []

            def sent_key(s):
                if "sentence_index" in s:
                    return s.get("sentence_index")
                return s.get("start_offset", 0)

            for sent in sorted(sentences, key=sent_key):
                label = sent.get("predicted_label")
                if label:
                    seq.append(role_category(label))
        if len(seq) >= 2:
            out[conv] = seq
    return out


def transition_counts(seqs: List[List[str]], index: Dict[str, int]) -> np.ndarray:
    k = len(index)
    counts = np.zeros((k, k), dtype=float)
    for seq in seqs:
        idx = [index[s] for s in seq]
        for a, b in zip(idx, idx[1:]):
            counts[a, b] += 1.0
    return counts


def row_normalise(counts: np.ndarray) -> np.ndarray:
    sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(sums > 0, counts / sums, 0.0)
    return probs


def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    passed = np.zeros(m, dtype=bool)
    max_k = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank / m * q:
            max_k = rank
    passed[order[:max_k]] = True
    return passed


def transition_fdr_analysis(rows: List[Dict], args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    index = {c: i for i, c in enumerate(ROLE_ORDER)}
    records = []
    for comp_cond, ref_cond in TRANSITION_COMPARISONS:
        comp_seqs = list(sequences_by_conversation(rows, comp_cond).values())
        ref_seqs = list(sequences_by_conversation(rows, ref_cond).values())

        comp_counts = transition_counts(comp_seqs, index)
        ref_counts = transition_counts(ref_seqs, index)
        comp_probs = row_normalise(comp_counts)
        ref_probs = row_normalise(ref_counts)
        delta_obs = comp_probs - ref_probs

        combined = comp_counts + ref_counts
        eligible = (combined >= args.min_count) & ~np.eye(len(ROLE_ORDER), dtype=bool)

        # Null: split-half differences within the reference condition
        n_ref = len(ref_seqs)
        half = n_ref // 2
        null_deltas = np.empty((args.n_trans, len(ROLE_ORDER), len(ROLE_ORDER)))
        for b in range(args.n_trans):
            perm = rng.permutation(n_ref)
            h1 = transition_counts([ref_seqs[i] for i in perm[:half]], index)
            h2 = transition_counts([ref_seqs[i] for i in perm[half:]], index)
            null_deltas[b] = row_normalise(h1) - row_normalise(h2)

        cells = np.argwhere(eligible)
        pvals = np.empty(len(cells))
        for j, (a, b_) in enumerate(cells):
            pvals[j] = (1.0 + float((np.abs(null_deltas[:, a, b_]) >= abs(delta_obs[a, b_])).sum())) / (
                args.n_trans + 1.0
            )
        passed = benjamini_hochberg(pvals, args.fdr_q)

        for j, (a, b_) in enumerate(cells):
            records.append(
                {
                    "comparison": comp_cond,
                    "reference": ref_cond,
                    "from": ROLE_ORDER[a],
                    "to": ROLE_ORDER[b_],
                    "p_comp": round(float(comp_probs[a, b_]), 4),
                    "p_ref": round(float(ref_probs[a, b_]), 4),
                    "delta": round(float(delta_obs[a, b_]), 4),
                    "combined_count": int(combined[a, b_]),
                    "p_value": round(float(pvals[j]), 4),
                    "fdr_significant": bool(passed[j]),
                }
            )
        n_sig = int(passed.sum())
        print(f"{comp_cond} vs {ref_cond}: {len(cells)} cells tested, {n_sig} significant at q={args.fdr_q}")
    return pd.DataFrame(records)


def write_band_tex(df: pd.DataFrame, path: Path) -> None:
    pretty = {
        "HH_roleplay_chat": "HH roleplay chat",
        "H_LLM_roleplay_chat": "H--LLM chat",
        "H_LLM_roleplay_chat [GPT_OSS_120B]": "H--LLM [GPT-OSS-120B]",
        "H_LLM_roleplay_chat [LLama3_3_70B]": "H--LLM [Llama 3.3 70B]",
        "H_LLM_roleplay_chat [Mixtral]": "H--LLM [Mixtral 8x7B]",
        "HH_CAIA_mail": "HH+assistant mail",
    }
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Comparison & Role & JSD & Band mean & Band P95 & nm-P95 & $V$ \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        comp = pretty.get(r["comparison"], r["comparison"]).replace("_", " ")
        lines.append(
            f"{comp} & {r['speaker']} & {r['jsd_rp']:.3f} & {r['band_mean']:.3f} & "
            f"{r['band_p95']:.3f} & {r['nm_p95']:.3f} & {r['cramers_v']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _suffixed(path: Path, suffix: str) -> Path:
    return path if not suffix else path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.data_json))

    # A variant run must never overwrite the primary tables the paper reads.
    suffix = args.out_suffix or ("" if args.unit == "span" else args.unit)
    default_json = str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized"
                       / "oncoco_classification_all.json")
    if not suffix and str(Path(args.data_json)) != default_json:
        raise SystemExit(
            f"--data-json is {Path(args.data_json).name}, not the primary artifact, but no "
            "--out-suffix was given; refusing to overwrite the primary noise-band tables.")
    band_df = noise_band_analysis(rows, args)
    out_csv = _suffixed(Path(args.out_band_csv), suffix)
    out_tex = _suffixed(Path(args.out_band_tex), suffix)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    band_df.to_csv(out_csv, index=False)
    write_band_tex(band_df, out_tex)
    print(f"Wrote {out_csv} and {out_tex}")

    if args.unit != "span":
        print("Skipping the transition/FDR section: it is a span-sequence analysis and is "
              "unaffected by the label unit.")
        return

    fdr_df = transition_fdr_analysis(rows, args)
    # Suffix this output like the band tables: a non-primary run (e.g. the regex corpus at
    # --unit span) otherwise silently overwrites the primary FDR table that the paper cites.
    out_fdr = _suffixed(Path(args.out_fdr_csv), suffix)
    fdr_df.to_csv(out_fdr, index=False)
    print(f"Wrote {out_fdr}")


if __name__ == "__main__":
    main()
