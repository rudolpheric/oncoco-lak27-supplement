#!/usr/bin/env python3
"""Transition matrices and HMM state analysis for OnCoCo sequences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ROLE_ORDER = [
    "CL:Opening",
    "CL:Empathy",
    "CL:Clarify",
    "CL:Objectives",
    "CL:Motivation",
    "CL:Resources",
    "CL:Help",
    "CL:Closing",
    "CL:Other",
    "CO:Opening",
    "CO:Moderation",
    "CO:Clarify",
    "CO:Objectives",
    "CO:Motivation",
    "CO:Resources",
    "CO:Help",
    "CO:Closing",
    "CO:Other",
]

COARSE_ORDER = [
    "Opening",
    "Empathy",
    "Clarify",
    "Objectives",
    "Motivation",
    "Resources",
    "Help",
    "Closing",
    "Moderation",
    "Other",
]

CONDITION_NAME_MAP = {
    "HH_roleplay_chat": "HH roleplay chat",
    "HH_real_chat": "HH real chat",
    "H_LLM_roleplay_chat": "H--LLM roleplay chat",
    "HH_roleplay_mail": "HH roleplay mail",
    "HH_real_mail": "HH real mail",
    "HH_CAIA_mail": "HH+assistant (CAIA) mail",
    "LLM_LLM_mail": "LLM--LLM mail",
}

# The paper reports five conditions (three chat, two mail for RQ3); the HMM
# summary table is restricted to these, while the CSV outputs keep everything
# present in the data.
PAPER_CONDITIONS = [
    "HH_roleplay_chat",
    "HH_real_chat",
    "H_LLM_roleplay_chat",
    "HH_roleplay_mail",
    "HH_CAIA_mail",
]


def pretty_condition(condition: str) -> str:
    return CONDITION_NAME_MAP.get(condition, condition.replace("_", " "))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transition and HMM analysis for OnCoCo sequences.")
    parser.add_argument(
        "--data-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="Path to OnCoCo sentence classification JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "figures" / "oncoco"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--out-transitions",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_transition_matrix.csv"),
        help="Output CSV with transition probabilities.",
    )
    parser.add_argument(
        "--out-top",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_transition_top.csv"),
        help="Output CSV with top transitions per condition.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum transition count to include in top transition list.",
    )
    parser.add_argument(
        "--out-hmm",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_hmm_emissions.csv"),
        help="Output CSV with HMM emission probabilities.",
    )
    parser.add_argument(
        "--out-hmm-tex",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_hmm_state_summary.tex"),
        help="Output LaTeX table summarizing HMM states.",
    )
    parser.add_argument(
        "--out-hmm-transitions",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_hmm_transitions.csv"),
        help="Output CSV with HMM state transition probabilities.",
    )
    parser.add_argument("--n-states", type=int, default=3, help="Number of HMM states.")
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=20,
        help="Random initialisations per HMM; the best-likelihood fit is kept.",
    )
    parser.add_argument("--hmm-n-iter", type=int, default=500, help="Max EM iterations per HMM fit.")
    parser.add_argument("--hmm-tol", type=float, default=1e-4, help="EM convergence tolerance.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; restart r uses seed+r.")
    parser.add_argument("--top-k", type=int, default=5, help="Top transitions to report per condition.")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Do not write transition heatmaps. The paper's heatmaps are owned by "
        "pub_figures.py (Okabe--Ito palette, vector PDF); use this when refreshing "
        "tables only, so those figures are not overwritten with default styling.",
    )
    return parser.parse_args()


def role_category(label: str) -> Tuple[str, str]:
    if label.startswith("CO-"):
        if label.startswith("CO-FA"):
            return "CO:Opening", "Opening"
        if label.startswith("CO-Mod"):
            return "CO:Moderation", "Moderation"
        if label.startswith("CO-FC"):
            return "CO:Closing", "Closing"
        if label.startswith("CO-O"):
            return "CO:Other", "Other"
        if label.startswith("CO-IF-AC"):
            return "CO:Clarify", "Clarify"
        if label.startswith("CO-IF-AO"):
            return "CO:Objectives", "Objectives"
        if label.startswith("CO-IF-Mot"):
            return "CO:Motivation", "Motivation"
        if label.startswith("CO-IF-RA"):
            return "CO:Resources", "Resources"
        if label.startswith("CO-IF-HP"):
            return "CO:Help", "Help"
        return "CO:Other", "Other"
    if label.startswith("CL-"):
        if label.startswith("CL-FB"):
            return "CL:Opening", "Opening"
        if label.startswith("CL-E"):
            return "CL:Empathy", "Empathy"
        if label.startswith("CL-FC"):
            return "CL:Closing", "Closing"
        if label.startswith("CL-O"):
            return "CL:Other", "Other"
        if label.startswith("CL-IF-ACP"):
            return "CL:Clarify", "Clarify"
        if label.startswith("CL-IF-AO"):
            return "CL:Objectives", "Objectives"
        if label.startswith("CL-IF-Mot"):
            return "CL:Motivation", "Motivation"
        if label.startswith("CL-IF-RA"):
            return "CL:Resources", "Resources"
        if label.startswith("CL-IF-HP"):
            return "CL:Help", "Help"
        return "CL:Other", "Other"
    return "UNK", "Other"


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_sequences(rows: List[Dict]) -> Dict[str, List[List[Dict[str, str]]]]:
    # condition -> list of sequences (each sequence is list of dicts with role_cat and coarse)
    sequences: Dict[str, List[List[Dict[str, str]]]] = {}
    by_conv: Dict[Tuple[str, str], List[Dict]] = {}

    for r in rows:
        condition = r.get("source")
        conv_id = r.get("msg_learn_counselling_id") or r.get("id")
        if condition is None or conv_id is None:
            continue
        # (source_file, id): ids repeat across raw exports, so keying on the id alone would
        # concatenate two conversations and invent the transition at the seam.
        key = (condition, f"{r.get('source_file', '')}::{conv_id}")
        by_conv.setdefault(key, []).append(r)

    for (condition, conv_id), messages in by_conv.items():
        # sort messages by message number, fallback to created_at
        def msg_key(m):
            num = m.get("msg_message_number")
            try:
                return (int(num), m.get("msg_created_at") or "")
            except Exception:
                return (10**9, m.get("msg_created_at") or "")

        messages_sorted = sorted(messages, key=msg_key)
        seq: List[Dict[str, str]] = []
        for msg in messages_sorted:
            sentences = msg.get("sentence_classification") or []
            def sent_key(s):
                if "sentence_index" in s:
                    return s.get("sentence_index")
                return s.get("start_offset", 0)
            sentences_sorted = sorted(sentences, key=sent_key)
            for sent in sentences_sorted:
                label = sent.get("predicted_label")
                if not label:
                    continue
                role_cat, coarse = role_category(label)
                seq.append({"role_cat": role_cat, "coarse": coarse})
        if len(seq) >= 2:
            sequences.setdefault(condition, []).append(seq)

    return sequences


def transition_matrix(seqs: List[List[Dict[str, str]]], key: str, order: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # key: role_cat or coarse
    counts = {a: {b: 0 for b in order} for a in order}
    for seq in seqs:
        labels = [s[key] for s in seq if s[key] in order]
        for a, b in zip(labels, labels[1:]):
            counts[a][b] += 1
    count_df = pd.DataFrame(counts).T
    prob_df = count_df.div(count_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return count_df, prob_df


def plot_heatmap(prob_df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(prob_df, cmap="Blues", linewidths=0.2)
    plt.title(title)
    plt.xlabel("Next label")
    plt.ylabel("Previous label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def fit_hmm(
    seqs: List[List[Dict[str, str]]],
    order: List[str],
    n_states: int,
    n_restarts: int = 20,
    n_iter: int = 500,
    tol: float = 1e-4,
    seed: int = 42,
) -> Tuple[CategoricalHMM, np.ndarray, np.ndarray]:
    """Fit a categorical HMM from several random initialisations, keep the best.

    The EM likelihood surface for these sequences is strongly multimodal: 20
    restarts land on 7--12 distinct optima, and a single fixed seed lands in a
    markedly inferior one for all three chat conditions, producing states whose
    emission rows are near-duplicates. We therefore keep the best-likelihood
    solution and order states by Viterbi occupancy, so state 0 is the most
    frequently occupied mode rather than an arbitrary EM index.
    Returns (model, emissions, occupancy) with rows in occupancy order.
    """
    label_to_idx = {label: i for i, label in enumerate(order)}
    sequences_int = []
    lengths = []
    for seq in seqs:
        labels = [label_to_idx[s["coarse"]] for s in seq if s["coarse"] in label_to_idx]
        if len(labels) < 2:
            continue
        sequences_int.append(labels)
        lengths.append(len(labels))
    if not sequences_int:
        return None, None, None  # type: ignore
    X = np.concatenate([np.array(s) for s in sequences_int]).reshape(-1, 1)

    best_model, best_ll = None, -np.inf
    for r in range(n_restarts):
        model = CategoricalHMM(
            n_components=n_states,
            n_features=len(order),
            n_iter=n_iter,
            tol=tol,
            random_state=seed + r,
        )
        try:
            model.fit(X, lengths)
            ll = float(model.score(X, lengths))
        except Exception:
            continue
        if np.isfinite(ll) and ll > best_ll:
            best_model, best_ll = model, ll
    if best_model is None:
        return None, None, None  # type: ignore

    counts = np.bincount(best_model.predict(X, lengths), minlength=n_states).astype(float)
    occupancy = counts / counts.sum()
    order_idx = np.argsort(-occupancy)
    best_model.startprob_ = best_model.startprob_[order_idx]
    best_model.transmat_ = best_model.transmat_[np.ix_(order_idx, order_idx)]
    best_model.emissionprob_ = best_model.emissionprob_[order_idx]
    return best_model, best_model.emissionprob_, occupancy[order_idx]


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.data_json))
    sequences = build_sequences(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transition_rows = []
    top_rows = []
    hmm_rows = []
    hmm_transition_rows = []

    for condition, seqs in sequences.items():
        count_df, prob_df = transition_matrix(seqs, key="role_cat", order=ROLE_ORDER)
        # store transition probabilities
        for a in ROLE_ORDER:
            for b in ROLE_ORDER:
                transition_rows.append(
                    {
                        "condition": condition,
                        "from": a,
                        "to": b,
                        "count": int(count_df.loc[a, b]),
                        "prob": float(prob_df.loc[a, b]),
                    }
                )

        # top transitions by count with minimum count threshold
        flat_counts = count_df.stack().reset_index()
        flat_counts.columns = ["from", "to", "count"]
        flat_counts["prob"] = flat_counts.apply(lambda r: float(prob_df.loc[r["from"], r["to"]]), axis=1)
        filtered = flat_counts[flat_counts["count"] >= args.min_count].copy()
        if filtered.empty:
            filtered = flat_counts.copy()
        filtered = filtered.sort_values(["count", "prob"], ascending=False).head(args.top_k)
        for _, row in filtered.iterrows():
            top_rows.append(
                {
                    "condition": condition,
                    "from": row["from"],
                    "to": row["to"],
                    "count": int(row["count"]),
                    "prob": float(row["prob"]),
                }
            )

        # plot heatmap
        if not args.skip_figures:
            plot_heatmap(prob_df, f"Transition probabilities: {pretty_condition(condition)}", out_dir / f"transition_{condition}.png")

        # HMM on coarse labels
        model, emissions, occupancy = fit_hmm(
            seqs,
            COARSE_ORDER,
            args.n_states,
            n_restarts=args.n_restarts,
            n_iter=args.hmm_n_iter,
            tol=args.hmm_tol,
            seed=args.seed,
        )
        if emissions is not None:
            for state_idx in range(emissions.shape[0]):
                for label_idx, label in enumerate(COARSE_ORDER):
                    hmm_rows.append(
                        {
                            "condition": condition,
                            "state": state_idx,
                            "occupancy": float(occupancy[state_idx]),
                            "label": label,
                            "prob": float(emissions[state_idx, label_idx]),
                        }
                    )
                for to_idx in range(emissions.shape[0]):
                    hmm_transition_rows.append(
                        {
                            "condition": condition,
                            "from_state": state_idx,
                            "to_state": to_idx,
                            "prob": float(model.transmat_[state_idx, to_idx]),
                        }
                    )

    # save transition matrices
    trans_df = pd.DataFrame(transition_rows)
    Path(args.out_transitions).parent.mkdir(parents=True, exist_ok=True)
    trans_df.to_csv(args.out_transitions, index=False)

    top_df = pd.DataFrame(top_rows)
    Path(args.out_top).parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(args.out_top, index=False)

    hmm_df = pd.DataFrame(hmm_rows)
    Path(args.out_hmm).parent.mkdir(parents=True, exist_ok=True)
    hmm_df.to_csv(args.out_hmm, index=False)

    hmm_trans_df = pd.DataFrame(hmm_transition_rows)
    Path(args.out_hmm_transitions).parent.mkdir(parents=True, exist_ok=True)
    hmm_trans_df.to_csv(args.out_hmm_transitions, index=False)

    # build LaTeX summary table for HMM states (top 3 labels per state),
    # restricted to the five conditions the paper reports
    if not hmm_df.empty:
        lines = []
        lines.append("\\begin{tabular}{l r r p{0.56\\linewidth}}")
        lines.append("\\toprule")
        lines.append("Condition & State & Occ. & Top labels (prob.) \\\\")
        lines.append("\\midrule")
        reported = [c for c in PAPER_CONDITIONS if c in set(hmm_df["condition"])]
        for condition in reported:
            sub = hmm_df[hmm_df["condition"] == condition]
            for state in sorted(sub["state"].unique()):
                sub_s = sub[sub["state"] == state].sort_values("prob", ascending=False)
                # top 3, but drop emissions that round to zero (noise, not structure)
                top = sub_s.head(3)
                top = top[top["prob"] >= 0.005]
                if top.empty:
                    top = sub_s.head(1)
                labels = ", ".join([f"{row['label']} ({row['prob']:.2f})" for _, row in top.iterrows()])
                name = pretty_condition(condition) if state == sub["state"].min() else ""
                lines.append(f"{name} & {state} & {sub_s['occupancy'].iloc[0]:.2f} & {labels} \\\\")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        Path(args.out_hmm_tex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_hmm_tex).write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {args.out_transitions}")
    print(f"Wrote {args.out_top}")
    print(f"Wrote {args.out_hmm}")
    print(f"Wrote {args.out_hmm_transitions}")
    print(f"Wrote {args.out_hmm_tex}")


if __name__ == "__main__":
    main()
