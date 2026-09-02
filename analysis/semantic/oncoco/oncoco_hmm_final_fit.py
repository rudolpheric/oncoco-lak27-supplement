#!/usr/bin/env python3
"""Best-of-N-restart HMM fit for the paper's OnCoCo state summary.

The original ``oncoco_sequence_analysis.py`` fits each condition from a single
initialisation (``random_state=42``). For the chat conditions that seed lands in
a markedly inferior local optimum whose emission rows are near-duplicates, which
is what produced the two indistinguishable "Clarify" states in the published
table. This script refits each condition from many random initialisations, keeps
the best-likelihood solution, orders states by Viterbi occupancy (so state 0 is
the most frequent mode rather than an arbitrary EM index), and additionally
reports the transition matrix, which the original pipeline never wrote out.

Run ``oncoco_hmm_model_selection.py`` first if you want the k sweep behind the
choice of ``--k``.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")

from hmmlearn.hmm import CategoricalHMM  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# paper scope: five conditions (three chat, two mail for RQ3); see paper Sec. "Data"
PAPER_CONDITIONS = [
    "HH_roleplay_chat",
    "HH_real_chat",
    "H_LLM_roleplay_chat",
    "HH_roleplay_mail",
    "HH_CAIA_mail",
]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_SEQ = _load("oncoco_sequence_analysis")
_SEL = _load("oncoco_hmm_model_selection")

COARSE_ORDER = _SEQ.COARSE_ORDER
pretty_condition = _SEQ.pretty_condition


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Best-of-N-restart HMM fit for the OnCoCo state summary.")
    p.add_argument(
        "--data-json",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"
        ),
    )
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "tables"))
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n-restarts", type=int, default=20)
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k-labels", type=int, default=3)
    p.add_argument(
        "--conditions",
        nargs="*",
        default=PAPER_CONDITIONS,
        help="Conditions to fit (default: the five reported in the paper).",
    )
    p.add_argument("--suffix", default="_v2", help="Suffix for output filenames.")
    return p.parse_args()


def occupancy(model: CategoricalHMM, seqs: List[List[int]]) -> np.ndarray:
    X, lengths = _SEL.pack(seqs)
    states = model.predict(X, lengths)
    counts = np.bincount(states, minlength=model.n_components).astype(float)
    return counts / counts.sum()


def main() -> None:
    args = parse_args()
    rows = _SEQ.load_rows(Path(args.data_json))
    sequences = _SEQ.build_sequences(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emission_rows: List[Dict] = []
    transition_rows: List[Dict] = []
    compare_rows: List[Dict] = []

    for condition in args.conditions:
        if condition not in sequences:
            print(f"!! {condition} not in data, skipping", flush=True)
            continue
        int_seqs, _ = _SEL.to_int_sequences(sequences[condition])
        X, lengths = _SEL.pack(int_seqs)

        best, ll, lls, n_conv = _SEL.fit_best(
            int_seqs, args.k, args.n_restarts, args.n_iter, args.tol, args.seed
        )

        # reference: the single-seed configuration used for the published table
        legacy = CategoricalHMM(n_components=args.k, n_iter=200, tol=0.01, random_state=args.seed)
        legacy.fit(X, lengths)
        legacy_ll = float(legacy.score(X, lengths))

        occ = occupancy(best, int_seqs)
        order = np.argsort(-occ)  # most frequent mode first

        compare_rows.append(
            {
                "condition": condition,
                "k": args.k,
                "loglik_best_of_restarts": ll,
                "loglik_single_seed_42": legacy_ll,
                "loglik_gain": ll - legacy_ll,
                "min_pairwise_TV_best": _SEL.min_pairwise_tv(best.emissionprob_),
                "min_pairwise_TV_single_seed": _SEL.min_pairwise_tv(legacy.emissionprob_),
                "n_restarts": len(lls),
                "n_converged": n_conv,
                "n_distinct_optima": int(len(np.unique(np.round(np.array(lls), 2)))),
            }
        )

        for new_idx, old_idx in enumerate(order):
            for li, label in enumerate(COARSE_ORDER):
                emission_rows.append(
                    {
                        "condition": condition,
                        "state": new_idx,
                        "occupancy": float(occ[old_idx]),
                        "label": label,
                        "prob": float(best.emissionprob_[old_idx, li]),
                    }
                )
            for new_j, old_j in enumerate(order):
                transition_rows.append(
                    {
                        "condition": condition,
                        "from_state": new_idx,
                        "to_state": new_j,
                        "prob": float(best.transmat_[old_idx, old_j]),
                    }
                )

        print(
            f"{condition}: logL {legacy_ll:.1f} (seed 42) -> {ll:.1f} (best of {len(lls)}), "
            f"minTV {_SEL.min_pairwise_tv(legacy.emissionprob_):.3f} -> "
            f"{_SEL.min_pairwise_tv(best.emissionprob_):.3f}, occupancy "
            f"{np.round(np.sort(occ)[::-1], 3).tolist()}",
            flush=True,
        )

    em_df = pd.DataFrame(emission_rows)
    tr_df = pd.DataFrame(transition_rows)
    cmp_df = pd.DataFrame(compare_rows)

    em_path = out_dir / f"oncoco_hmm_emissions{args.suffix}.csv"
    tr_path = out_dir / f"oncoco_hmm_transitions{args.suffix}.csv"
    cmp_path = out_dir / f"oncoco_hmm_restart_comparison{args.suffix}.csv"
    tex_path = out_dir / f"oncoco_hmm_state_summary{args.suffix}.tex"
    em_df.to_csv(em_path, index=False)
    tr_df.to_csv(tr_path, index=False)
    cmp_df.to_csv(cmp_path, index=False)

    lines = [
        "\\begin{tabular}{p{0.26\\linewidth} r r p{0.46\\linewidth}}",
        "\\toprule",
        "Condition & State & Occ. & Top labels (prob.) \\\\",
        "\\midrule",
    ]
    for condition in args.conditions:
        sub = em_df[em_df["condition"] == condition]
        if sub.empty:
            continue
        for state in sorted(sub["state"].unique()):
            s = sub[sub["state"] == state].sort_values("prob", ascending=False)
            labels = ", ".join(
                f"{r['label']} ({r['prob']:.2f})" for _, r in s.head(args.top_k_labels).iterrows()
            )
            name = pretty_condition(condition) if state == 0 else ""
            lines.append(f"{name} & {state} & {s['occupancy'].iloc[0]:.2f} & {labels} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nWrote:")
    for f in (em_path, tr_path, cmp_path, tex_path):
        print(" ", f)


if __name__ == "__main__":
    main()
