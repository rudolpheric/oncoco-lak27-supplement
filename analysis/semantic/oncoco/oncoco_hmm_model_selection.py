#!/usr/bin/env python3
"""Model selection for the OnCoCo categorical HMMs.

Answers two questions the fixed k=3 / single-seed fit in
``oncoco_sequence_analysis.py`` leaves open:

1. Is k=3 defensible?  We sweep k and report BIC/AIC plus a conversation-level
   cross-validated held-out log-likelihood per observation.
2. Are the fitted states stable and distinct?  Every (condition, k) cell is fit
   from many random restarts; we keep the best-likelihood solution and report
   the spread across restarts, the minimum pairwise total-variation distance
   between emission rows, and the smallest Viterbi state occupancy.

Sequence construction is imported from ``oncoco_sequence_analysis`` so the
sequences are byte-identical to the ones behind the paper's tables.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

# one BLAS thread per worker process; the EM inner loop is Cython and serial
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import CategoricalHMM

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# import the sibling analysis module by path (the package has no __init__.py)
_SPEC = importlib.util.spec_from_file_location(
    "oncoco_sequence_analysis", Path(__file__).with_name("oncoco_sequence_analysis.py")
)
_SEQ_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["oncoco_sequence_analysis"] = _SEQ_MOD
_SPEC.loader.exec_module(_SEQ_MOD)  # type: ignore[union-attr]

COARSE_ORDER = _SEQ_MOD.COARSE_ORDER
build_sequences = _SEQ_MOD.build_sequences
load_rows = _SEQ_MOD.load_rows
pretty_condition = _SEQ_MOD.pretty_condition


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIC / restart-based model selection for OnCoCo HMMs.")
    p.add_argument(
        "--data-json",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"
        ),
    )
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "tables"))
    p.add_argument("--fig-dir", default=str(PROJECT_ROOT / "results" / "figures" / "oncoco"))
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--n-restarts", type=int, default=20)
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--cv-folds", type=int, default=5, help="0 disables cross-validation.")
    p.add_argument("--seed", type=int, default=42, help="Base seed; restart s uses seed+s.")
    p.add_argument("--jobs", type=int, default=10, help="Parallel worker processes over (condition, k) cells.")
    p.add_argument(
        "--tables-only",
        action="store_true",
        help="Skip fitting; rebuild the LaTeX tables from the existing model-selection CSV.",
    )
    return p.parse_args()


# conditions reported in the paper, in paper order
PAPER_CONDITIONS = [
    "HH_roleplay_chat",
    "HH_real_chat",
    "H_LLM_roleplay_chat",
    "HH_roleplay_mail",
    "HH_CAIA_mail",
]


def to_int_sequences(seqs: List[List[Dict[str, str]]]) -> Tuple[List[List[int]], int]:
    """Coarse-label sequences -> integer sequences (same filter as fit_hmm)."""
    label_to_idx = {label: i for i, label in enumerate(COARSE_ORDER)}
    out: List[List[int]] = []
    for seq in seqs:
        labels = [label_to_idx[s["coarse"]] for s in seq if s["coarse"] in label_to_idx]
        if len(labels) >= 2:
            out.append(labels)
    n_symbols_observed = len({x for s in out for x in s})
    return out, n_symbols_observed


def pack(seqs: List[List[int]]) -> Tuple[np.ndarray, List[int]]:
    X = np.concatenate([np.asarray(s) for s in seqs]).reshape(-1, 1)
    return X, [len(s) for s in seqs]


def fit_best(
    seqs: List[List[int]], k: int, n_restarts: int, n_iter: int, tol: float, base_seed: int
) -> Tuple[CategoricalHMM, float, List[float], int]:
    """Fit k-state HMM from n_restarts random inits; return the best-likelihood model."""
    X, lengths = pack(seqs)
    best_model, best_ll = None, -np.inf
    lls: List[float] = []
    n_converged = 0
    for s in range(n_restarts):
        model = CategoricalHMM(
            n_components=k,
            n_features=len(COARSE_ORDER),
            n_iter=n_iter,
            tol=tol,
            random_state=base_seed + s,
            implementation="log",
        )
        try:
            model.fit(X, lengths)
            ll = float(model.score(X, lengths))
        except Exception:
            continue
        if not np.isfinite(ll):
            continue
        lls.append(ll)
        if getattr(model.monitor_, "converged", False):
            n_converged += 1
        if ll > best_ll:
            best_model, best_ll = model, ll
    return best_model, best_ll, lls, n_converged


def n_free_params(k: int, m_eff: int) -> int:
    """startprob (k-1) + transmat k(k-1) + emissions k(m_eff-1)."""
    return (k - 1) + k * (k - 1) + k * (m_eff - 1)


def min_pairwise_tv(emissionprob: np.ndarray) -> float:
    """Smallest total-variation distance between any two emission rows."""
    k = emissionprob.shape[0]
    if k < 2:
        return float("nan")
    return float(
        min(
            0.5 * np.abs(emissionprob[i] - emissionprob[j]).sum()
            for i in range(k)
            for j in range(i + 1, k)
        )
    )


def min_state_occupancy(model: CategoricalHMM, seqs: List[List[int]]) -> float:
    """Smallest share of observations assigned to any state by Viterbi decoding."""
    X, lengths = pack(seqs)
    states = model.predict(X, lengths)
    counts = np.bincount(states, minlength=model.n_components)
    return float(counts.min() / counts.sum())


def cv_heldout_ll(
    seqs: List[List[int]], k: int, folds: int, n_restarts: int, n_iter: int, tol: float, base_seed: int
) -> float:
    """Conversation-level K-fold CV; returns held-out log-likelihood per observation."""
    rng = np.random.default_rng(base_seed)
    idx = rng.permutation(len(seqs))
    assignment = np.array_split(idx, folds)
    total_ll, total_obs = 0.0, 0
    for f in range(folds):
        test_idx = set(assignment[f].tolist())
        train = [s for i, s in enumerate(seqs) if i not in test_idx]
        test = [s for i, s in enumerate(seqs) if i in test_idx]
        if not train or not test:
            return float("nan")
        model, _, _, _ = fit_best(train, k, n_restarts, n_iter, tol, base_seed + 1000 * f)
        if model is None:
            return float("nan")
        X_te, len_te = pack(test)
        try:
            ll = float(model.score(X_te, len_te))
        except Exception:
            return float("nan")
        if not np.isfinite(ll):
            return float("nan")
        total_ll += ll
        total_obs += int(sum(len_te))
    return total_ll / total_obs


def evaluate_cell(task: Tuple) -> Tuple[Dict, List[Dict]]:
    """Fit and score one (condition, k) cell. Module-level so it is picklable."""
    condition, int_seqs, m_eff, k, cfg = task
    n_obs = int(sum(len(s) for s in int_seqs))
    model, ll, lls, n_conv = fit_best(
        int_seqs, k, cfg["n_restarts"], cfg["n_iter"], cfg["tol"], cfg["seed"]
    )
    if model is None:
        return {"condition": condition, "k": k, "failed": True}, []
    p = n_free_params(k, m_eff)
    cv = (
        cv_heldout_ll(
            int_seqs, k, cfg["cv_folds"], max(3, cfg["n_restarts"] // 4), cfg["n_iter"], cfg["tol"], cfg["seed"]
        )
        if cfg["cv_folds"]
        else float("nan")
    )
    rec = {
        "condition": condition,
        "k": k,
        "loglik": ll,
        "n_params": p,
        "n_obs": n_obs,
        "n_seqs": len(int_seqs),
        "m_observed": m_eff,
        "BIC": -2.0 * ll + p * np.log(n_obs),
        "AIC": -2.0 * ll + 2.0 * p,
        "cv_ll_per_obs": cv,
        "min_pairwise_TV": min_pairwise_tv(model.emissionprob_),
        "min_state_occupancy": min_state_occupancy(model, int_seqs),
        "n_restarts": len(lls),
        "n_converged": n_conv,
        "loglik_range": (max(lls) - min(lls)) if lls else float("nan"),
        "n_distinct_optima": int(len(np.unique(np.round(np.array(lls), 2)))) if lls else 0,
        "failed": False,
    }
    emissions = [
        {
            "condition": condition,
            "k": k,
            "state": state,
            "label": label,
            "prob": float(model.emissionprob_[state, li]),
        }
        for state in range(model.n_components)
        for li, label in enumerate(COARSE_ORDER)
    ]
    return rec, emissions


def write_compact_table(df: pd.DataFrame, path: Path, conditions: List[str]) -> None:
    """One row per condition: DeltaBIC across k, plus the BIC- and CV-optimal k."""
    ks = sorted(df["k"].unique())
    header = " & ".join(f"{k}" for k in ks)
    lines = [
        "\\begin{tabular}{l r " + "r " * len(ks) + "r r}",
        "\\toprule",
        f"Condition & Spans & \\multicolumn{{{len(ks)}}}{{c}}{{$\\Delta$BIC by state count $k$}} & "
        "\\multicolumn{2}{c}{Optimal $k$} \\\\",
        f"\\cmidrule(lr){{3-{2 + len(ks)}}} \\cmidrule(lr){{{3 + len(ks)}-{4 + len(ks)}}}",
        f" & & {header} & BIC & CV \\\\",
        "\\midrule",
    ]
    for cond in conditions:
        sub = df[df["condition"] == cond].sort_values("k")
        if sub.empty:
            continue
        dbic = {int(r.k): r.BIC - sub["BIC"].min() for r in sub.itertuples()}
        kbic = int(sub.loc[sub["BIC"].idxmin(), "k"])
        kcv = int(sub.loc[sub["cv_ll_per_obs"].idxmax(), "k"]) if sub["cv_ll_per_obs"].notna().any() else 0
        cells = " & ".join(
            ("\\textbf{0}" if int(k) == kbic else f"{dbic.get(int(k), float('nan')):.0f}") for k in ks
        )
        spans = f"{int(sub['n_obs'].iloc[0]):,}".replace(",", "{,}")  # LaTeX-safe thousands separator
        lines.append(f"{pretty_condition(cond)} & {spans} & {cells} & {kbic} & {kcv} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.tables_only:
        df = pd.read_csv(Path(args.out_dir) / "oncoco_hmm_model_selection.csv")
        conds = [c for c in PAPER_CONDITIONS if c in set(df["condition"])]
        write_compact_table(df, Path(args.out_dir) / "oncoco_hmm_model_selection_compact.tex", conds)
        print(f"Wrote {Path(args.out_dir) / 'oncoco_hmm_model_selection_compact.tex'}")
        return

    rows = load_rows(Path(args.data_json))
    sequences = build_sequences(rows)

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ks = list(range(args.k_min, args.k_max + 1))
    cfg = {
        "n_restarts": args.n_restarts,
        "n_iter": args.n_iter,
        "tol": args.tol,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
    }

    tasks = []
    for condition in sorted(sequences):
        int_seqs, m_eff = to_int_sequences(sequences[condition])
        n_obs = int(sum(len(s) for s in int_seqs))
        print(
            f"{condition}: {len(int_seqs)} sequences, {n_obs} observations, "
            f"{m_eff}/{len(COARSE_ORDER)} symbols observed",
            flush=True,
        )
        for k in ks:
            tasks.append((condition, int_seqs, m_eff, k, cfg))

    records: List[Dict] = []
    emission_records: List[Dict] = []
    print(f"\nFitting {len(tasks)} (condition, k) cells on {args.jobs} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for done, (rec, emissions) in enumerate(pool.map(evaluate_cell, tasks), start=1):
            if rec.get("failed"):
                print(f"[{done}/{len(tasks)}] {rec['condition']} k={rec['k']}: all restarts failed", flush=True)
                continue
            records.append({key: v for key, v in rec.items() if key != "failed"})
            emission_records.extend(emissions)
            print(
                f"[{done}/{len(tasks)}] {rec['condition']} k={rec['k']}: "
                f"logL={rec['loglik']:.1f} BIC={rec['BIC']:.1f} cvLL/obs={rec['cv_ll_per_obs']:.4f} "
                f"minTV={rec['min_pairwise_TV']:.3f} minOcc={rec['min_state_occupancy']:.3f} "
                f"conv={rec['n_converged']}/{rec['n_restarts']} optima={rec['n_distinct_optima']}",
                flush=True,
            )

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "oncoco_hmm_model_selection.csv", index=False)
    pd.DataFrame(emission_records).to_csv(out_dir / "oncoco_hmm_emissions_by_k.csv", index=False)

    # --- BIC / CV curves -----------------------------------------------------
    conds = sorted(df["condition"].unique())
    fig, axes = plt.subplots(2, len(conds), figsize=(3.0 * len(conds), 6.0), sharex=True)
    if len(conds) == 1:
        axes = axes.reshape(2, 1)
    for j, cond in enumerate(conds):
        sub = df[df["condition"] == cond].sort_values("k")
        ax = axes[0, j]
        ax.plot(sub["k"], sub["BIC"] - sub["BIC"].min(), marker="o", color="#2c6fbb")
        ax.axvline(3, color="grey", ls="--", lw=0.8)
        ax.set_title(pretty_condition(cond).replace("--", "–"), fontsize=8)
        ax.set_ylabel(r"$\Delta$BIC" if j == 0 else "")
        ax2 = axes[1, j]
        ax2.plot(sub["k"], sub["cv_ll_per_obs"], marker="s", color="#b5522c")
        ax2.axvline(3, color="grey", ls="--", lw=0.8)
        ax2.set_xlabel("states k")
        ax2.set_ylabel("CV logL / obs" if j == 0 else "")
    fig.tight_layout()
    fig.savefig(fig_dir / "oncoco_hmm_model_selection.png", dpi=200)
    plt.close(fig)

    # --- LaTeX summary table -------------------------------------------------
    lines = [
        "\\begin{tabular}{l r r r r r r}",
        "\\toprule",
        "Condition & $k$ & $\\Delta$BIC & CV logL/obs & min.\\ TV & min.\\ occ. & optima \\\\",
        "\\midrule",
    ]
    for cond in conds:
        sub = df[df["condition"] == cond].sort_values("k")
        best_bic_k = int(sub.loc[sub["BIC"].idxmin(), "k"])
        best_cv_k = int(sub.loc[sub["cv_ll_per_obs"].idxmax(), "k"]) if sub["cv_ll_per_obs"].notna().any() else -1
        for _, r in sub.iterrows():
            k = int(r["k"])
            mark = "$^{\\dagger}$" if k == best_bic_k else ""
            mark += "$^{*}$" if k == best_cv_k else ""
            name = pretty_condition(cond) if k == sub["k"].min() else ""
            lines.append(
                f"{name} & {k}{mark} & {r['BIC'] - sub['BIC'].min():.0f} & {r['cv_ll_per_obs']:.4f} & "
                f"{r['min_pairwise_TV']:.3f} & {r['min_state_occupancy']:.3f} & {int(r['n_distinct_optima'])} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    (out_dir / "oncoco_hmm_model_selection.tex").write_text("\n".join(lines), encoding="utf-8")

    # The compact table is the one the supplement \input{}s (via make_chat_only_tables.py).
    # It used to be written only under --tables-only, so a full refit left it behind: after the
    # 2026-08-13 role repair it still carried HH real chat's pre-repair row (5,066 spans,
    # CV-optimum 6) while every other artifact had moved on. Always write it here.
    write_compact_table(
        df,
        out_dir / "oncoco_hmm_model_selection_compact.tex",
        [c for c in PAPER_CONDITIONS if c in set(df["condition"])],
    )

    print("\nWrote:")
    for f in (
        out_dir / "oncoco_hmm_model_selection.csv",
        out_dir / "oncoco_hmm_emissions_by_k.csv",
        out_dir / "oncoco_hmm_model_selection.tex",
        out_dir / "oncoco_hmm_model_selection_compact.tex",
        fig_dir / "oncoco_hmm_model_selection.png",
    ):
        print(" ", f)


if __name__ == "__main__":
    main()
