#!/usr/bin/env python3
"""Generate the supplement tables listing the FDR-significant transition cells.

The main paper reports only the counts (70 of 222, 19 of 125) and the three largest
shifts. These tables let a reader look up every cell that passes Benjamini-Hochberg,
derived from oncoco_transition_fdr.csv so a regeneration run leaves no hand-edited
numbers behind. The 70-cell H-LLM table is set as two side-by-side halves so it fits
one supplement page.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "results" / "tables"

MIN_P = 0.0005  # smallest attainable empirical p with 2,000 resamples


def fmt_p(p: float) -> str:
    if p <= MIN_P:
        return r"$<$0.001"
    return f"{p:.3f}"


def row_cells(r: pd.Series) -> list[str]:
    return [
        r["from"],
        r["to"],
        f"{r['p_comp']:.3f}",
        f"{r['p_ref']:.3f}",
        f"${r['delta']:+.3f}$",
        str(int(r["combined_count"])),
        fmt_p(r["p_value"]),
    ]


def load_sig(comparison: str) -> pd.DataFrame:
    df = pd.read_csv(T / "oncoco_transition_fdr.csv")
    sub = df[(df.comparison == comparison) & df.fdr_significant].copy()
    sub["absdelta"] = sub.delta.abs()
    return sub.sort_values("absdelta", ascending=False).reset_index(drop=True)


def write_two_column(df: pd.DataFrame, comp_label: str, out: Path) -> None:
    half = (len(df) + 1) // 2
    header = f"From & To & {comp_label} & HH real & $\\Delta$ & $n$ & $p$"
    lines = [
        r"\begin{tabular}{llrrrrr@{\hspace{1.5em}}llrrrrr}",
        r"\toprule",
        f"{header} & {header} \\\\",
        r"\midrule",
    ]
    for i in range(half):
        left = row_cells(df.iloc[i])
        j = i + half
        right = row_cells(df.iloc[j]) if j < len(df) else [""] * 7
        lines.append(" & ".join(left + right) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(df)} cells)")


def write_single(df: pd.DataFrame, comp_label: str, out: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        f"From & To & {comp_label} & HH real & $\\Delta$ & $n$ & $p$ \\\\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(" & ".join(row_cells(r)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(df)} cells)")


def main() -> None:
    hllm = load_sig("H_LLM_roleplay_chat")
    roleplay = load_sig("HH_roleplay_chat")
    write_two_column(hllm, "H--LLM", T / "oncoco_transition_fdr_sig_hllm.tex")
    write_single(roleplay, "HH roleplay", T / "oncoco_transition_fdr_sig_roleplay.tex")


if __name__ == "__main__":
    main()
