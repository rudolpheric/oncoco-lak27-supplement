#!/usr/bin/env python3
"""LaTeX tables for the prompt-replay experiment (paper RQ3 and supplement).

Reads results/tables/oncoco_prompt_replay.csv (span unit), oncoco_prompt_replay_message.csv
(message unit) and oncoco_phase_transition_test.csv (turn-level phase test) and writes
  results/tables/oncoco_prompt_replay_paper.tex        main-text table: human references + O, B, C, D, G
  results/tables/oncoco_prompt_replay_supplement.tex   all arms, span and message JSD
  results/tables/oncoco_phase_transition_test.tex      client-led phase moves per group
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "results/tables"

NAMES = {
    "HH_real_chat": ("", "HH real"),
    "HH_roleplay_chat": ("", "HH roleplay"),
    "H_LLM_replay_O_chat": ("O", "original prompt, re-classified"),
    "H_LLM_replay_A_chat": ("A", "logged prompt, replayed"),
    "H_LLM_replay_B_chat": ("B", "generic realism prompt"),
    "H_LLM_replay_C_chat": ("C", "measurement-based rules"),
    "H_LLM_replay_D_chat": ("D", "C, two rules dosed"),
    "H_LLM_replay_E_chat": ("E", "D, praise allowed"),
    "H_LLM_replay_F_chat": ("F", "phase-aware rules"),
    "H_LLM_replay_G_chat": ("G", "phase-dependent rules"),
}
PAPER_ROWS = ["HH_real_chat", "HH_roleplay_chat", "H_LLM_replay_O_chat", "H_LLM_replay_B_chat",
              "H_LLM_replay_C_chat", "H_LLM_replay_D_chat", "H_LLM_replay_G_chat"]
ALL_ROWS = ["HH_real_chat", "HH_roleplay_chat", "H_LLM_replay_O_chat", "H_LLM_replay_A_chat",
            "H_LLM_replay_B_chat", "H_LLM_replay_C_chat", "H_LLM_replay_D_chat", "H_LLM_replay_E_chat",
            "H_LLM_replay_F_chat", "H_LLM_replay_G_chat"]
TRANS_NAMES = {"HH_real_chat": "HH real", "HH_roleplay_chat": "HH roleplay", "GPT-OSS deployment": "GPT-OSS-120B deployment",
               "arm O": "O", "arm A": "A", "arm B": "B", "arm C": "C", "arm D": "D", "arm E": "E", "arm F": "F", "arm G": "G"}


def label(group: str) -> str:
    code, name = NAMES[group]
    return f"{code}: {name}" if code else name


def ci(r, col: str, nd: int = 3) -> str:
    if pd.isna(r[col]):
        return ""
    return f"{r[col]:.{nd}f} [{r[col + '_lo']:.{nd}f}, {r[col + '_hi']:.{nd}f}]"


def delta(r) -> str:
    if pd.isna(r["delta"]):
        return ""
    d = 0.0 if abs(r["delta"]) < 0.0005 else r["delta"]
    return f"{d:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"


def pct(v) -> str:
    return "" if pd.isna(v) else f"{100 * v:.0f}\\%"


def f3(v) -> str:
    return "" if pd.isna(v) else f"{v:.3f}"


def main() -> None:
    span = pd.read_csv(T / "oncoco_prompt_replay.csv").set_index("group")
    msg = pd.read_csv(T / "oncoco_prompt_replay_message.csv").set_index("group")
    trans = pd.read_csv(T / "oncoco_phase_transition_test.csv")
    entry = {TRANS_NAMES.get(g, g): v for g, v in
             trans[trans.phase_before == 0].set_index("group")["to_phase1"].items()}
    retreat = {TRANS_NAMES.get(g, g): v for g, v in
               trans[trans.phase_before == 1].set_index("group")["to_phase0"].items()}

    def entry_of(group: str) -> float:
        key = NAMES[group][0] or NAMES[group][1]
        return entry.get(key, np.nan)

    # main-text table
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Arm & JSD to HH real [95\% CI] & $\Delta$ [95\% CI] & Problem d1 & Solution & Rej.\ conv. & Neg:Pos & Words & Entry \\",
             r"\midrule"]
    for g in PAPER_ROWS:
        r = span.loc[g]
        lines.append(f"{label(g)} & {ci(r, 'jsd_real')} & {delta(r)} & {f3(r['problem_d1'])} & {f3(r['solution_pooled'])} & "
                     f"{pct(r['share_conv_rejection'])} & {r['neg_pos_ratio']:.2f} & {r['median_words']:.0f} & "
                     f"{100 * entry_of(g):.1f}\\% \\\\")
        if g == "HH_roleplay_chat":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / "oncoco_prompt_replay_paper.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # supplement table: all arms, both units
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r"Arm & JSD spans [95\% CI] & JSD messages [95\% CI] & $V$ & Problem d1 & Solution & Req.\ conv. & Rej.\ conv. & Neg:Pos & Words \\",
             r"\midrule"]
    for g in ALL_ROWS:
        r, m = span.loc[g], msg.loc[g]
        v = "" if pd.isna(r["cramers_v"]) else f"{r['cramers_v']:.2f}"
        lines.append(f"{label(g)} & {ci(r, 'jsd_real')} & {ci(m, 'jsd_real')} & {v} & {f3(r['problem_d1'])} & "
                     f"{f3(r['solution_pooled'])} & {pct(r['share_conv_request'])} & {pct(r['share_conv_rejection'])} & "
                     f"{r['neg_pos_ratio']:.2f} & {r['median_words']:.0f} \\\\")
        if g == "HH_roleplay_chat":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / "oncoco_prompt_replay_supplement.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # phase transition table
    order = ["HH real", "HH roleplay", "GPT-OSS-120B deployment", "O", "A", "B", "C", "D", "E", "F", "G"]
    n_clar = {TRANS_NAMES.get(g, g): v for g, v in trans[trans.phase_before == 0].set_index("group")["n_client_turns"].items()}
    n_help = {TRANS_NAMES.get(g, g): v for g, v in trans[trans.phase_before == 1].set_index("group")["n_client_turns"].items()}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Group & Clarification turns & Client-led entry into help & Help-phase turns & Client-led return to clarification \\",
             r"\midrule"]
    for g in order:
        if g not in entry:
            continue
        lines.append(f"{g} & {int(n_clar[g])} & {100 * entry[g]:.1f}\\% & {int(n_help.get(g, 0))} & {100 * retreat.get(g, np.nan):.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / "oncoco_phase_transition_test.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", T / "oncoco_prompt_replay_paper.tex", T / "oncoco_prompt_replay_supplement.tex",
          T / "oncoco_phase_transition_test.tex")


if __name__ == "__main__":
    main()
