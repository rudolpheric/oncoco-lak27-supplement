#!/usr/bin/env python3
"""Generate the summary .tex tables that previously had no generator.

Five tables the paper and supplement \\input were maintained by hand and therefore silently
went stale whenever the corpus was rebuilt: the transition summary, the within-condition JSD
summary, the persistence summary, the two label-distribution tables, and the H-LLM evolution
summary. This script derives all of them from the CSVs the analysis scripts write, so a
regeneration run leaves no hand-edited numbers behind.

Run AFTER the analysis scripts and BEFORE make_chat_only_tables.py, which derives the
`*_chat` variants from the tables written here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "results" / "tables"

DISPLAY = {
    "HH_roleplay_chat": "HH roleplay chat",
    "HH_real_chat": "HH real chat",
    "H_LLM_roleplay_chat": "H--LLM roleplay chat",
    "HH_roleplay_mail": "HH roleplay mail",
    "HH_real_mail": "HH real mail",
    "HH_CAIA_mail": "HH+assistant (CAIA) mail",
    "LLM_LLM_mail": "LLM--LLM mail",
}
MODEL_DISPLAY = {"GPT_OSS_120B": "GPT-OSS-120B", "LLama3_3_70B": "Llama 3.3 70B", "Mixtral": "Mixtral 8x7B"}
ORDER = ["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat",
         "HH_roleplay_mail", "HH_real_mail", "HH_CAIA_mail", "LLM_LLM_mail"]


def esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write(name: str, header: str, rows: list[str], colspec: str) -> None:
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", header, r"\midrule",
             *rows, r"\bottomrule", r"\end{tabular}"]
    (T / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {name}")


def transition_summary() -> None:
    d = pd.read_csv(T / "oncoco_transition_matrix.csv")
    d["from_role"] = d["from"].str.split(":").str[0]
    d["to_role"] = d["to"].str.split(":").str[0]
    rows = []
    for cond in [c for c in ORDER if c in set(d["condition"])]:
        s = d[d["condition"] == cond]
        tot = s["count"].sum()
        if tot <= 0:
            continue
        same = s[s["from_role"] == s["to_role"]]["count"].sum() / tot
        self_t = s[s["from"] == s["to"]]["count"].sum() / tot
        rows.append(f"{DISPLAY.get(cond, cond)} & {same:.3f} & {1 - same:.3f} & {self_t:.3f} \\\\")
    write("oncoco_transition_summary.tex",
          r"Condition & Same-role & Cross-role & Self-transition \\",
          rows, r"p{0.38\linewidth}ccc")


def within_condition_summary() -> None:
    d = pd.read_csv(T / "oncoco_within_condition_jsd.csv")
    d["frac"] = d["outlier_count"] / d["n_conversations"]
    rows = []
    for cond in [c for c in ORDER if c in set(d["condition"])]:
        for sp in ("Client", "Counsellor"):
            s = d[(d["condition"] == cond) & (d["speaker_type"] == sp)]
            if s.empty:
                continue
            r = s.iloc[0]
            label = "Counselor" if sp == "Counsellor" else sp
            rows.append(f"{DISPLAY.get(cond, cond)} & {label} & {r.median_jsd:.3f} & "
                        f"{r.p90_jsd:.3f} & {r.frac:.3f} \\\\")
    write("oncoco_within_condition_jsd_summary.tex",
          r"Condition & Speaker & Median JSD & P90 JSD & Outlier frac \\",
          rows, "llrrr")


def persistence_summary() -> None:
    d = pd.read_csv(T / "oncoco_persistence_top_labels.csv")
    rows = []
    for comp in d["comparison"].unique():
        for sp in ("Client", "Counsellor"):
            s = d[(d["comparison"] == comp) & (d["speaker_type"] == sp)]
            for _, r in s.iterrows():
                a, b = comp.split(" vs ")
                label = "Counselor" if sp == "Counsellor" else sp
                rows.append(
                    f"{DISPLAY.get(a, a)} vs {DISPLAY.get(b, b)} & {label} & "
                    f"{esc(r.label_text)} & {r.median_diff:+.3f} & {r.auc:.3f} & "
                    f"{r.p_a_gt_b_median:.3f} \\\\")
    write("oncoco_persistence_summary.tex",
          r"Comparison & Speaker & Label & Med. diff & AUC & P(a$>$med) \\",
          rows, r"p{0.30\linewidth} l p{0.40\linewidth} r r r")


def label_distributions(min_share: float = 0.01) -> None:
    d = pd.read_csv(T / "oncoco_label_distribution.csv")
    try:
        import json
        text_map = json.loads((Path(__file__).with_name("label_text_map.json")).read_text(encoding="utf-8"))
    except Exception:
        text_map = {}
    conds = ["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat"]
    for sp, out in (("Client", "oncoco_label_dist_client_chat.tex"),
                    ("Counsellor", "oncoco_label_dist_counselor_chat.tex")):
        s = d[(d["speaker_type"] == sp) & (d["condition"].isin(conds))]
        piv = s.pivot_table(index="label", columns="condition", values="proportion", fill_value=0.0)
        for c in conds:
            if c not in piv.columns:
                piv[c] = 0.0
        piv = piv[piv.max(axis=1) >= min_share].sort_values("H_LLM_roleplay_chat", ascending=False)
        rows = [f"{esc(text_map.get(lab, lab))} & {r['HH_roleplay_chat']:.3f} & "
                f"{r['HH_real_chat']:.3f} & {r['H_LLM_roleplay_chat']:.3f} \\\\"
                for lab, r in piv.iterrows()]
        write(out, r"Label & HH rp & HH real & H--LLM \\", rows, r"p{0.52\linewidth}rrr")


def evolution_summary() -> None:
    d = pd.read_csv(T / "oncoco_hllm_model_realness_both_baselines.csv")
    piv = d.pivot_table(index=["model", "first_seen_month"], columns="speaker_type",
                        values=["jsd_to_HH_real_chat", "jsd_to_HH_roleplay_chat"])
    rows = []
    for (model, seen), r in piv.sort_values("first_seen_month" if "first_seen_month" in piv.index.names else piv.index.names[1]).iterrows():
        def g(metric, sp):
            for cand in (sp, "Counselor" if sp == "Counsellor" else sp):
                if (metric, cand) in r.index:
                    return r[(metric, cand)]
            return float("nan")
        rows.append(f"{MODEL_DISPLAY.get(model, model)} & {seen} & "
                    f"{g('jsd_to_HH_real_chat', 'Client'):.3f} & {g('jsd_to_HH_real_chat', 'Counsellor'):.3f} & "
                    f"{g('jsd_to_HH_roleplay_chat', 'Client'):.3f} & {g('jsd_to_HH_roleplay_chat', 'Counsellor'):.3f} \\\\")
    write("oncoco_hllm_evolution_summary.tex",
          r"Model & Seen & C$\rightarrow$real & CO$\rightarrow$real & C$\rightarrow$roleplay & CO$\rightarrow$roleplay \\",
          rows, r"p{0.19\linewidth}ccccc")


if __name__ == "__main__":
    transition_summary()
    within_condition_summary()
    persistence_summary()
    label_distributions()
    evolution_summary()
