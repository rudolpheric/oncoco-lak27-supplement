#!/usr/bin/env python3
"""Message length and surface register per condition ("Verbosity against instruction").

Reproduces the numbers in the paper's verbosity paragraph so they have a generating script:
  median words per client message per condition, per model, per persona brevity flag and
  per template family; the share of client messages without sentence-final punctuation;
  the share of consecutive same-speaker messages (is the per-message gap also the per-turn
  gap?); and a persona audit (how many personas, how many carry a brevity cue).

Input : data/processed/combined/normalized/oncoco_classification_all.json  (one row per message)
        data/raw/human_llm/chats/**/*.json                                  (via oncoco_prompt_meta)
Output: results/tables/oncoco_verbosity.csv / .tex
        results/tables/oncoco_verbosity_personas.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from oncoco_prompt_meta import (  # noqa: E402
    MODEL_LABEL, client_word_count, has_final_punct, load_hllm_prompt_meta,
)

NORM = ROOT / "data/processed/combined/normalized"
TABLES = ROOT / "results/tables"
CHAT = ["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat"]
SHORT = {"HH_roleplay_chat": "HH roleplay", "HH_real_chat": "HH real", "H_LLM_roleplay_chat": "H--LLM"}
EXPECTED = {"HH_roleplay_chat": 68, "HH_real_chat": 53, "H_LLM_roleplay_chat": 414}


def load_messages() -> pd.DataFrame:
    rows = json.loads((NORM / "oncoco_classification_all.json").read_text(encoding="utf-8"))
    recs = []
    for r in rows:
        if r.get("source") not in CHAT:
            continue
        recs.append(dict(
            source=r["source"], conv_key=f"{r.get('source_file', '')}::{r.get('id')}",
            msg_no=int(r.get("msg_message_number") or 0), speaker=r.get("speaker_type"),
            model=r.get("model") or "", text=r.get("msg_content") or "",
        ))
    df = pd.DataFrame(recs)
    df["words"] = df["text"].map(client_word_count)
    df["final_punct"] = df["text"].map(has_final_punct)
    for c, n in EXPECTED.items():
        seen = df.loc[df["source"] == c, "conv_key"].nunique()
        if seen != n:
            raise SystemExit(f"{c}: {seen} conversations, expected {n}")
    return df


def q(s: pd.Series) -> tuple[float, float, float]:
    return float(s.median()), float(s.quantile(.25)), float(s.quantile(.75))


def main() -> None:
    df = load_messages()
    meta = load_hllm_prompt_meta()
    cl = df[df["speaker"] == "Client"].copy()
    cl = cl.join(meta[["persona_name", "brevity_persona", "template_family", "model_label"]], on="conv_key")

    out = []

    def add(group: str, level: str, s: pd.DataFrame) -> None:
        if not len(s):
            return
        med, q1, q3 = q(s["words"])
        out.append(dict(group=group, level=level, n_conversations=s["conv_key"].nunique(),
                        n_client_msgs=len(s), median_words=med, q1_words=q1, q3_words=q3,
                        share_no_final_punct=float((~s["final_punct"]).mean())))

    for c in CHAT:
        add("condition", SHORT[c], cl[cl["source"] == c])
    h = cl[cl["source"] == "H_LLM_roleplay_chat"]
    for m, lab in MODEL_LABEL.items():
        add("model", lab, h[h["model"] == m])
    for flag in (True, False):
        add("brevity_persona", "yes" if flag else "no", h[h["brevity_persona"] == flag])
        for m, lab in MODEL_LABEL.items():
            add("model x brevity_persona", f"{lab} / {'yes' if flag else 'no'}",
                h[(h["model"] == m) & (h["brevity_persona"] == flag)])
    for fam in ("plain", "chat", "not logged"):
        add("template_family", fam, h[h["template_family"] == fam])
    llama = h[h["model"] == "LLama3_3_70B"]
    for fam in ("plain", "chat"):
        add("Llama x template_family", fam, llama[llama["template_family"] == fam])

    res = pd.DataFrame(out)
    res.to_csv(TABLES / "oncoco_verbosity.csv", index=False)

    # consecutive same-speaker messages: if rare, per-message length is per-turn length
    same = []
    for c in CHAT:
        d = df[df["source"] == c].sort_values(["conv_key", "msg_no"])
        prev_same = (d["speaker"] == d.groupby("conv_key")["speaker"].shift(1))
        same.append(dict(condition=SHORT[c], share_consecutive_same_speaker=float(prev_same.mean())))
    same = pd.DataFrame(same)

    # persona audit (analysed H-LLM conversations only)
    hm = meta.loc[meta.index.intersection(h["conv_key"].unique())]
    personas = (hm.groupby("persona_name")
                  .agg(n_conversations=("brevity_persona", "size"),
                       brevity_persona=("brevity_persona", "max"),
                       models=("model", lambda s: ";".join(sorted(set(s)))),
                       sprachliche_merkmale=("sprachliche_merkmale", "first"))
                  .sort_values("n_conversations", ascending=False))
    personas.to_csv(TABLES / "oncoco_verbosity_personas.csv")
    n_brief = hm.groupby(["persona_name", "sprachliche_merkmale"]).ngroups

    # ---- console block mirroring the paper paragraph ----
    print("Median words per client message")
    print(res[res["group"].isin(["condition", "model", "brevity_persona", "model x brevity_persona",
                                 "template_family", "Llama x template_family"])]
          .to_string(index=False))
    print("\nConsecutive same-speaker messages")
    print(same.to_string(index=False))
    print(f"\nPersonas in the analysed H-LLM conversations: {len(personas)} names, "
          f"{n_brief} name+profile combinations, "
          f"{int(personas['brevity_persona'].sum())} names with a brevity cue in 'Sprachliche Merkmale'")
    conv_flag = hm["brevity_persona"].mean()
    print(f"Share of H-LLM conversations under a brevity-cue persona: {conv_flag:.1%}")

    # ---- supplement table ----
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Grouping & Level & Conv. & Client msgs & Median words [IQR] & No final punct. \\",
             r"\midrule"]
    last = None
    for _, r in res.iterrows():
        if r["group"] == "model x brevity_persona":
            continue
        lead = r["group"].replace("_", r"\_") if r["group"] != last else ""
        last = r["group"]
        lines.append(f"{lead} & {r['level']} & {int(r['n_conversations'])} & {int(r['n_client_msgs'])} & "
                     f"{r['median_words']:.0f} [{r['q1_words']:.0f}, {r['q3_words']:.0f}] & "
                     f"{r['share_no_final_punct'] * 100:.1f}\\% " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "oncoco_verbosity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    same.to_csv(TABLES / "oncoco_verbosity_turns.csv", index=False)
    print(f"\nWrote {TABLES / 'oncoco_verbosity.csv'}, .tex, _personas.csv, _turns.csv")


if __name__ == "__main__":
    main()
