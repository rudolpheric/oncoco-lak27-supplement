#!/usr/bin/env python3
"""Turn-level phase test: does the client's turn move the conversation into the help phase, and
how does the client respond given the phase and the counselor's last act?

For every client message the ORIGINAL history before it (what the model saw in the replay) is
decoded with the HH-real HMM (phase_before). Then the client's spans of that message are appended
and decoded again (phase_after). Reported per group:
  * client-led entry rate: P(phase_after = help | phase_before = clarification)
  * client retreat rate:   P(phase_after = clarification | phase_before = help)
  * response profile stratified by phase_before x counselor's last act (asks = Clarify,
    recommends = Help/Resources, other): problem, solution-space, rejection, positive feedback,
    request shares of the client's spans in that message
For HH real and HH roleplay the "original history" is the real one; for the replay arms it is the
history of the original GPT-OSS conversation (identical across arms), and the appended spans come
from the arm.

Output: results/tables/oncoco_phase_transition_test.csv, oncoco_phase_response_profile.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis/prompt_replay"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_decoder import PhaseDecoder  # noqa: E402

LABEL_MAP = json.load(open(Path(__file__).resolve().parent / "label_text_map.json"))
SIG = {
    "problem": lambda t: "problem" in t,
    "solution": lambda t: any(k in t for k in ["recommendation", "resource activation", "implementation"]),
    "request": lambda t: "general request" in t,
    "rejection": lambda t: "rejection" in t,
    "pos_feedback": lambda t: "positive feedback" in t,
    "consent": lambda t: t.startswith("consent"),
}
ACT = {"Clarify": "asks", "Help": "recommends", "Resources": "recommends"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default=str(ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"))
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--conditions", nargs="*", default=["HH_real_chat", "HH_roleplay_chat", "H_LLM_roleplay_chat"])
    p.add_argument("--out-dir", default=str(ROOT / "results/tables"))
    args = p.parse_args()

    dec = PhaseDecoder(Path(args.master))
    rows = list(dec.rows)
    for e in args.extra:
        rows += json.loads(Path(e).read_text(encoding="utf-8"))
    idx = dec.idx

    def messages(condition, model=None):
        conv = defaultdict(dict)
        for r in rows:
            if r.get("source") != condition or (model and r.get("model") != model):
                continue
            key = f"{r.get('source_file', '')}::{r.get('msg_learn_counselling_id') or r.get('id')}"
            sents = sorted(r.get("sentence_classification") or [],
                           key=lambda s: s.get("sentence_index", s.get("start_offset", 0)))
            labs = [s["predicted_label"] for s in sents if s.get("predicted_label")]
            conv[key][int(r.get("msg_message_number") or 0)] = (r.get("speaker_type"), labs)
        return conv

    orig = messages("H_LLM_roleplay_chat", "GPT_OSS_120B")  # the history the replay arms saw
    groups = []
    for c in args.conditions:
        if c == "H_LLM_roleplay_chat":
            groups.append(("GPT-OSS deployment", messages(c, "GPT_OSS_120B"), None))
        elif c.startswith("H_LLM_replay_"):
            groups.append((c.replace("H_LLM_replay_", "arm ").replace("_chat", ""), messages(c), orig))
        else:
            groups.append((c, messages(c), None))

    trans_rows, prof_rows = [], []
    for name, conv, history_src in groups:
        moves = Counter()
        n_before = Counter()
        prof = defaultdict(lambda: Counter())
        n_prof = Counter()
        n_spans = Counter()
        for key, msgs in conv.items():
            hist = (history_src or conv).get(key, msgs)
            for num in sorted(msgs):
                spk, labs = msgs[num]
                if spk != "Client" or not labs:
                    continue
                prefix = [(idx[dec.SEQ.role_category(l)[1]], s) for n in sorted(hist) if n < num
                          for s, l in [(hist[n][0], l2) for l2 in hist[n][1]]]
                coarse_prefix = [c for c, _ in prefix]
                before = dec.decode(coarse_prefix)[-1] if coarse_prefix else 2
                after = dec.decode(coarse_prefix + [idx[dec.SEQ.role_category(l)[1]] for l in labs])[-1]
                n_before[before] += 1
                moves[(before, after)] += 1
                last_co = next((c for c, s in reversed(prefix) if s == "Counsellor"), None)
                act = ACT.get(dec.SEQ.COARSE_ORDER[last_co], "other") if last_co is not None else "none"
                cell = (before, act)
                n_prof[cell] += 1
                for l in labs:
                    t = LABEL_MAP.get(l, l).lower()
                    n_spans[cell] += 1
                    for k, f in SIG.items():
                        if f(t):
                            prof[cell][k] += 1
        for b in (0, 1, 2):
            if n_before[b]:
                trans_rows.append(dict(group=name, phase_before=b, n_client_turns=n_before[b],
                                       **{f"to_phase{a}": moves[(b, a)] / n_before[b] for a in (0, 1, 2)}))
        for cell in sorted(n_prof):
            rec = dict(group=name, phase_before=cell[0], counselor_last=cell[1], n_turns=n_prof[cell],
                       n_client_spans=n_spans[cell])
            for k in SIG:
                rec[k] = prof[cell][k] / max(n_spans[cell], 1)
            prof_rows.append(rec)

    out = Path(args.out_dir)
    t = pd.DataFrame(trans_rows)
    pr = pd.DataFrame(prof_rows)
    t.to_csv(out / "oncoco_phase_transition_test.csv", index=False)
    pr.to_csv(out / "oncoco_phase_response_profile.csv", index=False)
    pd.set_option("display.width", 250)
    print("Client turn moves the phase (rows: phase before the turn; 0 clarification, 1 help, 2 opening):")
    print(t.round(3).to_string(index=False))
    print("\nClient response by phase x counselor's last act:")
    print(pr[pr.n_turns >= 20].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
