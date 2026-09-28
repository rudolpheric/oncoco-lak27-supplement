#!/usr/bin/env python3
"""What do clients do in each HMM phase of real counseling, and who moves the conversation on?

Refits the paper's 3-state HMM on HH real (same best-of-N restarts as oncoco_hmm_final_fit.py),
decodes every conversation of every condition with THAT model, and reports per phase:
  * phase occupancy and the client share of spans in the phase
  * the client's coarse-category profile and the signature labels (problem, solution space,
    request, rejection, consent, positive feedback) within the phase
  * who emits the first span after a phase change (client vs counselor), i.e. who leads

Purpose: ground a phase-aware client prompt (arm F of the prompt-replay experiment) in what real
clients actually do per phase, instead of assuming it.

    .venv/bin/python analysis/semantic/oncoco/oncoco_hmm_phase_profile.py \
        --extra data/processed/prompt_replay/oncoco_classification_O.json ...
Output: results/tables/oncoco_hmm_phase_profile.csv, oncoco_hmm_phase_transitions_by_role.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SEQ = _load("oncoco_sequence_analysis")
SEL = _load("oncoco_hmm_model_selection")
COARSE = SEQ.COARSE_ORDER
LABEL_MAP = json.load(open(HERE / "label_text_map.json"))
REF = "HH_real_chat"
SIG = {  # signature labels on the client side
    "problem": lambda t: "problem" in t,
    "solution": lambda t: any(k in t for k in ["recommendation", "resource activation", "implementation"]),
    "request": lambda t: "general request" in t,
    "rejection": lambda t: "rejection" in t,
    "consent": lambda t: t.startswith("consent"),
    "pos_feedback": lambda t: "positive feedback" in t,
    "own_emotion": lambda t: "own emotional" in t,
    "objective": lambda t: "objective of the assignment" in t,
}


def sequences_with_meta(rows, condition, model=None):
    """conversation -> list of (coarse_idx, speaker, fine_label), span order as in build_sequences."""
    by_conv = defaultdict(list)
    for r in rows:
        if r.get("source") != condition or (model and r.get("model") != model):
            continue
        by_conv[f"{r.get('source_file', '')}::{r.get('msg_learn_counselling_id') or r.get('id')}"].append(r)
    idx = {c: i for i, c in enumerate(COARSE)}
    out = {}
    for key, msgs in by_conv.items():
        def mk(m):
            try:
                return (int(m.get("msg_message_number")), m.get("msg_created_at") or "")
            except Exception:
                return (10 ** 9, m.get("msg_created_at") or "")
        seq = []
        for m in sorted(msgs, key=mk):
            sents = sorted(m.get("sentence_classification") or [],
                           key=lambda s: s.get("sentence_index", s.get("start_offset", 0)))
            for s in sents:
                lab = s.get("predicted_label")
                if lab:
                    seq.append((idx[SEQ.role_category(lab)[1]], m.get("speaker_type"), lab))
        if len(seq) >= 2:
            out[key] = seq
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default=str(ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"))
    p.add_argument("--extra", nargs="*", default=[], help="additional classification JSONs (replay arms)")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n-restarts", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(ROOT / "results/tables"))
    args = p.parse_args()

    rows = json.loads(Path(args.master).read_text(encoding="utf-8"))
    for e in args.extra:
        rows += json.loads(Path(e).read_text(encoding="utf-8"))
    conds = args.conditions or sorted({r["source"] for r in rows if r.get("modality") == "chat"})
    groups = []
    for c in conds:
        if c == "H_LLM_roleplay_chat":
            groups.append((c, "GPT_OSS_120B", "H_LLM [GPT-OSS deployment]"))
        else:
            groups.append((c, None, c))

    # fit on HH real exactly as the paper's final fit
    ref = sequences_with_meta(rows, REF)
    ref_int = [[t[0] for t in s] for s in ref.values()]
    model, ll, lls, n_conv = SEL.fit_best(ref_int, args.k, args.n_restarts, 500, 1e-4, args.seed)
    X, lengths = SEL.pack(ref_int)
    occ = np.bincount(model.predict(X, lengths), minlength=args.k) / sum(lengths)
    order = list(np.argsort(-occ))  # state 0 = most frequent, as in the published table
    remap = {old: new for new, old in enumerate(order)}
    print(f"HH real HMM: loglik {ll:.1f}, occupancy {np.round(occ[order], 3)}")
    em = pd.DataFrame(model.emissionprob_[order], columns=COARSE, index=[f"phase{i}" for i in range(args.k)])
    print(em.round(3).to_string())
    print("transitions per span:\n", pd.DataFrame(model.transmat_[np.ix_(order, order)]).round(3).to_string())

    prof_rows, trans_rows = [], []
    for cond, mdl, name in groups:
        seqs = sequences_with_meta(rows, cond, mdl)
        if not seqs:
            continue
        per_phase = defaultdict(lambda: dict(n=0, n_client=0, coarse=Counter(), sig=Counter()))
        lead = Counter()
        n_change = Counter()
        for key, seq in seqs.items():
            states = [remap[s] for s in model.predict(np.array([[t[0]] for t in seq]), [len(seq)])]
            for i, ((ci, spk, lab), st) in enumerate(zip(seq, states)):
                ph = per_phase[st]
                ph["n"] += 1
                if spk == "Client":
                    ph["n_client"] += 1
                    ph["coarse"][COARSE[ci]] += 1
                    t = LABEL_MAP.get(lab, lab).lower()
                    for k, f in SIG.items():
                        if f(t):
                            ph["sig"][k] += 1
                if i > 0 and st != states[i - 1]:
                    n_change[(states[i - 1], st)] += 1
                    lead[(states[i - 1], st, spk)] += 1
        n_total = sum(v["n"] for v in per_phase.values())
        for st in sorted(per_phase):
            v = per_phase[st]
            rec = dict(group=name, phase=st, occupancy=v["n"] / n_total, n_spans=v["n"],
                       client_share=v["n_client"] / max(v["n"], 1), n_client_spans=v["n_client"])
            for c in COARSE:
                rec[f"client_{c.lower()}"] = v["coarse"][c] / max(v["n_client"], 1)
            for k in SIG:
                rec[f"client_{k}"] = v["sig"][k] / max(v["n_client"], 1)
            prof_rows.append(rec)
        for (a, b), n in sorted(n_change.items()):
            trans_rows.append(dict(group=name, from_phase=a, to_phase=b, n_changes=n,
                                   share_led_by_client=lead[(a, b, "Client")] / n,
                                   changes_per_conversation=n / len(seqs)))

    out = Path(args.out_dir)
    prof = pd.DataFrame(prof_rows)
    trans = pd.DataFrame(trans_rows)
    prof.to_csv(out / "oncoco_hmm_phase_profile.csv", index=False)
    trans.to_csv(out / "oncoco_hmm_phase_transitions_by_role.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["group", "phase", "occupancy", "client_share", "client_clarify", "client_help", "client_resources",
            "client_objectives", "client_other", "client_problem", "client_solution", "client_request",
            "client_rejection", "client_consent", "client_pos_feedback"]
    print(prof[cols].round(3).to_string(index=False))
    print(trans.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
