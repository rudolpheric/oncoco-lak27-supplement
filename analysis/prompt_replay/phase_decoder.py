#!/usr/bin/env python3
"""Decode the HH-real HMM phase of a conversation prefix (for the phase-aware prompt, arm F).

Fits the paper's 3-state HMM on HH real exactly as oncoco_hmm_final_fit.py does (best of 20
restarts, seed 42), orders states by occupancy (0 = clarification, 1 = help/closing,
2 = opening) and Viterbi-decodes the span sequence of a conversation up to a given message.
Every condition is decoded with this one reference model, so "phase" means the same thing
everywhere: the phase a real conversation would be in at that point.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
ONCOCO = ROOT / "analysis/semantic/oncoco"
sys.path.insert(0, str(ONCOCO))

PHASE_NAMES = {0: "klaerung", 1: "hilfe", 2: "eroeffnung"}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ONCOCO / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class PhaseDecoder:
    def __init__(self, master_path: Path | None = None, k: int = 3, n_restarts: int = 20, seed: int = 42):
        self.SEQ = _load("oncoco_sequence_analysis")
        self.SEL = _load("oncoco_hmm_model_selection")
        self.idx = {c: i for i, c in enumerate(self.SEQ.COARSE_ORDER)}
        master_path = master_path or ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"
        self.rows = json.loads(Path(master_path).read_text(encoding="utf-8"))
        ref = self._spans_by_conversation("HH_real_chat")
        ref_int = [[s[0] for s in seq] for seq in ref.values()]
        self.model, ll, _, _ = self.SEL.fit_best(ref_int, k, n_restarts, 500, 1e-4, seed)
        X, lengths = self.SEL.pack(ref_int)
        occ = np.bincount(self.model.predict(X, lengths), minlength=k) / sum(lengths)
        order = list(np.argsort(-occ))
        self.remap = {old: new for new, old in enumerate(order)}
        self.loglik = ll
        self.occupancy = occ[order]
        self.spans = {}  # conv_key -> list of (coarse_idx, message_number, speaker)

    def _spans_by_conversation(self, condition: str, model: str | None = None):
        by_conv = defaultdict(list)
        for r in self.rows:
            if r.get("source") != condition or (model and r.get("model") != model):
                continue
            by_conv[f"{r.get('source_file', '')}::{r.get('msg_learn_counselling_id') or r.get('id')}"].append(r)
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
                        seq.append((self.idx[self.SEQ.role_category(lab)[1]], int(m.get("msg_message_number") or 0),
                                    m.get("speaker_type")))
            if seq:
                out[key] = seq
        return out

    def load_condition(self, condition: str, model: str | None = None) -> None:
        self.spans.update(self._spans_by_conversation(condition, model))

    def decode(self, coarse_ids: list[int]) -> list[int]:
        if not coarse_ids:
            return []
        st = self.model.predict(np.array(coarse_ids).reshape(-1, 1), [len(coarse_ids)])
        return [self.remap[s] for s in st]

    def phase_before(self, conv_key: str, message_number: int) -> int:
        """Phase at the end of the history that precedes `message_number` (2 = opening if empty)."""
        prefix = [c for c, n, _ in self.spans.get(conv_key, []) if n < message_number]
        if not prefix:
            return 2
        return self.decode(prefix)[-1]
