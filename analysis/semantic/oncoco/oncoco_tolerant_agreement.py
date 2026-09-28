#!/usr/bin/env python3
"""Agreement under successively more tolerant readings of the code book.

Not every disagreement between the classifier and the coders is an error. Some
category pairs are close enough that the model's answer is defensible. This
script scores agreement on a ladder of neighbourhood definitions, from label-for-
label to a reading that treats the whole problem-elaboration family as one, and
reports what is left over at each step.

Widening is a judgment call, so every step is listed explicitly and the residual
is broken down, rather than reporting one tolerant number on its own.

Reads the unit tables from the two agreement scripts.
Writes results/tables/oncoco_tolerant_agreement.csv and _residual.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/tables"
LABEL_TEXT = json.loads((Path(__file__).resolve().parent / "label_text_map.json").read_text())

BACKCHANNEL_CO = {"CO-Mod-*-*-*-*", "CO-O-*-*-O-*"}
BACKCHANNEL_CL = {"CL-IF-ACP-*-Cons-*", "CL-O-*-*-O-*",
                  "CL-IF-HP-*-PosF-*", "CL-IF-HP-*-PosFR-*"}
REFLECTION = {"CO-IF-HP-*-IEA", "CO-IF-AC-RF-SRx-*", "CO-IF-AC-RE-RCR-*"}
QUESTIONS = {"CO-IF-AC-RF-RTP-*", "CO-IF-AC-RF-RC-*", "CO-IF-AC-RE-RES-*",
             "CO-IF-AC-RF-RCD-*", "CO-IF-AC-RF-RPA-*", "CO-IF-AC-RF-RPD-*",
             "CO-IF-AO-*-ROW-*", "CO-IF-Mot-*-RS-*", "CO-IF-RA-*-RP-*"}
PS, PD = "CL-IF-ACP-*-PS-*", "CL-IF-ACP-*-PD-*"
OE, FPA = "CL-IF-ACP-*-OE-*", "CL-IF-ACP-*-FPA-*"
UCO = {"CO-O-*-*-UCO-*", "CL-O-*-*-UCO-*"}

LADDER = [
    ("Label für Label", []),
    ("+ Backchannel, Reflexion, Fragetypen",
     [BACKCHANNEL_CO, BACKCHANNEL_CL, REFLECTION, QUESTIONS]),
    ("+ Problemdarstellung/-definition",
     [BACKCHANNEL_CO, BACKCHANNEL_CL, REFLECTION, QUESTIONS, {PS, PD}]),
    ("+ eigene Gefühlsdarstellung",
     [BACKCHANNEL_CO, BACKCHANNEL_CL, REFLECTION, QUESTIONS, {PS, PD, OE}]),
    ("+ Rückmeldung zu Lösungsversuchen",
     [BACKCHANNEL_CO, BACKCHANNEL_CL, REFLECTION, QUESTIONS, {PS, PD, OE, FPA}]),
]


def equivalences(groups) -> dict[str, set[str]]:
    eq: dict[str, set[str]] = {}
    for g in groups:
        for a in g:
            eq.setdefault(a, set()).update(g)
    return eq


def kappa(gold, pred) -> float:
    labs = sorted(set(gold) | set(pred))
    i = {l: k for k, l in enumerate(labs)}
    m = np.zeros((len(labs), len(labs)))
    for x, y in zip(gold, pred):
        m[i[x], i[y]] += 1
    n = len(gold)
    po = np.trace(m) / n
    pe = float((m.sum(0) / n) @ (m.sum(1) / n))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load() -> pd.DataFrame:
    frames = []
    for name, f in (("HH-Rollenspiel", "oncoco_human_agreement_units.csv"),
                    ("HH-real", "oncoco_hh_real_agreement_units.csv")):
        d = pd.read_csv(OUT / f)
        d = d[d.in_corpus].dropna(subset=["gold", "pred"])
        d = d.assign(corpus=name)
        frames.append(d[["corpus", "role", "gold", "pred", "text"]])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    u = load()
    rows, residual_rows = [], []
    for step, groups in LADDER:
        eq = equivalences(groups)
        ok = np.array([g == p or p in eq.get(g, ()) for g, p in zip(u.gold, u.pred)])
        # for kappa, a defensible prediction is folded onto the gold label
        pred_tol = [g if p in eq.get(g, ()) else p for g, p in zip(u.gold, u.pred)]
        for corpus in ("HH-Rollenspiel", "HH-real", "beide"):
            m = np.ones(len(u), bool) if corpus == "beide" else (u.corpus == corpus).values
            rows.append(dict(step=step, corpus=corpus, units=int(m.sum()),
                             accuracy=float(ok[m].mean()),
                             kappa=kappa(u.gold[m], pd.Series(pred_tol)[m])))
        rest = u[~ok]
        residual_rows.append(dict(
            step=step, residual_units=len(rest), share_of_all=len(rest) / len(u),
            model_says_inappropriate=int(rest.pred.isin(UCO).sum()),
            counselor_side=float((rest.role == "B").mean()),
            largest_single_confusion=int(rest.groupby(["gold", "pred"]).size().max()) if len(rest) else 0,
        ))
    t = pd.DataFrame(rows)
    r = pd.DataFrame(residual_rows)
    t.to_csv(OUT / "oncoco_tolerant_agreement.csv", index=False)
    r.to_csv(OUT / "oncoco_tolerant_agreement_residual.csv", index=False)

    fmt = lambda v: f"{v:.3f}"
    print(t.pivot_table(index="step", columns="corpus", values=["accuracy", "kappa"], sort=False)
           .to_string(float_format=fmt))
    print()
    print(r.to_string(index=False, float_format=fmt))

    # what the widest reading leaves behind
    eq = equivalences(LADDER[-1][1])
    ok = np.array([g == p or p in eq.get(g, ()) for g, p in zip(u.gold, u.pred)])
    rest = u[~ok]
    print()
    print(f"Beim weitesten Zuschnitt bleiben {len(rest)} Einheiten ({len(rest)/len(u):.1%}).")
    top = rest.groupby(["gold", "pred"]).size().sort_values(ascending=False).head(12)
    for (g, p), n in top.items():
        print(f"  {n:4d}  {LABEL_TEXT.get(g, g)[:44]:46s} -> {LABEL_TEXT.get(p, p)}")


if __name__ == "__main__":
    main()
