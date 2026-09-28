#!/usr/bin/env python3
"""Does the classifier compress the distance to simulated clients as it compresses the human one?

Reads the blinded coding sheet written by oncoco_blind_sample_build.py after a coder has filled
in the category column, joins it with the hidden key, and computes on identical units:

  1. agreement of the pipeline label with the coder, per condition and for the signature labels,
  2. the client-side distances JSD(HH real, H-LLM) and JSD(HH real, HH roleplay) and their
     difference Delta_CL, once under coder labels and once under pipeline labels,
  3. the compression of each term (coder JSD minus pipeline JSD) with a paired bootstrap,
  4. the coder's agreement with the original expert annotation on the human units (calibration).

H-LLM strata are weighted by their population of client spans, so the pooled H-LLM distribution
mirrors the corpus composition. Units coded as not codable drop out of the distributions.
Resampling is by unit within stratum, the same draw feeding both label sources.

Outputs results/tables/oncoco_blind_sample_validation.{csv,tex} and a printed summary.
"""
from __future__ import annotations

import argparse
import difflib
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ANN = ROOT / "annotations/manual/human_llm"
TABLES = ROOT / "results/tables"
REF, RP, LLM = "HH_real_chat", "HH_roleplay_chat", "H_LLM_roleplay_chat"
SIGNATURE = {
    "Consent": {"CL-IF-ACP-*-Cons-*"},
    "General request": {"CL-IF-ACP-*-Req-*"},
    "Rejection": {"CL-IF-ACP-*-Rej-*"},
    "Problem statement + definition": {"CL-IF-ACP-*-PS-*", "CL-IF-ACP-*-PD-*"},
    "Solution space": {"CL-IF-HP-*-NegFR-*", "CL-IF-HP-*-PosFR-*", "CL-IF-HP-*-RepRA-*",
                       "CL-IF-RA-*-RF-*", "CL-IF-RA-*-RP-*"},
    "Other statements": {"CL-O-*-*-O-*"},
}


def js(p, q) -> float:
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def kappa(a, b) -> float:
    a, b = list(a), list(b)
    labs = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pa = np.array([a.count(l) for l in labs]) / len(a)
    pb = np.array([b.count(l) for l in labs]) / len(b)
    pe = float((pa * pb).sum())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load_coded(tag: str, coder_col: str, sheet_path: Path | None = None) -> pd.DataFrame:
    sheet = pd.read_excel(sheet_path or ANN / f"blind_sample_{tag}.xlsx", sheet_name="Kodierung")
    key = pd.read_csv(ANN / f"blind_sample_{tag}_key.csv")
    de_en = json.loads((HERE / "oncoco_de_en_label_map.json").read_text(encoding="utf-8"))

    def to_code(v):
        """Entries read 'name | code' (German sheet) or 'code | name' (first English sheet)."""
        if not isinstance(v, str) or not v.strip():
            return None
        for part in (p.strip() for p in v.split("|")):
            if part in de_en or part == "X":
                return part
        return None

    raw = sheet[coder_col].astype("string").str.strip()
    code = raw.map(to_code)
    sheet["coder_label"] = code.map(lambda c: de_en.get(c) if isinstance(c, str) else None)
    sheet["coder_uncodable"] = code.eq("X")
    sheet["coded"] = raw.notna() & raw.ne("")
    unknown = sheet[sheet.coded & ~sheet.coder_uncodable & sheet.coder_label.isna()]
    if len(unknown):
        raise SystemExit(f"{len(unknown)} entries not in the category list, e.g. {unknown[coder_col].iloc[0]!r}")
    return key.merge(sheet[["Nr", "coder_label", "coder_uncodable", "coded"]], left_on="unit_no", right_on="Nr")


def weighted_counts(df: pd.DataFrame, col: str, labels: list[str]) -> np.ndarray:
    """Pooled label distribution, each stratum weighted by its client-span population."""
    idx = {l: i for i, l in enumerate(labels)}
    v = np.zeros(len(labels))
    for _, g in df.groupby("stratum"):
        w = g.stratum_population.iloc[0] / len(g)
        for l in g[col]:
            v[idx[l]] += w
    return v


def distances(df: pd.DataFrame, labels: list[str]) -> dict:
    out = {}
    for src, col in (("coder", "coder_label"), ("pipeline", "pipeline_label")):
        d = {c: weighted_counts(df[df.condition == c], col, labels) for c in (REF, RP, LLM)}
        out[f"{src}_llm"] = js(d[REF], d[LLM])
        out[f"{src}_rp"] = js(d[REF], d[RP])
        out[f"{src}_delta"] = out[f"{src}_llm"] - out[f"{src}_rp"]
    out["compression_llm"] = out["coder_llm"] - out["pipeline_llm"]
    out["compression_rp"] = out["coder_rp"] - out["pipeline_rp"]
    out["delta_shift"] = out["coder_delta"] - out["pipeline_delta"]
    return out


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Coder against the original expert label of the best-matching expert unit (human units only)."""
    def norm(t):
        return " ".join(str(t).lower().split())
    experts = {
        RP: pd.read_csv(TABLES / "oncoco_human_agreement_units.csv"),
        REF: pd.read_csv(TABLES / "oncoco_hh_real_agreement_units.csv"),
    }
    rows = []
    for cond, ex in experts.items():
        ex = ex[ex.in_corpus & (ex.role == "K")].dropna(subset=["gold"])
        ex = ex.assign(cid=ex.conversation_id.astype(float).astype(int).astype(str), ntext=ex.text.map(norm))
        by_conv = {c: g for c, g in ex.groupby("cid")}
        for _, u in df[(df.condition == cond) & df.coder_label.notna()].iterrows():
            g = by_conv.get(str(u.conversation_id))
            if g is None:
                continue
            span = norm(u.span)
            if len(span) < 12:          # short spans ("ja", "ok") occur in many units, no unique match
                continue
            hits = [e for _, e in g.iterrows() if span in e.ntext or (len(e.ntext) >= 12 and e.ntext in span)]
            if not hits:
                scored = sorted(((difflib.SequenceMatcher(None, span, e.ntext).ratio(), i) for i, (_, e) in enumerate(g.iterrows())),
                                reverse=True)
                if scored and scored[0][0] >= 0.8 and (len(scored) == 1 or scored[1][0] < scored[0][0] - 0.1):
                    hits = [g.iloc[scored[0][1]]]
            if len(hits) == 1:
                rows.append(dict(condition=cond, unit_no=u.unit_no, coder=u.coder_label, expert=hits[0].gold,
                                 pipeline=u.pipeline_label))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--coder-col", default="Kategorie")
    ap.add_argument("--sheet", default=None, help="coded xlsx; default blind_sample_<tag>.xlsx")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = load_coded(args.tag, args.coder_col, Path(args.sheet) if args.sheet else None)
    print(f"{int(df.coded.sum())} of {len(df)} units coded, {int(df.coder_uncodable.sum())} marked not codable")
    df = df[df.coded & ~df.coder_uncodable].copy()
    labels = sorted(set(df.coder_label) | set(df.pipeline_label))

    rows = []
    for name, sub in [("HH real", df[df.condition == REF]), ("HH roleplay", df[df.condition == RP]),
                      ("H-LLM", df[df.condition == LLM])] + \
                     [(f"H-LLM {m}", df[df.stratum == m]) for m in sorted(df[df.condition == LLM].stratum.unique())]:
        if sub.empty:
            continue
        rec = dict(block="agreement", group=name, n=len(sub),
                   accuracy=float((sub.coder_label == sub.pipeline_label).mean()),
                   kappa=kappa(sub.coder_label, sub.pipeline_label))
        for sig, labs in SIGNATURE.items():
            c, p = sub.coder_label.isin(labs), sub.pipeline_label.isin(labs)
            tp = int((c & p).sum())
            prec = tp / p.sum() if p.sum() else np.nan
            rec_ = tp / c.sum() if c.sum() else np.nan
            rec[f"{sig} coder share"] = float(c.mean())
            rec[f"{sig} pipeline share"] = float(p.mean())
            rec[f"{sig} F1"] = 2 * prec * rec_ / (prec + rec_) if prec and rec_ and prec + rec_ > 0 else np.nan
        rows.append(rec)
        print(f"{name:26s} n={len(sub):3d}  accuracy {rec['accuracy']:.2f}  kappa {rec['kappa']:.2f}")

    point = distances(df, labels)
    rng = np.random.default_rng(args.seed)
    strata = {s: g for s, g in df.groupby("stratum")}
    boots = {k: [] for k in point}
    for _ in range(args.B):
        sample = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in strata.values()])
        for k, v in distances(sample, labels).items():
            boots[k].append(v)
    for k, v in point.items():
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        rows.append(dict(block="distance", group=k, estimate=v, ci_lo=lo, ci_hi=hi))
        print(f"{k:18s} {v:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    cal = calibration(df)
    for cond, g in cal.groupby("condition"):
        rows.append(dict(block="calibration", group=cond, n=len(g),
                         coder_vs_expert_accuracy=float((g.coder == g.expert).mean()),
                         coder_vs_expert_kappa=kappa(g.coder, g.expert),
                         pipeline_vs_expert_kappa=kappa(g.pipeline, g.expert)))
        print(f"calibration {cond}: n={len(g)}, coder-expert kappa {kappa(g.coder, g.expert):.2f}, "
              f"pipeline-expert kappa {kappa(g.pipeline, g.expert):.2f}")

    out = TABLES / f"oncoco_blind_sample_validation_{args.tag}"
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    # distances as point estimates (plug-in JSD is biased upward under resampling),
    # intervals for Delta and for the paired differences, where that bias largely cancels
    names = {"coder": "Blind expert labels", "pipeline": "Pipeline labels"}
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Labels on the same units & JSD(real, H--LLM) & JSD(real, roleplay) & $\Delta_{\mathrm{CL}}$ [95\% CI] \\", r"\midrule"]
    get = {r["group"]: r for r in rows if r["block"] == "distance"}
    for src in ("coder", "pipeline"):
        d = get[f"{src}_delta"]
        cells = [f"{get[f'{src}_llm']['estimate']:.3f}", f"{get[f'{src}_rp']['estimate']:.3f}",
                 f"{d['estimate']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"]
        lines.append(f"{names[src]} & " + " & ".join(cells) + r" \\")
    cells = [f"{get[k]['estimate']:+.3f} [{get[k]['ci_lo']:+.3f}, {get[k]['ci_hi']:+.3f}]"
             for k in ("compression_llm", "compression_rp", "delta_shift")]
    lines += [r"\midrule", "Expert minus pipeline & " + " & ".join(cells) + r" \\", r"\bottomrule", r"\end{tabular}"]
    out.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out.with_suffix(".csv"), out.with_suffix(".tex"))


if __name__ == "__main__":
    main()
