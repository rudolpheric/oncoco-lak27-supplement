#!/usr/bin/env python3
"""Two robustness checks on the client-side proximity gap Delta_CL = JSD(real, H-LLM) - JSD(real, HH roleplay).

1. Equal-weighted conversations: every conversation contributes its own label distribution with
   weight 1, instead of pooling spans (which weights long conversations more).
2. Persona-matched roleplay: both roleplay conditions restricted to the personas that occur in
   HH roleplay AND H-LLM (persona key as in make_persona_table.py: name + main concern).

Outputs results/tables/oncoco_equal_weight_persona_delta.{csv,tex}. Conversation-level bootstrap.
"""
import json, csv, collections, argparse
from math import sqrt
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NORM = ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"
OUT = ROOT / "results/tables/oncoco_equal_weight_persona_delta"
REF, RP, LLM = "HH_real_chat", "HH_roleplay_chat", "H_LLM_roleplay_chat"


def js_distance(p, q):
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def raw_path(source_file):
    alt = source_file.replace("/chats/", "/chats_filtered/") if "human_llm" in source_file else source_file
    for cand in (alt, source_file):
        if (ROOT / cand).exists():
            return ROOT / cand
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=5000); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = json.loads(NORM.read_text(encoding="utf-8"))
    # client label counts per (condition, source_file, id)
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("source") not in (REF, RP, LLM) or r.get("speaker_type") != "Client":
            continue
        key = (r["source"], r.get("source_file"), str(r.get("id")))
        for s in r.get("sentence_classification") or []:
            if s.get("predicted_label"):
                counts[key][s["predicted_label"]] += 1
    labels = sorted({l for c in counts.values() for l in c})
    idx = {l: i for i, l in enumerate(labels)}
    conv = {k: np.array([c.get(l, 0) for l in labels], float) for k, c in counts.items()}
    by_cond = collections.defaultdict(list)
    for k in conv:
        by_cond[k[0]].append(k)
    print({c: len(v) for c, v in by_cond.items()}, "conversations with client spans;", len(labels), "labels")

    # persona key per roleplay conversation
    persona = {}
    files = collections.defaultdict(set)
    for (cond, sf, cid) in conv:
        if cond in (RP, LLM):
            files[sf].add(cid)
    for sf, ids in files.items():
        p = raw_path(sf)
        if p is None:
            print("  WARNING no raw file for", sf); continue
        for c in json.loads(p.read_text(encoding="utf-8")):
            if str(c.get("id")) not in ids:
                continue
            pers = c.get("persona") or {}
            props = pers.get("properties") or {}
            concern = props.get("Hauptanliegen") or ""
            if isinstance(concern, list):
                concern = " ".join(concern)
            persona[(sf, str(c["id"]))] = ((pers.get("name") or "").strip().lower(), concern.strip()[:120].lower())
    pk = {k: persona.get((k[1], k[2])) for k in conv if k[0] in (RP, LLM)}
    shared = {v for k, v in pk.items() if k[0] == RP and v} & {v for k, v in pk.items() if k[0] == LLM and v}
    matched = {c: [k for k in by_cond[c] if pk.get(k) in shared] for c in (RP, LLM)}
    print(f"{len(shared)} personas shared; matched conversations: RP {len(matched[RP])}, LLM {len(matched[LLM])}")

    def pooled(keys):
        return np.sum([conv[k] for k in keys], axis=0)

    def equal(keys):
        return np.mean([conv[k] / conv[k].sum() for k in keys], axis=0)

    def delta(agg, ref_keys, rp_keys, llm_keys):
        ref = agg(ref_keys)
        return js_distance(ref, agg(llm_keys)) - js_distance(ref, agg(rp_keys)), js_distance(ref, agg(llm_keys)), js_distance(ref, agg(rp_keys))

    rng = np.random.default_rng(args.seed)
    def boot(agg, ref_keys, rp_keys, llm_keys):
        d, dl, dr = delta(agg, ref_keys, rp_keys, llm_keys)
        ds = []
        for _ in range(args.B):
            ds.append(delta(agg, [ref_keys[i] for i in rng.integers(0, len(ref_keys), len(ref_keys))],
                            [rp_keys[i] for i in rng.integers(0, len(rp_keys), len(rp_keys))],
                            [llm_keys[i] for i in rng.integers(0, len(llm_keys), len(llm_keys))])[0])
        lo, hi = np.percentile(ds, [2.5, 97.5])
        return d, dl, dr, lo, hi

    out = []
    for name, agg, rp_keys, llm_keys in [
        ("pooled spans (paper)", pooled, by_cond[RP], by_cond[LLM]),
        ("equal-weighted conversations", equal, by_cond[RP], by_cond[LLM]),
        ("persona-matched, pooled spans", pooled, matched[RP], matched[LLM]),
        ("persona-matched, equal-weighted", equal, matched[RP], matched[LLM]),
    ]:
        d, dl, dr, lo, hi = boot(agg, by_cond[REF], rp_keys, llm_keys)
        out.append(dict(variant=name, n_roleplay=len(rp_keys), n_llm=len(llm_keys), n_personas=(len(shared) if "persona" in name else ""),
                        jsd_llm=dl, jsd_roleplay=dr, delta=d, ci_lo=lo, ci_hi=hi))
        print(f"{name:32s} n_rp={len(rp_keys):3d} n_llm={len(llm_keys):3d}  JSD llm={dl:.3f} rp={dr:.3f}  Delta={d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    with open(OUT.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Variant & HH roleplay conv. & H--LLM conv. & JSD(real, H--LLM) & JSD(real, HH roleplay) & $\Delta_{\mathrm{CL}}$ [95\% CI] \\", r"\midrule"]
    for o in out:
        lines.append(f"{o['variant']} & {o['n_roleplay']} & {o['n_llm']} & {o['jsd_llm']:.3f} & {o['jsd_roleplay']:.3f} & {o['delta']:+.3f} [{o['ci_lo']:+.3f}, {o['ci_hi']:+.3f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT.with_suffix(".csv"), OUT.with_suffix(".tex"))


if __name__ == "__main__":
    main()
