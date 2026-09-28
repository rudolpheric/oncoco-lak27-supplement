#!/usr/bin/env python3
"""Student-clustered intervals for the proximity gaps and the per-model distances to HH real.

The 414 H-LLM conversations come from 281 students (platform course_member_id, unique within a
raw export). Conversation-level resampling treats a student's repeated conversations as
independent draws. This script resamples students instead, so all conversations of a student
enter or leave a bootstrap sample together, and reports both intervals side by side.
HH real and HH roleplay are resampled by conversation: HH roleplay ran under a shared course
account, so repeated participation cannot be reconstructed there.

Outputs results/tables/oncoco_student_clustered_bootstrap.{csv,tex}.
"""
import argparse
import collections
import csv
import json
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NORM = ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"
OUT = ROOT / "results/tables/oncoco_student_clustered_bootstrap"
REF, RP, LLM = "HH_real_chat", "HH_roleplay_chat", "H_LLM_roleplay_chat"
MODELS = [("GPT_OSS_120B", "GPT-OSS-120B"), ("LLama3_3_70B", "Llama 3.3 70B"), ("Mixtral", "Mixtral 8x7B")]
SPEAKERS = [("Client", "Client"), ("Counsellor", "Counselor")]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = json.loads(NORM.read_text(encoding="utf-8"))

    # label counts per (speaker, condition, source_file, id), model per H-LLM conversation
    counts = collections.defaultdict(collections.Counter)
    model_of = {}
    for r in rows:
        if r.get("source") not in (REF, RP, LLM):
            continue
        key = (r["source"], r.get("source_file"), str(r.get("id")))
        if r["source"] == LLM:
            model_of[key] = r.get("model")
        for s in r.get("sentence_classification") or []:
            if s.get("predicted_label"):
                counts[(r.get("speaker_type"), key)][s["predicted_label"]] += 1

    # student per H-LLM conversation: (raw export, course_member_id)
    student = {}
    files = collections.defaultdict(set)
    for (_, (cond, sf, cid)) in counts:
        if cond == LLM:
            files[sf].add(cid)
    for sf, ids in files.items():
        p = raw_path(sf)
        if p is None:
            raise SystemExit(f"no raw file for {sf}")
        for c in json.loads(p.read_text(encoding="utf-8")):
            if str(c.get("id")) in ids:
                student[(LLM, sf, str(c["id"]))] = (sf, c.get("course_member_id"))
    llm_convs = sorted({k for (_, k) in counts if k[0] == LLM})
    missing = [k for k in llm_convs if student.get(k, (None, None))[1] is None]
    if missing:
        raise SystemExit(f"{len(missing)} H-LLM conversations without course_member_id")
    print(f"{len(llm_convs)} H-LLM conversations from {len(set(student[k] for k in llm_convs))} students")

    rng = np.random.default_rng(args.seed)
    out = []
    for spk, spk_name in SPEAKERS:
        keys = {c: sorted(k for (s, k) in counts if s == spk and k[0] == c) for c in (REF, RP, LLM)}
        labels = sorted({l for (s, _), cnt in counts.items() if s == spk for l in cnt})
        vec = {k: np.array([counts[(spk, k)].get(l, 0) for l in labels], float) for k in set().union(*keys.values())}
        mats = {c: np.stack([vec[k] for k in keys[c]]) for c in keys}

        def clusters_of(conv_keys):
            groups = collections.defaultdict(list)
            for i, k in enumerate(conv_keys):
                groups[student[k]].append(i)
            return [np.array(v) for v in groups.values()]

        def draw_conv(mat):
            return mat[rng.integers(0, len(mat), len(mat))].sum(axis=0)

        def draw_students(mat, clusters):
            pick = rng.integers(0, len(clusters), len(clusters))
            return mat[np.concatenate([clusters[i] for i in pick])].sum(axis=0)

        # proximity gap Delta = JSD(real, H-LLM) - JSD(real, HH roleplay)
        ref, rp, llm = mats[REF], mats[RP], mats[LLM]
        cl = clusters_of(keys[LLM])
        point = js_distance(ref.sum(0), llm.sum(0)) - js_distance(ref.sum(0), rp.sum(0))
        for scheme in ("conversations", "students"):
            ds = []
            for _ in range(args.B):
                r_ = draw_conv(ref)
                l_ = draw_conv(llm) if scheme == "conversations" else draw_students(llm, cl)
                ds.append(js_distance(r_, l_) - js_distance(r_, draw_conv(rp)))
            lo, hi = np.percentile(ds, [2.5, 97.5])
            out.append(dict(quantity=f"Delta ({spk_name})", speaker=spk_name, resampling=scheme,
                            n_conv=len(keys[LLM]), n_students=len(cl), estimate=point, ci_lo=lo, ci_hi=hi))
            print(out[-1])

        # per-model distance to HH real, as in Figure 2
        for mkey, mname in MODELS:
            mk = [k for k in keys[LLM] if model_of.get(k) == mkey]
            mm = np.stack([vec[k] for k in mk])
            mcl = clusters_of(mk)
            point = js_distance(ref.sum(0), mm.sum(0))
            for scheme in ("conversations", "students"):
                ds = []
                for _ in range(args.B):
                    m_ = draw_conv(mm) if scheme == "conversations" else draw_students(mm, mcl)
                    ds.append(js_distance(draw_conv(ref), m_))
                lo, hi = np.percentile(ds, [2.5, 97.5])
                out.append(dict(quantity=f"JSD(HH real, {mname}) ({spk_name})", speaker=spk_name, resampling=scheme,
                                n_conv=len(mk), n_students=len(mcl), estimate=point, ci_lo=lo, ci_hi=hi))
                print(out[-1])

    with open(OUT.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    # tex: one row per quantity, conversation- and student-resampled interval side by side
    by_q = collections.OrderedDict()
    for o in out:
        by_q.setdefault(o["quantity"], {})[o["resampling"]] = o
    lines = [r"\begin{tabular}{lrrrcc}", r"\toprule",
             r"Quantity & Conv. & Students & Estimate & 95\% CI, conversations & 95\% CI, students \\", r"\midrule"]
    for q, d in by_q.items():
        c, s = d["conversations"], d["students"]
        name = (q.replace("Delta (Client)", r"$\Delta_{\mathrm{CL}}$").replace("Delta (Counselor)", r"$\Delta_{\mathrm{CO}}$")
                 .replace("JSD(HH real, ", "JSD to HH real, ").replace(") (", ", ").rstrip(")"))
        sign = "+" if q.startswith("Delta") else ""

        def f(x):
            return f"{0.0 if abs(x) < 5e-4 else x:{sign}.3f}".replace("-", "$-$")
        lines.append(f"{name} & {c['n_conv']} & {c['n_students']} & {f(c['estimate'])} & "
                     f"[{f(c['ci_lo'])}, {f(c['ci_hi'])}] & [{f(s['ci_lo'])}, {f(s['ci_hi'])}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT.with_suffix(".csv"), OUT.with_suffix(".tex"))


if __name__ == "__main__":
    main()
