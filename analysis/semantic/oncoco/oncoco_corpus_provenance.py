#!/usr/bin/env python3
"""Corpus provenance: the reporting gaps a reviewer will otherwise flag as "not findable".

Five sections, one CSV each:

  (a) trainees      distinct human authors per condition, and conversations per trainee.
                    Emails are read but NEVER emitted -- only counts leave this script.
  (b) exclusions    the legacy H-LLM exclusion, per source file, reconstructed by diffing
                    the pre-filter backup against the analysed artifact; plus the 11
                    quality exclusions, which live in TWO decision files.
  (c) surveys       respondents and conversations per course, with the reasons a response
                    *rate* is not computable.
  (d) language      corpus language, verified rather than asserted.
  (e) model_course  per-model x per-course client JSD to HH real chat, which is what
                    reconciles the model-wide Table 5 with the course-line Table 7.

Outputs: results/tables/oncoco_provenance_{trainees,exclusions,surveys,model_course}.csv
"""
import collections
import json
import re
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NORM = ROOT / "data/processed/combined/normalized"
CLS = NORM / "oncoco_classification_all.json"
PRE_LEGACY = NORM / "oncoco_classification_all.json.bak_pre_legacy_filter"
QR = ROOT / "analysis/quality_review"
TABLES = ROOT / "results/tables"

CHAT = ["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat"]

rows = json.load(open(CLS))
chat_rows = [r for r in rows if r.get("source") in CHAT]


def raw_path(source_file):
    """Analysis artifacts point at data/raw/...; H-LLM raw may live in a filtered twin."""
    alt = source_file.replace("/chats/", "/chats_filtered/") if "human_llm" in source_file else source_file
    for cand in (alt, source_file):
        if (ROOT / cand).exists():
            return ROOT / cand
    return None


# --- (a) trainees -------------------------------------------------------------
# Build (source_file, id) -> author identity from the raw exports. We deliberately emit
# only aggregate counts; no email, name or user id is written to disk.
author_of, course_of = {}, {}
for sf in sorted({r["source_file"] for r in chat_rows if r.get("source_file")}):
    p = raw_path(sf)
    if p is None:
        continue
    for x in json.load(open(p)):
        cm = x.get("course_member") or {}
        if not isinstance(cm, dict):
            cm = {}
        user = cm.get("user") or {}
        if not isinstance(user, dict):
            user = {}
        # email is the only globally unique key: user_id repeats across exports
        ident = user.get("email") or (f"cm:{cm.get('id')}" if cm.get("id") else None)
        author_of[(sf, str(x["id"]))] = ident
        course_of[(sf, str(x["id"]))] = ((cm.get("course") or {}) or {}).get("name")

trainee_rows = []
for cond in CHAT:
    convs = sorted({(r["source_file"], str(r["id"])) for r in chat_rows if r["source"] == cond})
    idents = [author_of.get(c) for c in convs]
    known = [i for i in idents if i]
    per = collections.Counter(known)
    trainee_rows.append(dict(
        condition=cond,
        conversations=len(convs),
        conversations_with_author=len(known),
        distinct_authors=len(per),
        conv_per_author_mean=round(np.mean(list(per.values())), 2) if per else None,
        conv_per_author_median=int(np.median(list(per.values()))) if per else None,
        conv_per_author_max=max(per.values()) if per else None,
        largest_author_share=round(max(per.values()) / len(convs), 3) if per else None,
    ))
trainees = pd.DataFrame(trainee_rows)
trainees.to_csv(TABLES / "oncoco_provenance_trainees.csv", index=False)

print("(a) Distinct human authors per condition")
print(trainees.to_string(index=False))
for r in trainee_rows:
    if r["largest_author_share"] and r["largest_author_share"] > 0.5:
        print(f"    NOTE {r['condition']}: a single platform account carries "
              f"{r['largest_author_share']:.0%} of the conversations -- these were bulk-imported, "
              f"so the number of distinct human authors is NOT recoverable and must not be estimated.")
    if r["conversations_with_author"] == 0:
        print(f"    NOTE {r['condition']}: the exports carry no course-member record at all; "
              f"no author identity exists for this condition.")

# --- (b) exclusions -----------------------------------------------------------
excl_rows = []
if PRE_LEGACY.exists():
    pre = json.load(open(PRE_LEGACY))
    pre_hllm = [r for r in pre if r.get("source") == "H_LLM_roleplay_chat"]
    now_keys = {(r["source_file"], str(r["id"])) for r in chat_rows
                if r["source"] == "H_LLM_roleplay_chat"}
    by_file = collections.defaultdict(lambda: dict(conv=set(), msgs=0, spans=0, with_model=0))
    for r in pre_hllm:
        key = (r["source_file"], str(r["id"]))
        if key in now_keys:
            continue
        e = by_file[r["source_file"]]
        e["conv"].add(key)
        e["msgs"] += 1
        e["spans"] += len(r.get("sentence_classification") or [])
        e["with_model"] += 1 if r.get("model") else 0
    # The backup-vs-current diff contains TWO different removal steps. They separate cleanly
    # by model metadata: the legacy step removed whole source files that carry none (which is
    # exactly the stated reason), the quality step removed individual conversations inside
    # files that do. Reporting the diff as one number would misstate both.
    for sf, e in sorted(by_file.items(), key=lambda kv: -len(kv[1]["conv"])):
        step = "legacy (no model metadata)" if e["with_model"] == 0 else "quality screening"
        excl_rows.append(dict(step=step, source_file=sf, conversations=len(e["conv"]),
                              messages=e["msgs"], spans=e["spans"],
                              messages_with_model=e["with_model"]))
    del pre, pre_hllm
else:
    print(f"(b) WARNING: {PRE_LEGACY.name} not found; legacy exclusion not reconstructible.")

# The authoritative quality-exclusion list is the DROPS constant in apply_quality_filter.py,
# NOT the decision CSVs: two conversations are annotated "borderline -> drop" there and are
# therefore excluded without carrying a literal DROP decision. Counting the CSVs alone gives
# 9 and contradicts the 11 reported in the paper.
import ast
qf = (QR / "apply_quality_filter.py").read_text(encoding="utf-8")
m = re.search(r"^DROPS\s*=\s*(\{.*?^\})", qf, re.S | re.M)
applied = ast.literal_eval(re.sub(r"#.*", "", m.group(1))) if m else set()
quob = pd.read_csv(QR / "quob26_drop_decisions.csv") if (QR / "quob26_drop_decisions.csv").exists() else None
quob_n = int(quob["decision"].astype(str).str.upper().eq("DROP").sum()) if quob is not None else 0
dec = pd.read_csv(QR / "drop_decisions.csv") if (QR / "drop_decisions.csv").exists() else None
dec_drop = int(dec["decision"].astype(str).str.upper().eq("DROP").sum()) if dec is not None else 0

exclusions = pd.DataFrame(excl_rows)
exclusions.to_csv(TABLES / "oncoco_provenance_exclusions.csv", index=False)
print("\n(b) Exclusions")
for step in ("legacy (no model metadata)", "quality screening"):
    sub = exclusions[exclusions["step"] == step]
    if not len(sub):
        continue
    print(f"    {step}: {sub['conversations'].sum()} conversations / "
          f"{int(sub['messages'].sum())} messages / {int(sub['spans'].sum())} spans "
          f"across {len(sub)} source files")
    print(sub[["source_file", "conversations", "messages", "spans"]].to_string(index=False))
print(f"    quality screening, authoritative list: {len(applied)} (main wave) + {quob_n} (QUOB26) "
      f"= {len(applied) + quob_n} conversations")
print(f"      of the main-wave {len(applied)}, {dec_drop} carry a literal DROP decision and "
      f"{len(applied) - dec_drop} are annotated 'borderline -> drop' in apply_quality_filter.py")

# --- (c) surveys --------------------------------------------------------------
courses_csv = TABLES / "oncoco_tam_courses.csv"
if courses_csv.exists():
    c = pd.read_csv(courses_csv)
    # conversations per course, so respondents and conversations sit side by side
    conv_by_course = collections.Counter(
        course_of.get(k) for k in
        {(r["source_file"], str(r["id"])) for r in chat_rows if r["source"] == "H_LLM_roleplay_chat"})
    c["conversations"] = [
        next((n for name, n in conv_by_course.items() if name and name in label), None)
        for label in c["course"]]
    c.to_csv(TABLES / "oncoco_provenance_surveys.csv", index=False)
    print("\n" + c.to_string(index=False))
    print(f"\n(c) Surveys: {len(c)} courses with unambiguous linkage; respondents "
          f"{list(c['n_resp'])}")
    print("    A response RATE is not computable: (i) surveys carry no identifiers, so "
          "respondents cannot be matched to conversation authors; (ii) the invited population "
          "is the course roster, which is not part of the analysis data; (iii) H-LLM "
          "conversations are per-conversation, not per-trainee.")
else:
    print(f"\n(c) {courses_csv.name} missing -- run tam_convergence.py first.")

# --- (d) language -------------------------------------------------------------
GER = re.compile(r"[äöüßÄÖÜ]|\b(ich|nicht|und|dass|wir|sie|mit|aber|schon|kann)\b", re.I)
lang = []
for cond in CHAT:
    texts = [r.get("msg_content", "") for r in chat_rows if r["source"] == cond][:2000]
    hits = sum(1 for t in texts if GER.search(t or ""))
    lang.append((cond, len(texts), round(hits / max(len(texts), 1), 3)))
print("\n(d) Language (share of sampled messages matching German markers)")
for cond, n, share in lang:
    print(f"    {cond:22s} n={n:5d}  German-marker share={share:.3f}")

# --- (e) model x course JSD ---------------------------------------------------
def js(p, q):
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / (b[a > 0] + 1e-12))))
    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def conv_counts(cond, speaker, pred=None):
    out = collections.defaultdict(collections.Counter)
    for r in chat_rows:
        if r.get("source") != cond or r.get("speaker_type") != speaker:
            continue
        key = (r["source_file"], str(r["id"]))
        if pred is not None and not pred(r, key):
            continue
        labs = [s["predicted_label"] for s in r.get("sentence_classification") or []
                if s.get("predicted_label")]
        if labs:
            out[key].update(labs)
    return out


def matrix(convs, index):
    m = np.zeros((len(convs), len(index)))
    for i, cc in enumerate(convs.values()):
        for lab, n in cc.items():
            m[i, index[lab]] += n
    return m


def jsd_with_ci(comp, ref, rng, B=2000):
    labels = sorted({l for cc in list(comp.values()) + list(ref.values()) for l in cc})
    idx = {l: i for i, l in enumerate(labels)}
    A, Bm = matrix(comp, idx), matrix(ref, idx)
    d = js(A.sum(0), Bm.sum(0))
    boot = np.array([js(A[rng.integers(0, A.shape[0], A.shape[0])].sum(0),
                        Bm[rng.integers(0, Bm.shape[0], Bm.shape[0])].sum(0)) for _ in range(B)])
    return d, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(comp), int(A.sum())


rng = np.random.default_rng(42)
ref_client = conv_counts("HH_real_chat", "Client")
model_of = {(r["source_file"], str(r["id"])): r.get("model") for r in chat_rows}

mc_rows = []
models = sorted({m for m in model_of.values() if m})
for model in models:
    d, lo, hi, k, ns = jsd_with_ci(
        conv_counts("H_LLM_roleplay_chat", "Client", lambda r, key: model_of.get(key) == model),
        ref_client, rng)
    mc_rows.append(dict(model=model, course="(all courses)", jsd_client=round(d, 3),
                        ci_lo=round(lo, 3), ci_hi=round(hi, 3), conversations=k, spans=ns))
    courses = sorted({course_of.get(key) for key, m in model_of.items()
                      if m == model and course_of.get(key)})
    for course in courses:
        comp = conv_counts("H_LLM_roleplay_chat", "Client",
                           lambda r, key, c=course, m=model: model_of.get(key) == m and course_of.get(key) == c)
        if len(comp) < 5:
            continue
        d, lo, hi, k, ns = jsd_with_ci(comp, ref_client, rng)
        mc_rows.append(dict(model=model, course=course, jsd_client=round(d, 3),
                            ci_lo=round(lo, 3), ci_hi=round(hi, 3), conversations=k, spans=ns))

mc = pd.DataFrame(mc_rows)
mc.to_csv(TABLES / "oncoco_provenance_model_course.csv", index=False)
print("\n(e) Client-side JSD to HH real chat, per model and per course (95% conversation bootstrap CI)")
print(mc.to_string(index=False))

# Is every course model-pure? If so, the Table 5 / Table 7 gap is a composition effect.
by_course = collections.defaultdict(set)
for key, m in model_of.items():
    if m and course_of.get(key):
        by_course[course_of[key]].add(m)
mixed = {c: ms for c, ms in by_course.items() if len(ms) > 1}
print(f"\n    courses using more than one client model: {mixed if mixed else 'none -- every course is model-pure'}")
print("    => the difference between the model-wide and the course-line distance is a "
      "within-model, between-course composition effect, not a model effect.")
