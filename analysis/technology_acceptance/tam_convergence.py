#!/usr/bin/env python3
"""RQ3: Convergence of learner-perceived realism (shortened TAM) with distributional proximity.

Within course line A, three consecutive semesters used the
same shortened TAM questionnaire (incl. the A270 'simulated client' realism battery)
while only the client model changed (Mixtral 8x7B -> Llama 3.3 70B -> GPT-OSS-120B).
This script computes per-course client/counselor JSD to HH real chat on the merged
corpus, size-matched nulls, survey statistics (item A270_02 'felt like a real person'),
Mann-Whitney tests with Cliff's delta for the model change, and writes the paper table.

Surveys carry no participant identifiers (SERIAL/REF empty), so linkage is course-level.
"""
import json, os, collections
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
CLS = ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"
XLSX = ROOT / "data/surveys/course_surveys.xlsx"
OUT_TEX = ROOT / "results/tables/oncoco_tam_convergence.tex"

rows = json.load(open(CLS))

# --- course lookup keyed by (source_file, id); ids collide across raw files ---
course_of = {}
for sf in {r["source_file"] for r in rows if r.get("modality") == "chat" and r.get("source_file")}:
    raw = sf.replace("/chats/", "/chats_filtered/") if "human_llm" in sf else sf
    path = ROOT / (raw if (ROOT / raw).exists() else sf)
    for x in json.load(open(path)):
        cm = x.get("course_member") or {}
        co = (cm.get("course") or {}) if isinstance(cm, dict) else {}
        course_of[(sf, str(x["id"]))] = co.get("name")


def js(p, q):
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / (b[a > 0] + 1e-12))))
    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def conv_counts(cond, speaker, course=None):
    out = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("source") != cond or r.get("speaker_type") != speaker:
            continue
        if course is not None and course_of.get((r["source_file"], str(r["id"]))) != course:
            continue
        labs = [s["predicted_label"] for s in r.get("sentence_classification") or [] if s.get("predicted_label")]
        if labs:
            out[(r["source_file"], str(r["id"]))].update(labs)
    return out


REF = {sp: conv_counts("HH_real_chat", sp) for sp in ("Client", "Counsellor")}


def matrix(convs, index):
    m = np.zeros((len(convs), len(index)))
    for i, c in enumerate(convs.values()):
        for lab, n in c.items():
            m[i, index[lab]] += n
    return m


def course_jsd(course, speaker, rng):
    comp = conv_counts("H_LLM_roleplay_chat", speaker, course)
    ref = REF[speaker]
    labels = sorted({l for c in list(comp.values()) + list(ref.values()) for l in c})
    idx = {l: i for i, l in enumerate(labels)}
    A, B = matrix(comp, idx), matrix(ref, idx)
    d = js(A.sum(0), B.sum(0))
    nm = np.array([js(B[rng.integers(0, B.shape[0], A.shape[0])].sum(0),
                      B[rng.integers(0, B.shape[0], B.shape[0])].sum(0)) for _ in range(1000)])
    return d, len(comp), float(np.percentile(nm, 95))


# --- survey ------------------------------------------------------------------
xl = pd.ExcelFile(XLSX)
A270 = [f"A270_{i:02d}" for i in range(1, 14)]
TAM = {"PE": ["A302_11", "A302_12", "A302_10"], "EE": ["A302_01", "A302_03"], "HM": ["A302_02"],
       "BI": ["A302_05", "A302_07", "A302_06"], "SI": ["A302_14", "A302_15"]}


def has_label_row(d):
    """A German label row is present iff row 0's CASE is not numeric.

    The flag used to be hand-declared per sheet, and one declaration was wrong:
    'survey_C09' has no label row, so d.iloc[1:] silently dropped a real
    respondent (CASE 292, A270_02=4) and reported n=25 instead of 26.
    """
    if "CASE" not in d.columns or len(d) == 0:
        return True
    return bool(pd.isna(pd.to_numeric(pd.Series([d.iloc[0]["CASE"]]), errors="coerce")).iloc[0])


def sheet(name, label_row=None):
    d = xl.parse(name)
    detected = has_label_row(d)
    if label_row is not None and label_row != detected:
        raise AssertionError(
            f"sheet {name!r}: declared label_row={label_row} but detected {detected} "
            f"(row 0 CASE={d.iloc[0]['CASE']!r}). Fix the declaration, do not silence this."
        )
    d = d.iloc[1:] if detected else d
    return d.apply(pd.to_numeric, errors="coerce")


def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.sign(a[:, None] - b[None, :])))


COURSES = [  # (semester, model, corpus course name, sheet, label_row)
    ("WiSe 24/25", "Mixtral 8x7B", "C03", "survey_C03", True),
    ("WiSe 25/26", "Llama 3.3 70B", "C09", "survey_C09", False),
    ("SoSe 26", "GPT-OSS-120B", "C13", "survey_C13", False),
]

rng = np.random.default_rng(42)
tab = []
for sem, model, cname, sh, lr in COURSES:
    d_cl, n_conv, nm95 = course_jsd(cname, "Client", rng)
    d_co, _, _ = course_jsd(cname, "Counsellor", rng)
    s = sheet(sh, lr)
    real = s["A270_02"].dropna()
    tab.append(dict(sem=sem, model=model, n_conv=n_conv, jsd_cl=d_cl, nm95=nm95, jsd_co=d_co,
                    n_resp=len(real), real_m=real.mean(), real_sd=real.std(), sheet_df=s))
    print(f"{sem:11s} {model:14s} conv={n_conv:3d}  JSD_cl={d_cl:.3f} (nm-P95 {nm95:.3f})  "
          f"JSD_co={d_co:.3f}  n_resp={len(real):2d}  A270_02 M={real.mean():.2f} SD={real.std():.2f}")

# --- tests: GPT-OSS (SoSe 26) vs Llama (WiSe 25/26), same course line --------
g = tab[2]["sheet_df"]; l = tab[1]["sheet_df"]
a, b = g["A270_02"].dropna(), l["A270_02"].dropna()
u, p = mannwhitneyu(a, b)
print(f"\nRealismus-Item GPT-OSS vs Llama: U={u:.0f} p={p:.4f} Cliff's d={cliffs(a, b):+.2f}")
ab = g[A270].mean(axis=1).dropna(); bb = l[[c for c in A270 if c in l.columns]].mean(axis=1).dropna()
u2, p2 = mannwhitneyu(ab, bb)
print(f"A270-Gesamtbatterie:            U={u2:.0f} p={p2:.4f} d={cliffs(ab, bb):+.2f}")
for k, items in TAM.items():
    pa = ((5 - g[items].mask(g[items] > 5)) / 4 * 100).mean(axis=1).dropna()
    pb = ((5 - l[items].mask(l[items] > 5)) / 4 * 100).mean(axis=1).dropna()
    uu, pp = mannwhitneyu(pa, pb)
    print(f"  TAM {k}: POMP {pa.mean():5.1f} vs {pb.mean():5.1f}  p={pp:.3f}  d={cliffs(pa, pb):+.2f}")

# --- pooled item pattern across all shortened-TAM waves ----------------------
# The hand-maintained GESAMT sheet is NOT the union of the waves: it has 213 rows but only
# 190 distinct CASE ids, and it is missing three entire waves (both most recent semesters
# and one external cohort). CASE is also only unique *within* a wave, so the pool must be
# keyed on (sheet, CASE). We build the pool programmatically and print the reconciliation.
# A wave qualifies if it carries the A270 battery AND a CASE id. 'survey_pilot' carries the
# battery but no CASE column: it is the student-assistant test wave, not a course cohort,
# and is excluded here deliberately (it is absent from the hand-maintained sheet as well).
WAVE_SHEETS = []
for s in xl.sheet_names:
    if s.startswith("GESAMT"):
        continue
    cols = xl.parse(s, nrows=0).columns
    if "A270_02" in cols and "CASE" in cols:
        WAVE_SHEETS.append(s)

parts = []
for s in WAVE_SHEETS:
    d = sheet(s)
    d = d.assign(_wave=s)
    parts.append(d[d["A270_02"].notna()])
pool = pd.concat(parts, ignore_index=True)
dupes = pool.duplicated(subset=["_wave", "CASE"]).sum()
assert dupes == 0, f"{dupes} duplicate (wave, CASE) keys in the pooled frame"

print(f"\nGepoolt über {len(WAVE_SHEETS)} Wellen (n={len(pool)}): "
      f"A270_02 M={pool['A270_02'].mean():.2f} (Menschenähnlichkeit) vs "
      f"A270_12 M={pool['A270_12'].mean():.2f} (Verständlichkeit)")
means = {c: pool[c].mean() for c in A270 if c in pool.columns}
print("  schwächstes Item der Batterie:", max(means, key=means.get), round(max(means.values()), 2))

print("  Wellen im Pool: " + ", ".join(
    f"{w}={int((pool['_wave'] == w).sum())}" for w in WAVE_SHEETS))
ges = sheet("GESAMT")
print(f"  [Abgleich] handgepflegtes GESAMT-Sheet: n={ges['A270_02'].notna().sum()} "
      f"(A270_02 M={ges['A270_02'].mean():.2f}, A270_12 M={ges['A270_12'].mean():.2f}); "
      f"es ist NICHT die Vereinigung der Wellen (213 Zeilen, nur 190 distinkte CASE-Ids, "
      f"jüngste Wellen fehlen) und wird deshalb nicht mehr für die gepoolten Kennwerte benutzt.")

# --- secondary: cross-course association (unambiguous survey mappings only) --
EXTRA = [("C05", "survey_C05", True),
         ("C04", "survey_C04", True),
         ("C10", "survey_C10", True)]
xs, ys, ns, labels, sheets_used = [], [], [], [], []
for sem, model, cname, sh, lr in COURSES:
    t = [t for t in tab if t["sem"] == sem][0]
    xs.append(t["jsd_cl"]); ys.append(t["real_m"]); ns.append(t["n_resp"])
    labels.append(f"{cname} ({model})"); sheets_used.append(sh)
for cname, sh, lr in EXTRA:
    d_cl, n_conv, _ = course_jsd(cname, "Client", rng)
    s = sheet(sh, lr); real = s["A270_02"].dropna()
    xs.append(d_cl); ys.append(real.mean()); ns.append(len(real))
    labels.append(cname); sheets_used.append(sh)
rho, prho = spearmanr(xs, ys)
print(f"\nKursweise Assoziation JSD x Realismus-Disagreement: rho={rho:+.3f} p={prho:.3f} "
      f"(n={len(xs)} Kurse; Befragten-n: {ns})")

# Persist the six-course frame so the multiplicity/sensitivity script does not have to
# recompute the size-matched nulls (analysis/technology_acceptance/tam_multiplicity.py).
OUT_CSV = ROOT / "results/tables/oncoco_tam_courses.csv"
pd.DataFrame({"course": labels, "sheet": sheets_used, "jsd_client": xs,
              "realism_mean": ys, "n_resp": ns}).to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV}")

# --- paper table --------------------------------------------------------------
from scipy import stats as _st
lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
         "Semester & Client model & Conv. & Client JSD (nm-P95) & Survey $n$ & Realism item M (SD) [95\\% CI] \\\\",
         "\\midrule"]
for t in tab:
    # t-based 95% interval of the item mean, so the reader sees the precision behind n=9, 26 and 23
    half = _st.t.ppf(0.975, t['n_resp'] - 1) * t['real_sd'] / (t['n_resp'] ** 0.5)
    lines.append(f"{t['sem']} & {t['model']} & {t['n_conv']} & {t['jsd_cl']:.3f} ({t['nm95']:.3f}) & "
                 f"{t['n_resp']} & {t['real_m']:.2f} ({t['real_sd']:.2f}) [{t['real_m']-half:.2f}, {t['real_m']+half:.2f}] \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
OUT_TEX.write_text("\n".join(lines) + "\n")
print("\nWrote", OUT_TEX)
