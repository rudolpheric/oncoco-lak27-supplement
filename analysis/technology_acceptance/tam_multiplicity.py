#!/usr/bin/env python3
"""RQ3 multiplicity control and sensitivity checks.

Two reviewer objections are answered here:

  (1) "You test one item out of a 13-item battery and report p=.003."
      -> Mann-Whitney per item for the two most recent semesters of the course line
         (GPT-OSS-120B vs Llama 3.3 70B), with Holm step-down and Bonferroni over all 13
         items, and separately with Holm over the five TAM constructs.

  (2) "Your six-course Spearman rho=.83 (p=.042) rests on courses with 2 and 9 respondents."
      -> the same correlation after dropping the two smallest courses, plus a Pearson
         specification, so the fragility is reported rather than discovered by a reviewer.

Inputs : data/Testungen eb.KIT Gesamtübersicht.xlsx
         results/tables/oncoco_tam_courses.csv   (written by tam_convergence.py)
Outputs: results/tables/oncoco_tam_multiplicity.csv / .tex
         results/tables/oncoco_tam_spearman_sensitivity.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "data/Testungen eb.KIT Gesamtübersicht.xlsx"
COURSES_CSV = ROOT / "results/tables/oncoco_tam_courses.csv"
OUT_CSV = ROOT / "results/tables/oncoco_tam_multiplicity.csv"
OUT_TEX = ROOT / "results/tables/oncoco_tam_multiplicity.tex"
OUT_SENS = ROOT / "results/tables/oncoco_tam_spearman_sensitivity.csv"

A270 = [f"A270_{i:02d}" for i in range(1, 14)]
TAM = {"PE": ["A302_11", "A302_12", "A302_10"], "EE": ["A302_01", "A302_03"], "HM": ["A302_02"],
       "BI": ["A302_05", "A302_07", "A302_06"], "SI": ["A302_14", "A302_15"]}

# Short English glosses for the battery, for the supplement table.
ITEM_TEXT = {
    "A270_01": "Individual messages fit the conversation",
    "A270_02": "Felt like writing with a real person",
    "A270_03": "Language quality of the messages",
    "A270_04": "Individual messages were excellent",
    "A270_05": "Responded understandably to my messages",
    "A270_06": "Engaged with my questions and suggestions",
    "A270_07": "Messages coherent and clearly phrased",
    "A270_08": "The conversation as a whole was coherent",
    "A270_09": "Could run a complete counseling conversation",
    "A270_10": "Satisfied with the interaction",
    "A270_11": "Personality felt likeable",
    "A270_12": "No trouble understanding the messages",
    "A270_13": "Reacted adequately to my follow-up questions",
}

xl = pd.ExcelFile(XLSX)


def has_label_row(d):
    if "CASE" not in d.columns or len(d) == 0:
        return True
    return bool(pd.isna(pd.to_numeric(pd.Series([d.iloc[0]["CASE"]]), errors="coerce")).iloc[0])


def sheet(name):
    d = xl.parse(name)
    d = d.iloc[1:] if has_label_row(d) else d
    return d.apply(pd.to_numeric, errors="coerce")


def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.sign(a[:, None] - b[None, :])))


def holm(pvals):
    """Holm step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# --- (1) multiplicity over the 13-item battery --------------------------------
g = sheet("QS SoSe 2026")      # GPT-OSS-120B
l = sheet("QS WiSe 25 26")     # Llama 3.3 70B

rows = []
for item in A270:
    if item not in g.columns or item not in l.columns:
        continue
    a, b = g[item].dropna(), l[item].dropna()
    u, p = mannwhitneyu(a, b)
    rows.append(dict(item=item, text=ITEM_TEXT.get(item, ""), n_gptoss=len(a), n_llama=len(b),
                     m_gptoss=a.mean(), m_llama=b.mean(), U=u, p=p, delta=cliffs(a, b)))

df = pd.DataFrame(rows)
df["p_holm"] = holm(df["p"])
df["p_bonf"] = np.minimum(df["p"] * len(df), 1.0)
df = df.sort_values("p").reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)

print(f"13-item battery, GPT-OSS-120B (n={len(g['A270_02'].dropna())}) vs "
      f"Llama 3.3 70B (n={len(l['A270_02'].dropna())}), Mann-Whitney + Holm:\n")
print(f"  {'item':9s} {'M gpt':>6s} {'M llama':>8s} {'U':>6s} {'p':>8s} {'p_Holm':>8s} "
      f"{'p_Bonf':>8s} {'delta':>7s}  text")
for _, r in df.iterrows():
    star = " *" if r["p_holm"] < 0.05 else ""
    print(f"  {r['item']:9s} {r['m_gptoss']:6.2f} {r['m_llama']:8.2f} {r['U']:6.0f} "
          f"{r['p']:8.4f} {r['p_holm']:8.4f} {r['p_bonf']:8.4f} {r['delta']:+7.2f}  "
          f"{r['text']}{star}")

surviving = df[df["p_holm"] < 0.05]["item"].tolist()
print(f"\n  surviving Holm at .05 across all {len(df)} items: {surviving or 'none'}")

# --- TAM constructs, Holm within the five constructs --------------------------
print("\nFive acceptance constructs (POMP-scaled), Holm within the construct family:")
crows = []
for k, items in TAM.items():
    items_g = [c for c in items if c in g.columns]
    items_l = [c for c in items if c in l.columns]
    pa = ((5 - g[items_g].mask(g[items_g] > 5)) / 4 * 100).mean(axis=1).dropna()
    pb = ((5 - l[items_l].mask(l[items_l] > 5)) / 4 * 100).mean(axis=1).dropna()
    u, p = mannwhitneyu(pa, pb)
    crows.append(dict(construct=k, pomp_gptoss=pa.mean(), pomp_llama=pb.mean(),
                      U=u, p=p, delta=cliffs(pa, pb)))
cdf = pd.DataFrame(crows)
cdf["p_holm"] = holm(cdf["p"])
for _, r in cdf.iterrows():
    print(f"  {r['construct']:3s} POMP {r['pomp_gptoss']:5.1f} vs {r['pomp_llama']:5.1f}  "
          f"p={r['p']:.3f}  p_Holm={r['p_holm']:.3f}  delta={r['delta']:+.2f}")
print(f"  min raw p across constructs: {cdf['p'].min():.3f}; "
      f"max |delta|: {cdf['delta'].abs().max():.2f}")

# --- (2) six-course association sensitivity -----------------------------------
print("\nSix-course association between client-side JSD and realism-item disagreement:")
if not COURSES_CSV.exists():
    print(f"  {COURSES_CSV} missing -- run tam_convergence.py first.")
else:
    c = pd.read_csv(COURSES_CSV)
    print(c.to_string(index=False))
    sens = []
    for label, sub in [
        ("all six courses", c),
        ("drop the two smallest (n=2, n=9)", c.nlargest(len(c) - 2, "n_resp")),
        ("drop only the smallest (n=2)", c.nlargest(len(c) - 1, "n_resp")),
        ("courses with n>=20", c[c["n_resp"] >= 20]),
    ]:
        if len(sub) < 3:
            continue
        rho, prho = spearmanr(sub["jsd_client"], sub["realism_mean"])
        r, pr = pearsonr(sub["jsd_client"], sub["realism_mean"])
        sens.append(dict(subset=label, k_courses=len(sub), spearman_rho=rho, spearman_p=prho,
                         pearson_r=r, pearson_p=pr))
        print(f"  {label:34s} k={len(sub)}  rho={rho:+.3f} p={prho:.3f}   "
              f"r={r:+.3f} p={pr:.3f}")
    pd.DataFrame(sens).to_csv(OUT_SENS, index=False)
    print(f"\nWrote {OUT_SENS}")

# --- supplement table ---------------------------------------------------------
lines = [
    r"\begin{tabular}{llrrrrrr}",
    r"\toprule",
    r"Item & Content & $M_{\text{GPT-OSS}}$ & $M_{\text{Llama}}$ & $U$ & $p$ & $p_{\text{Holm}}$ & $\delta$ \\",
    r"\midrule",
]
for _, r in df.iterrows():
    txt = r["text"].replace("&", r"\&")
    item = r["item"].replace("_", r"\_")
    lines.append(f"{item} & {txt} & {r['m_gptoss']:.2f} & "
                 f"{r['m_llama']:.2f} & {r['U']:.0f} & {r['p']:.3f} & {r['p_holm']:.3f} & "
                 f"{r['delta']:+.2f} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {OUT_TEX}")
