#!/usr/bin/env python3
"""Generate the LAK hero figure as standalone TikZ.

Three panels: (1) the three-way distance geometry anchored in real counseling's
own noise band, (2) the four behavioural signatures of simulated clients by
conversation decile, (3) course-level convergence of measured distance and
learner-perceived realism.

All series are read from the analysis outputs so the figure regenerates with the
results. Stdlib only.

Writes results/figures/oncoco/tikz/hero_figure.tex
"""
import csv
import math
import json
from tikz_fonts import scale_fonts
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TIKZ = ROOT / "results/figures/oncoco/tikz"

# ---------------------------------------------------------------- input data
lm = json.load(open(ROOT / "analysis/semantic/oncoco/label_text_map.json"))
temporal = list(csv.DictReader(open(ROOT / "results/tables/oncoco_temporal_distribution.csv")))

CONDS = [("HH_real_chat", "cReal"), ("HH_roleplay_chat", "cRP"), ("H_LLM_roleplay_chat", "cLLM")]


def bucket(code):
    """Same bucketing as oncoco_behavioral_signatures.py, plus Rejection."""
    t = lm.get(code, code).lower()
    if "problem" in t:
        return "problem"
    if any(k in t for k in ["recommendation", "resource activation", "implementation"]):
        return "solution"
    if "general request" in t:
        return "request"
    if "rejection" in t:
        return "rejection"
    return None


agg = {}
for r in temporal:
    if r["speaker_type"] != "Client":
        continue
    b = bucket(r["label"])
    if not b:
        continue
    key = (r["condition"], b, int(r["decile"]))
    agg[key] = agg.get(key, 0.0) + float(r["proportion"])


def series(cond, bkt):
    return [agg.get((cond, bkt, d), 0.0) for d in range(10)]


# Noise bands, read unrounded from the CSV rather than from the rounded LaTeX table
# (the rounded band mean would turn the 4.4x ratio reported in the text into 4.3x).
band = {}
for r in csv.DictReader(open(ROOT / "results/tables/oncoco_noise_band.csv")):
    band[(r["comparison"], r["speaker"])] = dict(
        jsd=float(r["jsd_rp"]), mean=float(r["band_mean"]), p95=float(r["band_p95"]),
        nm=float(r["nm_p95"]), v=float(r["cramers_v"]), ratio=float(r["ratio_to_floor"]))

# RQ3 convergence: semester, model, conversations, client JSD, survey n, realism M
tam = []
for line in open(ROOT / "results/tables/oncoco_tam_convergence.tex"):
    parts = [p.strip() for p in line.split("&")]
    if len(parts) == 6 and parts[-1].endswith(r"\\"):
        try:
            nconv = int(parts[2])
            jsd = float(parts[3].split("(")[0])
            nm = float(parts[3].split("(")[1].rstrip(") "))
            n = int(parts[4])
            realism = float(parts[5].split("(")[0])
        except (ValueError, IndexError):
            continue  # header row
        tam.append(dict(sem=parts[0], model=parts[1], nconv=nconv, jsd=jsd, nm=nm,
                        n=n, realism=realism))

# The parser above drops any row it cannot read into the `except: continue` branch, so a
# column-count change in the TAM table would silently shrink panel 3 instead of failing.
assert len(tam) == 3, (
    f"parsed {len(tam)} TAM rows from oncoco_tam_convergence.tex, expected 3 semesters; "
    "the table layout changed and the parser above needs updating")

# Conversations per condition, so the triangle's vertex labels follow the corpus
nconv = {}
for r in csv.DictReader(open(ROOT / "results/tables/oncoco_within_condition_jsd.csv")):
    nconv[r["condition"]] = max(nconv.get(r["condition"], 0), int(r["n_conversations"]))

# Pairwise between-condition distances (single source for every number in panel 1)
dist = {}
for r in csv.DictReader(open(ROOT / "results/tables/oncoco_rq1_distances.csv")):
    role = "Counselor" if r["speaker_type"].startswith("Couns") else "Client"
    dist[(r["comparison"], role)] = float(r["jsd"])

C_RP = "HH_real_chat vs HH_roleplay_chat"
C_LLM = "HH_real_chat vs H_LLM_roleplay_chat"
C_RP_LLM = "HH_roleplay_chat vs H_LLM_roleplay_chat"

# ------------------------------------------------------- panel 1: the triad
# Client-side JSDs among the three conditions, drawn to scale as a Euclidean
# triangle (JSD is a metric, so the three distances embed exactly in the plane).
D_REAL_RP = dist[(C_RP, "Client")]
D_REAL_LLM = dist[(C_LLM, "Client")]
D_RP_LLM = dist[(C_RP_LLM, "Client")]
S = 6.4  # cm per JSD unit

bx = D_REAL_RP * S
_a, _b = D_REAL_LLM * S, D_RP_LLM * S
cx = (_a**2 - _b**2 + bx**2) / (2 * bx)
cy = (_a**2 - cx**2) ** 0.5

CL_LLM = band[("H_LLM_roleplay_chat", "Client")]
R_BAND = CL_LLM["p95"] * S      # split-half P95 disc
FLOOR_MULT = CL_LLM["ratio"]
V_CLIENT = CL_LLM["v"]

CO_BAND = band[("H_LLM_roleplay_chat", "Counselor")]["p95"]
CO_RP = dist[(C_RP, "Counselor")]
CO_LLM = dist[(C_LLM, "Counselor")]

# --------------------------------------------------------------- geometry
W, H = 17.40, 7.40
PA = (0.00, 5.12)
PB = (5.42, 12.42)
PC = (12.72, 17.40)

TINY = r"\fontsize{4.7}{5.4}\selectfont"
MICRO = r"\fontsize{5.2}{6.0}\selectfont"
SMALLX = r"\fontsize{5.8}{6.8}\selectfont"
MED = r"\fontsize{6.4}{7.4}\selectfont"
NOTE = r"\fontsize{5.0}{5.8}\selectfont"   # one-line takeaway under each mini-panel


def panel(x0, x1, num, title, sub):
    return r"""
  \draw[rounded corners=2.5pt, draw=cRule, fill=cPanel, line width=0.5pt]
        ({x0:.2f},0) rectangle ({x1:.2f},{H:.2f});
  \node[circle, fill=cInk, text=white, inner sep=0pt, minimum size=3.4mm,
        font=\bfseries\fontsize{{6}}{{7}}\selectfont] at ({cx:.2f},{cy:.2f}) {{{num}}};
  \node[anchor=west, font=\small\bfseries, text=cInk, inner sep=0pt]
        at ({tx:.2f},{cy:.2f}) {{{title}}};
  \node[anchor=north west, font=\fontsize{{5.6}}{{6.6}}\selectfont, text=cMute, inner sep=0pt,
        align=left, text width={tw:.2f}cm] at ({tx:.2f},{sy:.2f}) {{{sub}}};""".format(
        x0=x0, x1=x1, H=H, num=num, title=title, sub=sub,
        cx=x0 + 0.40, cy=H - 0.42, tx=x0 + 0.63, sy=H - 0.64,
        tw=x1 - x0 - 0.95,
    )


# ------------------------------------------------------------------ helpers
def coords(vals, w, h, ymax):
    return " ".join(
        "({:.3f},{:.3f})".format((d + 0.5) / 10 * w, min(v / ymax, 1.0) * h)
        for d, v in enumerate(vals)
    )


def ytick(v):
    return "0" if v == 0 else ("%.2f" % v).lstrip("0")


def sparkline(bkt, ymax, title, subtitle, note, w=2.78, h=1.30, ticks=False):
    """One mini-panel of panel 2. Origin at the plot's bottom-left corner."""
    o = []
    o.append("  \\draw[cRule, line width=0.4pt] (0,0) -- ({:.2f},0);".format(w))
    for frac in (0.5, 1.0):
        o.append("  \\draw[cRule!55, line width=0.3pt, dash pattern=on 1pt off 1.4pt] "
                 "(0,{0:.3f}) -- ({1:.2f},{0:.3f});".format(frac * h, w))
    for frac in (0.0, 0.5, 1.0):
        o.append("  \\node[anchor=east, font=%s, text=cMute!85, inner sep=0pt] "
                 "at (-0.07,%.3f) {%s};" % (TINY, frac * h, ytick(frac * ymax)))
    llm = coords(series("H_LLM_roleplay_chat", bkt), w, h, ymax)
    x_first = llm.split()[0].strip("()").split(",")[0]
    x_last = llm.split()[-1].strip("()").split(",")[0]
    o.append("  \\fill[cLLM, opacity=0.09] ({},0) -- plot coordinates {{{}}} -- ({},0) -- cycle;"
             .format(x_first, llm, x_last))
    for cond, col in CONDS:
        lw = "1.0pt" if cond == "H_LLM_roleplay_chat" else "0.6pt"
        o.append("  \\draw[{}, line width={}, line join=round] plot coordinates {{{}}};"
                 .format(col, lw, coords(series(cond, bkt), w, h, ymax)))
    o.append("  \\node[anchor=west, font=\\scriptsize\\bfseries, text=cInk, inner sep=0pt] "
             "at (-0.02,{:.2f}) {{{}}};".format(h + 0.56, title))
    o.append("  \\node[anchor=west, font=%s, text=cMute, inner sep=0pt] at (-0.02,%.2f) {%s};"
             % (TINY, h + 0.23, subtitle))
    if ticks:
        for frac, lab in ((0.0, "0"), (0.5, "50"), (1.0, "100")):
            anchor = {0.0: "north west", 0.5: "north", 1.0: "north east"}[frac]
            o.append("  \\node[anchor=%s, font=%s, text=cMute!80, inner sep=0pt] "
                     "at (%.3f,%.2f) {%s};" % (anchor, TINY, frac * w, -0.44, lab))
    return "\n".join(o)


# --------------------------------------------------------------- the figure
parts = [r"""%% Auto-generated by analysis/semantic/oncoco/make_hero_tikz.py -- do not edit by hand.
\documentclass[border=1pt,10pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[scaled=0.92]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{sansmath}
\sansmath
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,patterns}

\definecolor{cReal}{HTML}{2C6E63}
\definecolor{cRP}{HTML}{4A6E9B}
\definecolor{cLLM}{HTML}{C0503C}
\definecolor{cAcc}{HTML}{7A5EA6}
\definecolor{cMeas}{HTML}{3F4A57}
\definecolor{cInk}{HTML}{1F2328}
\definecolor{cMute}{HTML}{70757D}
\definecolor{cRule}{HTML}{CFD4DA}
\definecolor{cPanel}{HTML}{FBFBFC}
\definecolor{cBand}{HTML}{E5E9EE}

\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm]
\path (0,0) rectangle (%.2f,%.2f);""" % (W, H)]

# ------------------------------------------------------------------ panel 1
parts.append(panel(PA[0], PA[1], 1, "Distance between conditions",
                   r"Client side, Jensen--Shannon distance"))

ox, oy = PA[0] + 1.58, 3.07
parts.append(r"""
  \begin{{scope}}[shift={{({ox:.2f},{oy:.2f})}}]
    \fill[cReal!7] (0,0) circle ({rb:.3f});
    \fill[pattern=north east lines, pattern color=cReal!45] (0,0) circle ({rb:.3f});
    \draw[cReal!45, line width=0.45pt, dash pattern=on 1.6pt off 1.4pt] (0,0) circle ({rb:.3f});
    \draw[cRule!95, line width=0.6pt] (0,0) -- ({bx:.3f},0);
    \draw[cRule!95, line width=0.6pt] ({bx:.3f},0) -- ({cx:.3f},{cy:.3f});
    \draw[cLLM!60, line width=1.1pt] (0,0) -- ({cx:.3f},{cy:.3f});
    \node[font={T}, text=cMute, fill=cPanel, inner sep=0.8pt]
         at ({mab:.3f},0) {{{d_rp:.3f}}};
    \node[font={T}, text=cMute, fill=cPanel, inner sep=0.8pt, rotate=84]
         at ({mbcx:.3f},{mbcy:.3f}) {{{d_rp_llm:.3f}}};
    \node[font={M}\bfseries, text=cLLM, fill=cPanel, inner sep=0.8pt, rotate=51]
         at ({macx:.3f},{macy:.3f}) {{{d_llm:.3f}}};
    \fill[cReal] (0,0) circle (2.1pt);
    \fill[cRP] ({bx:.3f},0) circle (2.1pt);
    \fill[cLLM] ({cx:.3f},{cy:.3f}) circle (2.7pt);
    \node[anchor=north, font={D}\bfseries, text=cReal, inner sep=0pt] at (0,{ra:.3f})
         {{HH real}};
    \node[anchor=north west, font={D}\bfseries, text=cRP, inner sep=0pt] at ({bxl:.3f},-0.10)
         {{HH roleplay}};
    \node[anchor=south, font=\scriptsize\bfseries, text=cLLM, inner sep=0pt]
         at ({cx:.3f},{cyl:.3f}) {{H--LLM}};
  \end{{scope}}
  \node[anchor=north west, font={T}, text=cMute, align=left, inner sep=0pt]
       at ({lx:.2f},{ly:.2f}) {{real counseling's own\\split-half band}};
  \draw[cMute!65, line width=0.3pt] ({ax:.2f},{ay:.2f}) -- ({bx2:.2f},{by2:.2f});""".format(
    ox=ox, oy=oy, rb=R_BAND, bx=bx, cx=cx, cy=cy,
    T=TINY, M=MED, D=MED,
    d_rp=D_REAL_RP, d_llm=D_REAL_LLM, d_rp_llm=D_RP_LLM,
    mab=bx / 2, mbcx=(bx + cx) / 2 + 0.07, mbcy=cy / 2,
    macx=cx / 2 - 0.11, macy=cy / 2,
    ra=-R_BAND - 0.03,
    bxl=bx + 0.06, cyl=cy + 0.14,
    lx=PA[0] + 0.24, ly=4.39,
    ax=PA[0] + 0.72, ay=4.01, bx2=ox - R_BAND * 0.87, by2=oy + R_BAND * 0.50,
))

# counselor-side strip
sx0, sx1 = PA[0] + 0.50, PA[0] + 4.62
sy, smax = 0.57, 0.33
sc = (sx1 - sx0) / smax
parts.append(r"""
  \draw[cRule, line width=0.4pt] ({sx0:.2f},{div:.2f}) -- ({sx1:.2f},{div:.2f});
  \node[anchor=west, font={S}\bfseries, text=cInk, inner sep=0pt] at ({sx0:.2f},{lab:.2f})
       {{Counselor side}};
  \node[anchor=north west, font={T}, text=cMute, align=left, inner sep=0pt, text width=3.9cm]
       at ({sx0:.2f},{lab2:.2f})
       {{both distances exceed real counseling's noise band; the conditions separate on the client side}};
  \fill[cReal!7] ({sx0:.2f},{y0:.2f}) rectangle ({bandx:.3f},{y1:.2f});
  \fill[pattern=north east lines, pattern color=cReal!45] ({sx0:.2f},{y0:.2f}) rectangle ({bandx:.3f},{y1:.2f});
  \draw[cReal!45, line width=0.45pt, dash pattern=on 1.6pt off 1.4pt]
       ({bandx:.3f},{y0:.2f}) -- ({bandx:.3f},{y1:.2f});
  \draw[cRule, line width=0.4pt] ({sx0:.2f},{y0:.2f}) rectangle ({sx1:.2f},{y1:.2f});
  \node[anchor=west, font={T}, text=cMute, inner sep=1.4pt] at ({sx0:.2f},{ym:.2f})
       {{noise band}};
  \fill[cRP] ({rpx:.3f},{ym:.2f}) circle (2.0pt);
  \fill[cLLM] ({llmx:.3f},{ym:.2f}) circle (2.4pt);
  \node[anchor=north east, font={T}, text=cRP, inner sep=1.8pt] at ({rpx:.3f},{y0:.2f})
       {{{d_rp:.3f}}};
  \node[anchor=north west, font={M}\bfseries, text=cLLM, inner sep=1.8pt] at ({llmx:.3f},{y0:.2f})
       {{{d_llm:.3f}}};""".format(
    sx0=sx0, sx1=sx1, div=1.75, lab=1.55, lab2=1.43,
    T=TINY, S=SMALLX, M=MICRO,
    y0=sy - 0.17, y1=sy + 0.17, ym=sy,
    bandx=sx0 + CO_BAND * sc, d_rp=CO_RP, d_llm=CO_LLM,
    rpx=sx0 + CO_RP * sc, llmx=sx0 + CO_LLM * sc,
))

# ------------------------------------------------------------------ panel 2
parts.append(panel(PB[0], PB[1], 2, "Client categories over time",
                   r"Share of client spans, by decile of conversation progress"))

# legend
lx = PB[0] + 0.44
for lab, col, lw in (("HH real", "cReal", "0.9pt"),
                     ("HH roleplay", "cRP", "0.9pt"),
                     ("H--LLM roleplay", "cLLM", "1.4pt")):
    parts.append(r"""  \draw[{col}, line width={lw}] ({x:.2f},{y:.2f}) -- ({x2:.2f},{y:.2f});
  \node[anchor=west, font={T}, text=cInk, inner sep=0pt] at ({x3:.2f},{y:.2f}) {{{lab}}};"""
                 .format(col=col, lw=lw, x=lx, x2=lx + 0.30, x3=lx + 0.37, y=6.05, lab=lab, T=TINY))
    lx += 0.37 + 0.062 * len(lab) + 0.34

# Panel-2 captions used to be literal strings, so a corpus change could not reach them.
# Every number in them is derived here from the same artifacts the curves are drawn from.
sig = {}
for r in csv.DictReader(open(ROOT / "results/tables/oncoco_signature_shares.csv")):
    if not r["model"]:
        sig[r["condition"]] = r

rej_span = {}
_tot = {}
for r in csv.DictReader(open(ROOT / "results/tables/oncoco_label_distribution.csv")):
    if r["speaker_type"] != "Client":
        continue
    c = r["condition"]
    _tot[c] = _tot.get(c, 0) + int(r["count"])
    if "rejection" in lm.get(r["label"], r["label"]).lower():
        rej_span[c] = rej_span.get(c, 0) + int(r["count"])

_H = ["HH_real_chat", "HH_roleplay_chat"]
_L = "H_LLM_roleplay_chat"


def _peak(cond, bkt):
    v = series(cond, bkt)
    return max(v), v.index(max(v)) + 1


_p_llm, _p_dec = _peak(_L, "problem")
_p_real, _ = _peak("HH_real_chat", "problem")
_s_llm = max(series(_L, "solution"))
_s_hum = max(max(series(c, "solution")) for c in _H)
_ask = [1 - float(sig[c]["share_with_request"]) for c in _H]
_rej = sorted(rej_span[c] / _tot[c] for c in _H)

MINIS = [
    ("problem", 0.40, "Discloses too early", r"\textit{Problem statement + definition}",
     rf"peaks at {_p_llm:.2f} (HH real {_p_real:.2f})", False),
    ("solution", 0.50, "Drifts into solutions", r"recommendations, resources",
     rf"climbs to {_s_llm:.2f}, humans below {math.ceil(_s_hum * 100) / 100:.2f}", False),
    ("request", 0.08, "Rarely asks back", r"\textit{General request}",
     rf"{1 - float(sig[_L]['share_with_request']):.0%} ask nothing, humans {min(_ask) * 100:.0f}--{max(_ask):.0%}".replace("%", r"\%"), True),
    ("rejection", 0.06, "Rarely disagrees", r"\textit{Rejection}",
     rf"{rej_span[_L] / _tot[_L]:.3f}, humans {_rej[0]:.3f}--{_rej[1]:.3f}", True),
]
gx = [PB[0] + 0.70, PB[0] + 3.92]
gy = [3.70, 1.17]
for i, (bkt, ymax, title, sub, note, ticks) in enumerate(MINIS):
    parts.append("\n  \\begin{{scope}}[shift={{({:.2f},{:.2f})}}]\n{}\n  \\end{{scope}}".format(
        gx[i % 2], gy[i // 2], sparkline(bkt, ymax, title, sub, note, ticks=ticks)))

parts.append(r"""  \node[anchor=north, font=%s, text=cMute, inner sep=0pt]
       at (%.2f,%.2f) {conversation progress (\%%)};""" % (TINY, (PB[0] + PB[1]) / 2, 0.51))

# ------------------------------------------------------------------ panel 3
parts.append(panel(PC[0], PC[1], 3, "Course-level convergence",
                   r"One course line, three semesters"))

bx0, bx1 = PC[0] + 0.52, PC[1] - 0.42
slot = (bx1 - bx0) / 3
bw = slot * 0.44
centers = [bx0 + slot * (i + 0.5) for i in range(3)]


def barchart(y0, height, values, vmin, vmax, color, title, unit, fmt,
             refs=None, reflabel=None, hi=1):
    """Zero-based bar row; title left and unit right on one header line.

    refs adds a short dashed marker per bar (used for the size-matched null).
    """
    o = ["""  \\node[anchor=north west, font=%s\\bfseries, text=cInk, inner sep=0pt]
       at (%.2f,%.2f) {%s};
  \\node[anchor=north east, font=%s, text=cMute, inner sep=0pt]
       at (%.2f,%.2f) {%s};""" % (SMALLX, bx0 - 0.10, y0 + height + 0.60, title,
                                  TINY, bx1 + 0.02, y0 + height + 0.56, unit)]
    o.append("  \\draw[cRule, line width=0.4pt] ({:.2f},{:.2f}) -- ({:.2f},{:.2f});"
             .format(bx0 - 0.10, y0, bx1 + 0.02, y0))
    for i, v in enumerate(values):
        hgt = (v - vmin) / (vmax - vmin) * height
        op = "" if i == hi else "!40"
        o.append("  \\fill[{c}{op}] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y1:.3f});".format(
            c=color, op=op, x0=centers[i] - bw / 2, y0=y0, x1=centers[i] + bw / 2, y1=y0 + hgt))
        o.append("  \\node[anchor=south, font=%s%s, text=%s, inner sep=1.3pt] at (%.3f,%.3f) {%s};"
                 % (MICRO, r"\bfseries" if i == hi else "", color,
                    centers[i], y0 + hgt, fmt % v))
    for i, r in enumerate(refs or []):
        ry = y0 + (r - vmin) / (vmax - vmin) * height
        o.append("  \\draw[cMute, line width=0.5pt] ({:.3f},{:.3f}) -- ({:.3f},{:.3f});".format(
            centers[i] - bw * 0.72, ry, centers[i] + bw * 0.72, ry))
    if reflabel:
        o.append("  \\node[anchor=east, font=%s, text=cMute, inner sep=1.2pt] at (%.2f,%.2f) {%s};"
                 % (TINY, centers[0] - bw * 0.72,
                    y0 + (refs[0] - vmin) / (vmax - vmin) * height, reflabel))
    return "\n".join(o)


parts.append(barchart(4.05, 1.45, [t["jsd"] for t in tam], 0.0, 0.70, "cMeas",
                      r"Measured distance", r"JSD to HH real", "%.3f",
                      refs=[t["nm"] for t in tam], reflabel="noise"))
parts.append(barchart(1.70, 1.45, [t["realism"] for t in tam], 1.0, 5.0, "cAcc",
                      r"Perceived as \emph{less} human",
                      r"realism, 1--5", "%.2f"))

for i, t in enumerate(tam):
    short = t["model"].replace(" 8x7B", "").replace(" 3.3 70B", " 3.3").replace("-120B", "")
    parts.append(r"""  \node[anchor=north, font={M}\bfseries, text=cInk, inner sep=1.3pt]
       at ({x:.3f},{y:.2f}) {{{short}}};
  \node[anchor=north, font={T}, text=cMute, inner sep=1.1pt]
       at ({x:.3f},{y2:.2f}) {{{sem}}};
  \node[anchor=north, font={T}, text=cMute, inner sep=1.1pt]
       at ({x:.3f},{y3:.2f}) {{{nc} conv.}};
  \node[anchor=north, font={T}, text=cMute, inner sep=1.1pt]
       at ({x:.3f},{y4:.2f}) {{$n$\,=\,{n}}};""".format(
        x=centers[i], y=1.62, y2=1.36, y3=1.15, y4=0.94, short=short, nc=t["nconv"], n=t["n"],
        M=MICRO, T=TINY,
        sem=t["sem"].replace("WiSe ", "WS\\,").replace("SoSe ", "SS\\,"),
    ))

parts.append(r"""\end{tikzpicture}
\end{document}""")

TIKZ.mkdir(parents=True, exist_ok=True)
out = TIKZ / "hero_figure.tex"
out.write_text(scale_fonts("\n".join(parts)) + "\n")
print("wrote", out)
