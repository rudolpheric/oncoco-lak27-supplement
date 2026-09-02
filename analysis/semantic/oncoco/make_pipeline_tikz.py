#!/usr/bin/env python3
"""Generate pipeline_figure.tex from the analysis outputs.

The pipeline figure was hand-written, and every corpus update left stale numbers
behind in it (the 2026-08-13 role repair changed the HH-real span count, both
client-side distances, the noise-floor ratio, and all six panel-5 values, and every
one of them had to be found and patched by hand). This script renders the identical
layout from the same artifacts the paper reads, so the figure follows the results.

Data sources
  panel 1  results/tables/oncoco_label_distribution.csv      spans per condition
           results/tables/oncoco_within_condition_jsd.csv    conversations per condition
           results/tables/oncoco_persona_cases.csv           number of case studies
           results/tables/oncoco_hllm_model_realness_both_baselines.csv  model chips
  panel 4  results/tables/oncoco_noise_band.csv              distances, band, null, ratio
           (the UNROUNDED csv, not the rounded _chat.tex -- same convention as the hero)
  panel 5  results/tables/oncoco_tam_convergence.tex         semester / JSD / realism rows

Panels 2 and 3 are illustrations of fixed method choices (segmentation sketch,
classifier, the 66-category inventory) and stay static in the template.

Usage:
    python analysis/semantic/oncoco/make_pipeline_tikz.py            # writes + compiles
    python analysis/semantic/oncoco/make_pipeline_tikz.py --no-compile
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures" / "oncoco" / "tikz" / "pipeline_figure.tex"

CHAT = [("HH_real_chat", "cReal", "HH real"),
        ("HH_roleplay_chat", "cRP", "HH roleplay"),
        ("H_LLM_roleplay_chat", "cLLM", "H--LLM roleplay")]
MODEL_SHORT = {"GPT_OSS_120B": "GPT-OSS", "LLama3_3_70B": "Llama 3.3", "Mixtral": "Mixtral",
               "GPT-OSS-120B": "GPT-OSS", "Llama 3.3 70B": "Llama 3.3", "Mixtral 8x7B": "Mixtral"}

BAR_MAX = 2.400          # panel-1 bar length of the largest corpus, in cm
RULER_X0, RULER_SCALE = 7.60, 12.0   # panel-4 distance ruler: origin and cm per JSD unit
TAM_JSD_MAX, TAM_BAR_H = 0.70, 1.20  # panel-5 bar scaling


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def load_panel1():
    spans = {}
    for r in csv.DictReader(open(T / "oncoco_label_distribution.csv")):
        key = (r["condition"], r["speaker_type"])
        spans.setdefault(key, int(r["total"]))
    per_cond = {c: sum(v for (cc, _), v in spans.items() if cc == c) for c, _, _ in CHAT}

    convs = {}
    for r in csv.DictReader(open(T / "oncoco_within_condition_jsd.csv")):
        c = r["condition"]
        convs[c] = max(convs.get(c, 0), int(r["n_conversations"]))

    n_cases = sum(1 for _ in csv.DictReader(open(T / "oncoco_persona_cases.csv")))

    models = sorted(csv.DictReader(open(T / "oncoco_hllm_model_realness_both_baselines.csv")),
                    key=lambda r: r["first_seen_month"], reverse=True)
    chips = []
    for m in models:
        s = MODEL_SHORT.get(m["model"], m["model"])
        if s not in chips:
            chips.append(s)
    return per_cond, convs, n_cases, chips


def load_panel4():
    rows = {}
    for r in csv.DictReader(open(T / "oncoco_noise_band.csv")):
        if r["speaker"] == "Client" and r["reference"] == "HH_real_chat":
            rows[r["comparison"]] = r
    rp, llm = rows["HH_roleplay_chat"], rows["H_LLM_roleplay_chat"]
    assert float(llm["jsd_rp"]) * RULER_SCALE < 5.70, "H-LLM distance exceeds the ruler"
    return rp, llm


def load_panel5():
    """Rows of the generated TAM convergence table: (semester, model, jsd, realism)."""
    rows = []
    for line in (T / "oncoco_tam_convergence.tex").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.rstrip("\\").split("&")]
        if len(cells) == 6 and re.match(r"(WiSe|SoSe)", cells[0]):
            jsd = float(re.match(r"([\d.]+)", cells[3]).group(1))
            realism = float(re.match(r"([\d.]+)", cells[5]).group(1))
            sem = cells[0].replace("WiSe ", "WS ").replace("SoSe ", "SS ")
            rows.append((sem, MODEL_SHORT.get(cells[1], cells[1]), jsd, realism))
    assert rows, "no semester rows parsed from oncoco_tam_convergence.tex"
    assert all(j <= TAM_JSD_MAX for _, _, j, _ in rows), "panel-5 JSD exceeds the bar scale"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    per_cond, convs, n_cases, chips = load_panel1()
    rp, llm = load_panel4()
    tam = load_panel5()

    max_spans = max(per_cond.values())
    corpus_rows = ",\n".join(
        f"  {y:.2f}/{col}/{name}/{per_cond[c] / max_spans * BAR_MAX:.3f}"
        f"/{convs[c]} conv.\\ $\\cdot$ {thousands(per_cond[c])} spans"
        for (c, col, name), y in zip(CHAT, (3.86, 3.06, 2.26)))
    chip_rows = ", ".join(f"{i}/{m}" for i, m in enumerate(chips))
    total_line = (f"{sum(convs[c] for c, _, _ in CHAT)} conversations, "
                  f"{thousands(sum(per_cond.values()))} spans")

    band_x = RULER_X0 + float(llm["band_p95"]) * RULER_SCALE
    null_x = RULER_X0 + float(llm["nm_p95"]) * RULER_SCALE
    rp_x = RULER_X0 + float(rp["jsd_rp"]) * RULER_SCALE
    llm_x = RULER_X0 + float(llm["jsd_rp"]) * RULER_SCALE
    ratio = float(llm["ratio_to_floor"])
    # the two null constructions are drawn with the real group sizes
    n_comp, n_ref = int(llm["n_comp_conv"]), int(llm["n_ref_conv"])

    tam_rows = ", ".join(f"{i}/{j:.3f}/{r:.2f}/{{{m}}}" for i, (_, m, j, r) in enumerate(tam))
    tam_sems = "\n".join(
        f"\\node[anchor=north, font=\\TINY, text=cMute, inner sep=1.2pt] "
        f"at ({14.93 + i * 0.90:.2f},1.68) {{{s}}};" for i, (s, _, _, _) in enumerate(tam))

    tex = (HEADER
           + PANEL1.format(rows=corpus_rows, chips=chip_rows, n_cases=n_cases, total=total_line)
           + PANEL2
           + PANEL3
           + PANEL4.format(band_x=band_x, null_x=null_x, rp_x=rp_x, llm_x=llm_x,
                           rp_jsd=float(rp["jsd_rp"]), llm_jsd=float(llm["jsd_rp"]),
                           band_p95=float(llm["band_p95"]), null_p95=float(llm["nm_p95"]),
                           n_comp=n_comp, n_ref=n_ref,
                           n_half_a=n_ref // 2, n_half_b=n_ref - n_ref // 2,
                           ratio=ratio)
           + PANEL5.format(tam_rows=tam_rows, tam_sems=tam_sems)
           + FOOTER)
    OUT.write_text(tex, encoding="utf-8")
    print(f"wrote {OUT}")

    if not args.no_compile:
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", OUT.name],
                           cwd=OUT.parent, capture_output=True, text=True)
        ok = r.returncode == 0 and (OUT.parent / "pipeline_figure.pdf").exists()
        print("compiled pipeline_figure.pdf" if ok else f"pdflatex FAILED:\n{r.stdout[-800:]}")


# ---------------------------------------------------------------- template
# Layout: panel 1 full height, panels 2 and 3 stacked in one column, panel 4
# double width with the measurement mechanics spelled out, panel 5 full height.
# Only the values marked by format fields are computed.

HEADER = r"""%% Study pipeline for the LAK/JLA submission -- standalone TikZ.
%% GENERATED by analysis/semantic/oncoco/make_pipeline_tikz.py -- do not edit by hand.
%% Picture-first: each step shows its operation, one short line states its outcome.
\documentclass[border=1pt,10pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[scaled=0.92]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{sansmath}
\sansmath
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}

\definecolor{cReal}{HTML}{2C6E63}
\definecolor{cRP}{HTML}{4A6E9B}
\definecolor{cLLM}{HTML}{C0503C}
\definecolor{cAcc}{HTML}{7A5EA6}
\definecolor{cInk}{HTML}{1F2328}
\definecolor{cMeas}{HTML}{3F4A57}
\definecolor{cMute}{HTML}{70757D}
\definecolor{cRule}{HTML}{CFD4DA}
\definecolor{cPanel}{HTML}{FBFBFC}
\definecolor{cBand}{HTML}{DDE2E8}
\definecolor{cNull}{HTML}{B9C2CC}

\newcommand{\TINY}{\fontsize{4.7}{5.4}\selectfont}
\newcommand{\MICRO}{\fontsize{5.2}{6.0}\selectfont}

% #1 x0  #2 y0  #3 width  #4 height  #5 number  #6 title
\newcommand{\stepbox}[6]{%
  \draw[rounded corners=2.5pt, draw=cRule, fill=cPanel, line width=0.5pt,
        dash pattern=on 2.2pt off 1.6pt] (#1,#2) rectangle (#1+#3,#2+#4);
  \node[circle, fill=cInk, text=white, inner sep=0pt, minimum size=3.2mm,
        font=\bfseries\fontsize{5.6}{6.6}\selectfont] at (#1+0.36,#2+#4-0.38) {#5};
  \node[anchor=west, font=\scriptsize\bfseries, text=cInk, inner sep=0pt]
       at (#1+0.58,#2+#4-0.38) {#6};
}
% #1 x0  #2 y0  #3 width  #4 accent colour  #5 one-line outcome
\newcommand{\outcome}[5]{%
  \fill[#4!8, rounded corners=1.6pt] (#1+0.16,#2) rectangle (#1+#3-0.16,#2+0.46);
  \draw[#4, line width=0.9pt] (#1+0.16,#2) -- (#1+0.16,#2+0.46);
  \node[anchor=west, font=\MICRO\bfseries, text=#4, inner sep=0pt] at (#1+0.30,#2+0.23) {#5};
}
% #1 x  #2 y  #3 text
\newcommand{\ann}[3]{%
  \node[anchor=north west, font=\TINY, text=cMute, align=left, text width=2.80cm,
        inner sep=0pt] at (#1,#2) {#3};
}

\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm]
\path (0,0) rectangle (17.40,4.80);

\draw[-{Stealth[length=2.6pt,width=2.4pt]}, cMute!70, line width=0.8pt] (3.30,3.62) -- (3.50,3.62);
\draw[-{Stealth[length=2.6pt,width=2.4pt]}, cMute!70, line width=0.8pt] (5.16,2.49) -- (5.16,2.31);
\draw[-{Stealth[length=2.6pt,width=2.4pt]}, cMute!70, line width=0.8pt] (6.84,1.12) -- (7.04,1.12);
\draw[-{Stealth[length=2.6pt,width=2.4pt]}, cMute!70, line width=0.8pt] (13.92,2.40) -- (14.12,2.40);
"""

PANEL1 = r"""
%% ================================================= 1  three chat corpora
\stepbox{{0.00}}{{0.00}}{{3.24}}{{4.80}}{{1}}{{Three chat conditions}}

% each row: name, a span bar drawn to scale, the counts
\foreach \y/\c/\name/\w/\meta in {{%
{rows}}}{{
  \node[anchor=north west, font=\MICRO\bfseries, text=\c, inner sep=0pt]
       at (0.30,\y) {{\name}};
  \fill[cRule!45, rounded corners=0.8pt] (0.30,\y-0.44) rectangle (2.70,\y-0.26);
  \fill[\c!75, rounded corners=0.8pt] (0.30,\y-0.44) rectangle (0.30+\w,\y-0.26);
  \ann{{0.30}}{{\y-0.48}}{{\meta}}
}}
\draw[cRule, line width=0.4pt] (0.30,1.56) -- (2.94,1.56);
\foreach \i/\m in {{{chips}}}{{
  \pgfmathsetmacro{{\xa}}{{0.30 + \i*0.90}}
  \fill[cLLM!12, rounded corners=1.4pt] (\xa,1.06) rectangle (\xa+0.84,1.42);
  \node[anchor=center, font=\TINY, text=cLLM, inner sep=0pt] at (\xa+0.42,1.24) {{\m}};
}}
\ann{{0.30}}{{0.98}}{{three model families, {n_cases} case studies}}
\outcome{{0.00}}{{0.20}}{{3.24}}{{cInk}}{{{total}}}
"""

PANEL2 = r"""
%% ================================================= 2  learned segmentation
\stepbox{3.54}{2.55}{3.24}{2.25}{2}{Learned segmentation}

\node[anchor=west, font=\TINY, text=cRP, inner sep=0pt] at (3.86,4.01) {learned};
\foreach \xa/\w/\lab in {4.52/0.58/1, 5.16/0.74/2, 5.96/0.50/3}{
  \fill[cRP!16, rounded corners=1.4pt] (\xa,3.86) rectangle (\xa+\w,4.16);
  \draw[cRP!55, line width=0.4pt, rounded corners=1.4pt] (\xa,3.86) rectangle (\xa+\w,4.16);
  \node[anchor=center, font=\TINY, text=cRP, inner sep=0pt] at (\xa+\w/2,4.01) {\lab};
}
\node[anchor=west, font=\TINY, text=cMute, inner sep=0pt] at (3.86,3.57) {regex};
\foreach \xa/\w in {4.52/0.92, 5.52/0.94}{
  \fill[cMute!12, rounded corners=1.4pt] (\xa,3.42) rectangle (\xa+\w,3.72);
  \draw[cMute!45, line width=0.4pt, dash pattern=on 1.2pt off 1.2pt, rounded corners=1.4pt]
       (\xa,3.42) rectangle (\xa+\w,3.72);
}
\ann{3.86}{3.32}{adapter never sees study data}
\outcome{3.54}{2.63}{3.24}{cRP}{ordering robust under three units}
"""

PANEL3 = r"""
%% ================================================= 3  interaction coding
\stepbox{3.54}{0.00}{3.24}{2.25}{3}{Interaction coding}

\fill[cInk!8, rounded corners=1.8pt] (4.30,1.32) rectangle (5.80,1.70);
\draw[cInk!45, line width=0.4pt, rounded corners=1.8pt] (4.30,1.32) rectangle (5.80,1.70);
\node[anchor=center, font=\TINY, text=cInk, inner sep=0pt] at (5.05,1.51) {OnCoCo classifier};

% the label inventory, one cell per category (38 counselor + 28 client = the
% published 66-category scheme; a fixed property of the instrument, not a result)
\foreach \i in {0,...,65}{
  \pgfmathsetmacro{\col}{mod(\i,17)}
  \pgfmathsetmacro{\row}{floor(\i/17)}
  \pgfmathsetmacro{\xa}{3.88 + \col*0.13}
  \pgfmathsetmacro{\ya}{1.10 - \row*0.14}
  \ifnum\i<38
    \fill[cRP!70, rounded corners=0.3pt] (\xa,\ya) rectangle (\xa+0.10,\ya+0.10);
  \else
    \fill[cReal!70, rounded corners=0.3pt] (\xa,\ya) rectangle (\xa+0.10,\ya+0.10);
  \fi
}
\fill[cRP!70, rounded corners=0.3pt] (3.88,0.52) rectangle (3.98,0.62);
\node[anchor=west, font=\TINY, text=cMute, inner sep=0pt] at (4.05,0.57) {38 counselor};
\fill[cReal!70, rounded corners=0.3pt] (5.10,0.52) rectangle (5.20,0.62);
\node[anchor=west, font=\TINY, text=cMute, inner sep=0pt] at (5.27,0.57) {28 client};
\outcome{3.54}{0.04}{3.24}{cReal}{one classifier for all conditions}
"""

PANEL4 = r"""
%% ================================================= 4  anchored comparison
%% Three stacked rows: what is measured, how the two nulls are built, and the
%% ruler both are read on. The null constructions are drawn as mechanics --
%% disjoint brackets for the split-half band, full-span brackets for the
%% size-matched null -- because that contrast is the whole point of having two.
\stepbox{{7.08}}{{0.00}}{{6.78}}{{4.80}}{{4}}{{Anchored comparison}}

%% ---- row 1: what is measured -----------------------------------------
\node[anchor=west, font=\MICRO\bfseries, text=cInk, inner sep=0pt] at (7.44,4.08) {{the observed distance}};
\foreach \i/\h in {{0/0.36, 1/0.24, 2/0.16, 3/0.10, 4/0.06}}{{
  \pgfmathsetmacro{{\xa}}{{7.60 + \i*0.19}}
  \fill[cReal!60] (\xa,3.58) rectangle (\xa+0.13,3.58+\h);
}}
\foreach \i/\h in {{0/0.13, 1/0.33, 2/0.08, 3/0.25, 4/0.17}}{{
  \pgfmathsetmacro{{\xa}}{{8.98 + \i*0.19}}
  \fill[cLLM!65] (\xa,3.58) rectangle (\xa+0.13,3.58+\h);
}}
\draw[cRule, line width=0.4pt] (7.60,3.58) -- (8.56,3.58);
\draw[cRule, line width=0.4pt] (8.98,3.58) -- (9.94,3.58);
\node[anchor=north, font=\TINY, text=cReal, inner sep=1.2pt] at (8.08,3.54) {{HH real}};
\node[anchor=north, font=\TINY, text=cLLM, inner sep=1.2pt] at (9.46,3.54) {{H--LLM}};
\draw[{{Stealth[length=2.2pt,width=2.0pt]}}-{{Stealth[length=2.2pt,width=2.0pt]}},
      cMute, line width=0.5pt] (8.62,3.80) -- (8.92,3.80);
\node[anchor=south, font=\TINY, text=cInk, inner sep=1.0pt] at (8.77,3.84) {{JSD}};
\node[anchor=north west, font=\TINY, text=cMute, align=left, text width=3.44cm, inner sep=0pt]
     at (10.30,4.26) {{JSD is 0 when two label mixes are identical and grows as they part. It brings no scale of its own.}};

%% ---- row 2: how the two nulls are built ------------------------------
\node[anchor=west, font=\MICRO\bfseries, text=cInk, inner sep=0pt] at (7.44,3.22) {{what chance alone produces}};
\node[anchor=west, font=\TINY, text=cMute, inner sep=0pt] at (10.02,3.22) {{1{{,}}000 draws, both from HH real only}};

% split-half band: the reference cut in two, every conversation on one side
\fill[cBand] (7.46,2.79) rectangle (7.60,2.93);
\node[anchor=west, font=\TINY\bfseries, text=cInk, inner sep=0pt] at (7.66,2.86) {{split-half band}};
\foreach \i in {{0,...,7}}{{
  \pgfmathsetmacro{{\xa}}{{9.18 + \i*0.105}}
  \fill[cReal!70] (\xa,2.86) circle (0.9pt);
}}
\draw[cMute!60, line width=0.35pt, dash pattern=on 1.0pt off 1.0pt] (9.55,2.73) -- (9.55,2.99);
\draw[cMeas, line width=0.5pt] (9.14,2.96) -- (9.50,2.96);
\draw[cMeas, line width=0.5pt] (9.14,2.96) -- (9.14,2.92);
\draw[cMeas, line width=0.5pt] (9.50,2.96) -- (9.50,2.92);
\node[anchor=south, font=\TINY, text=cMeas, inner sep=0.6pt] at (9.32,2.97) {{{n_half_a}}};
\draw[cMeas, line width=0.5pt] (9.60,2.76) -- (9.96,2.76);
\draw[cMeas, line width=0.5pt] (9.60,2.76) -- (9.60,2.80);
\draw[cMeas, line width=0.5pt] (9.96,2.76) -- (9.96,2.80);
\node[anchor=north, font=\TINY, text=cMeas, inner sep=0.6pt] at (9.78,2.75) {{{n_half_b}}};
\draw[-{{Stealth[length=2.2pt,width=2.0pt]}}, cMute!70, line width=0.6pt] (10.10,2.86) -- (10.26,2.86);
\node[anchor=west, font=\TINY, text=cInk, inner sep=0pt] at (10.32,2.86) {{JSD}};
\node[anchor=north west, font=\TINY, text=cMute, align=left, text width=2.92cm, inner sep=0pt]
     at (10.84,2.98) {{HH real's {n_ref} client-side conversations cut in two disjoint halves}};

% size-matched null: both groups redrawn from the whole reference, with repeats
\fill[cNull] (7.46,2.19) rectangle (7.60,2.33);
\node[anchor=west, font=\TINY\bfseries, text=cInk, inner sep=0pt] at (7.66,2.26) {{size-matched null}};
\foreach \i in {{0,...,7}}{{
  \pgfmathsetmacro{{\xa}}{{9.18 + \i*0.105}}
  \fill[cReal!70] (\xa,2.26) circle (0.9pt);
}}
\draw[cMeas, line width=0.5pt] (9.14,2.36) -- (9.96,2.36);
\draw[cMeas, line width=0.5pt] (9.14,2.36) -- (9.14,2.32);
\draw[cMeas, line width=0.5pt] (9.96,2.36) -- (9.96,2.32);
\node[anchor=south, font=\TINY, text=cMeas, inner sep=0.6pt] at (9.55,2.37) {{{n_comp} draws}};
\draw[cMeas, line width=0.5pt] (9.14,2.16) -- (9.96,2.16);
\draw[cMeas, line width=0.5pt] (9.14,2.16) -- (9.14,2.20);
\draw[cMeas, line width=0.5pt] (9.96,2.16) -- (9.96,2.20);
\node[anchor=north, font=\TINY, text=cMeas, inner sep=0.6pt] at (9.55,2.15) {{{n_ref} draws}};
\draw[-{{Stealth[length=2.2pt,width=2.0pt]}}, cMute!70, line width=0.6pt] (10.10,2.26) -- (10.26,2.26);
\node[anchor=west, font=\TINY, text=cInk, inner sep=0pt] at (10.32,2.26) {{JSD}};
\node[anchor=north west, font=\TINY, text=cMute, align=left, text width=2.92cm, inner sep=0pt]
     at (10.84,2.38) {{both groups redrawn from all {n_ref}, with repeats, at the sizes really compared}};

%% ---- row 3: the ruler both nulls calibrate ---------------------------
% client side, {scale} cm per JSD unit; shaded bars run from 0 to each P95
\draw[cRule, line width=0.4pt] (7.56,1.42) -- (13.62,1.42);
\fill[cBand] (7.56,1.42) rectangle ({band_x:.3f},1.68);
\fill[cNull] (7.56,1.42) rectangle ({null_x:.3f},1.68);
\fill[cRP] ({rp_x:.3f},1.55) circle (2.0pt);
\fill[cLLM] ({llm_x:.3f},1.55) circle (2.4pt);
\node[anchor=south, font=\TINY, text=cRP, inner sep=1.6pt] at ({rp_x:.3f},1.70) {{HH roleplay}};
\node[anchor=south, font=\TINY, text=cLLM, inner sep=1.6pt] at ({llm_x:.3f},1.70) {{H--LLM}};
\node[anchor=north, font=\TINY, text=cRP, inner sep=1.6pt] at ({rp_x:.3f},1.40) {{{rp_jsd:.3f}}};
\node[anchor=north, font=\TINY\bfseries, text=cLLM, inner sep=1.6pt] at ({llm_x:.3f},1.40) {{{llm_jsd:.3f}}};
\node[anchor=east, font=\TINY, text=cMute, inner sep=1.6pt] at (7.52,1.55) {{0}};
\draw[cMeas, line width=0.5pt] ({null_x:.3f},1.68) -- ({null_x:.3f},1.79);
\node[anchor=south, font=\TINY, text=cMeas, inner sep=0.6pt] at ({null_x:.3f},1.80) {{size-matched P95 {null_p95:.3f}}};
\draw[cMeas, line width=0.5pt] ({band_x:.3f},1.42) -- ({band_x:.3f},1.20);
\node[anchor=north, font=\TINY, text=cMeas, inner sep=0.6pt] at ({band_x:.3f},1.19) {{split-half P95 {band_p95:.3f}}};

\node[anchor=west, font=\TINY, text=cMute, inner sep=0pt] at (7.56,0.84) {{client side throughout; Cram\'er's $V$ reported alongside as effect size}};
\outcome{{7.08}}{{0.20}}{{6.78}}{{cLLM}}{{{ratio:.1f}$\times$ beyond the noise floor}}
""".replace("{scale}", str(RULER_SCALE))

PANEL5 = r"""
%% ================================================= 5  convergent validity
\stepbox{{14.16}}{{0.00}}{{3.24}}{{4.80}}{{5}}{{Convergent validity}}

\draw[cMeas, line width=2.4pt] (14.44,4.02) -- (14.62,4.02);
\node[anchor=west, font=\TINY, text=cMeas, inner sep=0pt] at (14.69,4.02) {{measured distance}};
\draw[cAcc, line width=2.4pt] (16.14,4.02) -- (16.32,4.02);
\node[anchor=west, font=\TINY, text=cAcc, inner sep=0pt] at (16.39,4.02) {{perceived}};

\foreach \i/\j/\r/\m in {{{tam_rows}}}{{
  \pgfmathsetmacro{{\xa}}{{14.52 + \i*0.90}}
  \pgfmathsetmacro{{\hj}}{{\j/{jsd_max}*{bar_h}}}
  \pgfmathsetmacro{{\hr}}{{(\r-1)/4*{bar_h}}}
  \fill[cMeas!85] (\xa,1.94) rectangle (\xa+0.26,1.94+\hj);
  \fill[cAcc!75] (\xa+0.30,1.94) rectangle (\xa+0.56,1.94+\hr);
  \node[anchor=south, font=\TINY, text=cMeas, inner sep=0.8pt] at (\xa+0.13,1.94+\hj) {{\j}};
  \node[anchor=south, font=\TINY, text=cAcc, inner sep=0.8pt] at (\xa+0.43,1.94+\hr) {{\r}};
  \node[anchor=north, font=\TINY\bfseries, text=cInk, inner sep=1.4pt] at (\xa+0.41,1.90) {{\m}};
}}
\draw[cRule, line width=0.4pt] (14.44,1.94) -- (17.14,1.94);
{tam_sems}
\ann{{14.44}}{{1.40}}{{one course line, three semesters, client deployment exchanged}}
\outcome{{14.16}}{{0.20}}{{3.24}}{{cAcc}}{{perception tracks the measurement}}
""".replace("{jsd_max}", str(TAM_JSD_MAX)).replace("{bar_h}", str(TAM_BAR_H))

FOOTER = r"""
\end{tikzpicture}
\end{document}
"""

if __name__ == "__main__":
    main()
