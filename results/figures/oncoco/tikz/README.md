# TikZ figures for the LAK/JLA submission

Two full-width candidates for the lead figure of `oncoco_analysis_paper.tex`.
Both are `standalone` documents, compiled to PDF and included with
`\includegraphics`. Nothing is added to the paper's preamble.

| file | what it is | size |
|---|---|---|
| `hero_figure.pdf` | result-first hero: the gap, the signature, the convergence | 17.4 × 7.3 cm |
| `pipeline_figure.pdf` | five-step study pipeline, picture-first | 17.4 × 4.8 cm |
| `noise_band_figure.pdf` | Figure 3: every comparison on HH real chat's split-half band, two role panels | 17.4 × 5.05 cm |
| `hmm_phase_figure.pdf` | Figure 4: three-state HMM phase structure, one panel per chat condition | 13.05 × 4.85 cm |

`*_preview.png` are 260 dpi renders for quick viewing, not used by the paper.

## Building

```bash
python3 analysis/semantic/oncoco/make_hero_tikz.py       # regenerates hero_figure.tex
python3 analysis/semantic/oncoco/make_pipeline_tikz.py   # regenerates + compiles pipeline_figure
python3 analysis/semantic/oncoco/make_noise_band_tikz.py # regenerates + compiles noise_band_figure
python3 analysis/semantic/oncoco/make_hmm_phase_tikz.py  # regenerates + compiles hmm_phase_figure
pdflatex hero_figure.tex
```

The hero is generated, not hand-written: every number and every decile series is
read from the analysis outputs, so the figure follows the results.

| panel | source |
|---|---|
| 1 triangle, counselor strip | `results/tables/oncoco_rq1_distances.csv` (distances), `oncoco_noise_band.csv` (bands, ratio, *V*) |
| 2 decile series | `results/tables/oncoco_temporal_distribution.csv`, bucketed exactly as `oncoco_behavioral_signatures.py` plus a Rejection bucket |
| 3 bars | `results/tables/oncoco_tam_convergence.tex` |

Read the *unrounded* `oncoco_noise_band.csv`, not `oncoco_noise_band_chat.tex`:
the rounded band mean (0.111 instead of 0.1105) turns the 4.4× ratio reported in
Section 5.2 into 4.3×.

The panel-1 triangle is drawn to scale. JSD is a metric, so the three client-side
distances (0.270 / 0.375 / 0.482) embed exactly in the plane; the vertex positions
are solved from them, not placed by hand. Changing the distances moves the vertices.

`pipeline_figure.tex` is generated as well (`make_pipeline_tikz.py`) — it was
hand-written until 2026-08-15, and every corpus update silently left stale numbers
behind in it. Each step shows its operation rather than describing it — corpus bars
on a common span scale, one turn splitting into spans under two segmenters, 66 cells
for the 66 OnCoCo categories, a distance ruler with the noise band and the
size-matched null shaded on it, paired bars per semester.

| panel | source |
|---|---|
| 1 corpus bars, totals, model chips | `oncoco_label_distribution.csv`, `oncoco_within_condition_jsd.csv`, `oncoco_persona_cases.csv`, `oncoco_hllm_model_realness_both_baselines.csv` |
| 2, 3 | static illustration of fixed method choices |
| 4 distance ruler | `oncoco_noise_band.csv` (unrounded, same convention as the hero) |
| 5 semester bars | `oncoco_tam_convergence.tex` |

The last hand-written version is kept as `pipeline_figure.tex.bak_handwritten`.

## Including in the paper

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{results/figures/oncoco/tikz/hero_figure.pdf}
\caption{...}
\label{fig:hero}
\end{figure*}
```

The natural width is 495.7 pt against a JLA text width of 498.9 pt, so
`width=\textwidth` scales by 0.6\% — a 4.7 pt label stays at 4.7 pt.

## `noise_band_figure`

Replaces the former Tables 4 and 5 of the main paper. One row per comparison,
two panels for the two speaker roles, a shared JSD axis so client and counselor
side are read against each other. Dot = JSD to HH real chat, shaded band = HH
real chat split against itself (mean to P95), tick = size-matched null P95,
bars = 95% conversation-level bootstrap CI (per model), right column = Cramér's
*V*. The exact numbers moved to Supplementary Tables S12 and S13.

| element | source |
|---|---|
| dots, band, null ticks, *V*, ×floor | `results/tables/oncoco_noise_band.csv` (unrounded) |
| per-model CIs | `results/tables/oncoco_rq1b_model_realness_bootstrap.csv` |

## `hmm_phase_figure`

Generated since 2026-08-24 (the 2026-08-17 draft was hand-authored and is kept as
`hmm_phase_figure.tex.bak_handwritten`). Node geometry is identical across the
three panels, so only phase size and transition weight differ. States are matched
to Opening / Clarify / third by their leading emission rather than by fitted index,
so a refit cannot silently swap the panels. The third state carries the same
structural name (*Phase 3*) in all three panels so that they stay comparable —
what it is made of differs and is said in the line underneath (a Help-led
solution phase in H--LLM, a residual phase in both human conditions).

Laid out at 13.05 cm, so `width=0.75\textwidth` scales by 1.0 and the 4.8 pt
labels stay at 4.8 pt. Opening and Clarify keep their self-loop but not its
number — the text cites neither, and the narrow panel needs the margin.

Circle area and the strip below each panel encode occupancy on the same scale.
Arrow width is linear in the transition probability (`w = 40 p`, through the
origin) for the Clarify ↔ third pair only — the Opening arrow carries a
much larger probability and is drawn at fixed width, which the caption states.

| element | source |
|---|---|
| occupancy, leading labels | `results/tables/oncoco_hmm_emissions_v2.csv` |
| transition probabilities | `results/tables/oncoco_hmm_transitions_v2.csv` |
| conversation and span counts | `results/tables/oncoco_hmm_model_selection.csv` (k=3) |
