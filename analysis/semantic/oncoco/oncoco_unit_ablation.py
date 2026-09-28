#!/usr/bin/env python3
"""Assemble the unit-of-analysis robustness table (replaces the old segmentation ablation).

The RQ1 proximity gap delta' = JSD(HH real, H-LLM) - JSD(HH real, HH roleplay) is reported
across four units so that no single segmentation choice carries the result:

  SaT spans        SaT-6l + LoRA adapter, the paper's primary unit
  Regex spans      corrected sentence regex
  Messages         no segmentation at all. NOTE: OnCoCo 1.0 is annotated on utterances
                   (typically one sentence or less), NOT on whole messages, so this unit
                   is the most out-of-distribution one for the classifier, not the least.
  Regex (published) the regex as originally shipped -- an artifact, kept as a record

Reads : results/tables/oncoco_seg_ordering_bootstrap.csv   (oncoco_seg_ordering_bootstrap.py)
        results/tables/oncoco_noise_band{,_regex,_message}.csv (oncoco_noise_band_analysis.py)
Writes: results/tables/oncoco_unit_ablation.tex   -> main paper, replaces Table 3
        results/tables/oncoco_unit_noise_band.tex -> supplement
        results/tables/oncoco_unit_ablation.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "results" / "tables"

# display name -> (bootstrap `unit` value, noise-band csv suffix)
UNITS = [
    ("SaT spans", "SaT", ""),
    ("Regex spans", "regex", "_regex"),
    ("Messages", "message", "_message"),
    ("Naive character split", "regex-published", None),
]

boot = pd.read_csv(T / "oncoco_seg_ordering_bootstrap.csv")
missing = [u for _, u, _ in UNITS if u not in set(boot["unit"])]
if missing:
    raise SystemExit(
        f"missing units {missing} in oncoco_seg_ordering_bootstrap.csv; run\n"
        f"  oncoco_seg_ordering_bootstrap.py --units SaT,regex,message,regex-published")

# --- main table: ordering across units ----------------------------------------
rows, csv_rows = [], []
for disp, unit, _ in UNITS:
    cells = []
    for speaker in ("Client", "Counselor"):
        r = boot[(boot["unit"] == unit) & (boot["speaker"] == speaker)].iloc[0]
        sig = "" if (r["ci_lo"] <= 0 <= r["ci_hi"]) else r"$^{*}$"
        cells += [f"{r['jsd_real_vs_hllm']:.3f}", f"{r['jsd_real_vs_roleplay']:.3f}",
                  f"{r['delta_point']:+.3f}{sig} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"]
        csv_rows.append(dict(unit=disp, speaker=speaker,
                             jsd_real_vs_hllm=r["jsd_real_vs_hllm"],
                             jsd_real_vs_roleplay=r["jsd_real_vs_roleplay"],
                             delta=r["delta_point"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             excludes_zero=not (r["ci_lo"] <= 0 <= r["ci_hi"])))
    rows.append(f"{disp} & " + " & ".join(cells) + r" \\")

tex = [
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r" & \multicolumn{3}{c}{Client} & \multicolumn{3}{c}{Counselor} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
    r"Unit of analysis & real--LLM & real--roleplay & $\Delta$ [95\% CI] "
    r"& real--LLM & real--roleplay & $\Delta$ [95\% CI] \\",
    r"\midrule",
    *rows[:3],
    r"\midrule",
    rows[3],
    r"\bottomrule",
    r"\end{tabular}",
]
(T / "oncoco_unit_ablation.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
pd.DataFrame(csv_rows).to_csv(T / "oncoco_unit_ablation.csv", index=False)

print("Proximity gap across units: delta' = JSD(HH real, H-LLM) - JSD(HH real, HH roleplay)\n")
print(f"  {'unit':28s} {'speaker':10s} {'real-LLM':>9s} {'real-rp':>8s} {'delta':>8s}  95% CI")
for r in csv_rows:
    mark = " *" if r["excludes_zero"] else "  (CI includes 0)"
    print(f"  {r['unit']:28s} {r['speaker']:10s} {r['jsd_real_vs_hllm']:9.3f} "
          f"{r['jsd_real_vs_roleplay']:8.3f} {r['delta']:+8.3f}  "
          f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]{mark}")

# --- supplement table: noise band across the three legitimate units -----------
band_rows = []
for disp, unit, suffix in UNITS:
    if suffix is None:
        continue
    p = T / f"oncoco_noise_band{suffix}.csv"
    if not p.exists():
        print(f"\nWARNING: {p.name} missing; skipping it in the supplement table.")
        continue
    b = pd.read_csv(p)
    b = b[b["comparison"].isin(["HH_roleplay_chat", "H_LLM_roleplay_chat"])]
    for _, r in b.iterrows():
        comp = "HH roleplay" if r["comparison"] == "HH_roleplay_chat" else "H--LLM"
        band_rows.append(dict(unit=disp, comparison=comp, speaker=r["speaker"],
                              jsd=r["jsd_rp"], band_mean=r["band_mean"], band_p95=r["band_p95"],
                              pctl=r["band_pctl_of_jsd"], nm_p95=r["nm_p95"], v=r["cramers_v"]))

bt = [
    r"\begin{tabular}{llcrrrrr}",
    r"\toprule",
    r"Unit & Comparison & Role & JSD & Band mean & Band P95 & Pctl. & $V$ \\",
    r"\midrule",
]
last = None
for r in band_rows:
    lead = r["unit"] if r["unit"] != last else ""
    last = r["unit"]
    bt.append(f"{lead} & {r['comparison']} & {r['speaker']} & {r['jsd']:.3f} & "
              f"{r['band_mean']:.3f} & {r['band_p95']:.3f} & {r['pctl']:.1f} & {r['v']:.3f} " + r"\\")
bt += [r"\bottomrule", r"\end{tabular}"]
(T / "oncoco_unit_noise_band.tex").write_text("\n".join(bt) + "\n", encoding="utf-8")

print(f"\nWrote {T / 'oncoco_unit_ablation.tex'}")
print(f"Wrote {T / 'oncoco_unit_noise_band.tex'}")
print(f"Wrote {T / 'oncoco_unit_ablation.csv'}")
