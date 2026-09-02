#!/usr/bin/env python3
"""Do the between-condition distances survive removing categories that are tied to the training
setting rather than to counseling behavior?

A reviewer's objection: some OnCoCo categories are structurally determined by the exercise (the
assignment brief, the platform's opening ritual) rather than by how the client behaves, so part of
the measured distance between real counseling and roleplay is mechanical, not behavioral.

The exclusion sets below are PRE-DECLARED with a one-line rationale each, and reproduced verbatim
in the supplement so that the choice is auditable:

  S1  setting-bound (primary)
      CL-IF-AO-*-Obj-*   Objective of the assignment      the "Auftrag" is supplied by the persona
                                                          prompt and the course task; a simulated
                                                          client states it because it is in context
      CL-IF-AO-*-Ext-*   Extension of the assignment      same construct family, exercise-defined
      CO-IF-AO-*-ICO-*   Definition of counseling objs.   counselor mirror of the same construct
      CL-FB-*-*-*-*      Client opening formalities       the opening ritual is platform-dictated:
      CO-FA-*-*-*-*      Counselor opening formalities    archived tickets open with a system join
                                                          event, H-LLM sessions with a scripted turn
      CL-O-*-*-UCO-*     Inappropriate remark (client)    NOT a setting argument: excluded as a
      CO-O-*-*-UCO-*     Inappropriate remark (counselor) classifier artifact. It is the only label
                                                          predicted with low confidence throughout
                                                          (median 0.48-0.61 vs >=0.88 for every
                                                          signature category) and absorbs 6.5% of
                                                          HH real counselor spans against ~1%
                                                          elsewhere, on benign text. Established
                                                          empirically in the confidence diagnostics
                                                          BEFORE this analysis was run.

  S2  measurement residual (secondary)
      CL-O-*-*-O-*, CO-O-*-*-O-*   the catch-all sink for anything the classifier cannot place.
      Kept as a separate tier because it tests a different threat (measurement error, not a setting
      confound); merging the two would let one exclusion set carry two arguments.

  S3  closing formalities (tertiary, reported but not pre-declared)
      CO-FC-*-*-F-*, CL-FC-*-*-F-*   session end is partly clock-driven in the exercise, but closing
      IS a counseling skill, so the case is weaker and we report rather than assume it.

DELIBERATELY NOT EXCLUDED: CL-IF-ACP-*-Cons-* (Consent). It is a backchannel/assent class, not a
setting artifact, and it is the single largest client-side driver. Removing the biggest effect
driver *because* it is big would be the very post-hoc move the objection warns about. It appears
instead as one point among all others in the leave-one-out sweep below, which is the honest way to
show what any single category contributes.

Outputs: results/tables/oncoco_category_sensitivity.csv / .tex
         results/tables/oncoco_leave_one_out.csv
         results/figures/oncoco/leave_one_out_delta.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

# Reuse the paper's own statistics rather than reimplementing them.
from oncoco_noise_band_analysis import (  # noqa: E402
    count_matrix, cramers_v, js_distance, load_rows, size_matched_null,
    span_labels_by_conversation, split_half_band,
)

TABLES = ROOT / "results/tables"
FIGS = ROOT / "results/figures/oncoco"
NORM = ROOT / "data/processed/combined/normalized"

CONDITIONS = ["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat"]
REF = "HH_real_chat"
SPEAKERS = ["Client", "Counsellor"]

S1 = ["CL-IF-AO-*-Obj-*", "CL-IF-AO-*-Ext-*", "CO-IF-AO-*-ICO-*",
      "CL-FB-*-*-*-*", "CO-FA-*-*-*-*", "CL-O-*-*-UCO-*", "CO-O-*-*-UCO-*"]
S2 = ["CL-O-*-*-O-*", "CO-O-*-*-O-*"]
S3 = ["CO-FC-*-*-F-*", "CL-FC-*-*-F-*"]

SETS = [("full", []), ("S1", S1), ("S1+S2", S1 + S2), ("S1+S2+S3", S1 + S2 + S3)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-json", default=str(NORM / "oncoco_classification_all.json"))
    p.add_argument("--n-splits", type=int, default=1000)
    p.add_argument("--n-null", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def unit_ablation_ci(speaker: str):
    """95% CI of the full-support delta as computed by the unit-ablation run.

    The "full" row here is the same quantity as that run's SaT row. Two independent
    2,000-draw bootstraps of it disagree in the third decimal (seed-to-seed spread is
    about 0.003 at this B), so printing both would show two intervals for one number.
    The baseline interval is therefore sourced from that run rather than recomputed.
    """
    path = TABLES / "oncoco_unit_ablation.csv"
    df = pd.read_csv(path)
    row = df[(df["unit"] == "SaT spans") & (df["speaker"] == speaker)]
    if len(row) != 1:
        raise SystemExit(f"{path}: expected one 'SaT spans'/{speaker} row, found {len(row)}")
    return float(row["ci_lo"].iloc[0]), float(row["ci_hi"].iloc[0])


def drop(conv_labels, excluded):
    """Remove excluded labels, then drop conversations left with nothing."""
    out = {}
    for conv, labs in conv_labels.items():
        keep = [l for l in labs if l not in excluded]
        if keep:
            out[conv] = keep
    return out


def v_thresholds(k):
    """Cohen's small/medium/large benchmarks for Cramer's V on a 2 x k contingency table.

    V = w / sqrt(df*) with df* = min(rows, cols) - 1. The tables here have two rows
    (the two conditions), so df* = 1 regardless of k and Cohen's w benchmarks
    .10/.30/.50 carry over to V unscaled. Dividing by sqrt(k-1) uses the number of
    categories as the degrees of freedom and makes the thresholds roughly sevenfold
    too lenient at the observed support.
    """
    del k  # thresholds do not depend on the number of categories for a 2 x k table
    return (0.1, 0.3, 0.5)


def _v_thresholds_legacy(k):
    """Previous (incorrect) scaling, kept only so the change is auditable."""
    if k < 2:
        return (np.nan,) * 3
    return tuple(w / np.sqrt(k - 1) for w in (0.1, 0.3, 0.5))


def main():
    args = parse_args()
    rows = load_rows(Path(args.data_json))

    results, loo_rows = [], []
    for set_name, excluded in SETS:
        rng = np.random.default_rng(args.seed)  # same stream per set, so sets are comparable
        ex = set(excluded)
        for speaker in SPEAKERS:
            per_cond = {c: drop(span_labels_by_conversation(rows, c, speaker), ex)
                        for c in CONDITIONS}
            labels = sorted({l for c in CONDITIONS for labs in per_cond[c].values() for l in labs})
            idx = {l: i for i, l in enumerate(labels)}
            mats = {c: count_matrix(per_cond[c], idx)[1] for c in CONDITIONS}
            pooled = {c: mats[c].sum(axis=0) for c in CONDITIONS}
            k = len(labels)
            small, medium, large = v_thresholds(k)

            band = split_half_band(mats[REF], args.n_splits, rng)
            for comp in ("HH_roleplay_chat", "H_LLM_roleplay_chat"):
                jsd = js_distance(pooled[comp], pooled[REF])
                nm = size_matched_null(mats[REF], mats[comp].shape[0], mats[REF].shape[0],
                                       args.n_null, rng)
                results.append(dict(
                    exclusion_set=set_name, n_excluded=len(ex), k_labels=k,
                    comparison=comp, speaker="Counselor" if speaker == "Counsellor" else speaker,
                    jsd=round(jsd, 4), band_mean=round(float(band.mean()), 4),
                    band_p95=round(float(np.percentile(band, 95)), 4),
                    band_pctl=round(float((band < jsd).mean()) * 100, 1),
                    ratio_to_floor=round(jsd / float(band.mean()), 1),
                    nm_p95=round(float(np.percentile(nm, 95)), 4),
                    cramers_v=round(cramers_v(pooled[comp], pooled[REF]), 3),
                    v_large_threshold=round(large, 3)))

            # ordering bootstrap under this exclusion set
            deltas = np.empty(args.n_boot)
            for b in range(args.n_boot):
                s = {c: mats[c][rng.integers(0, mats[c].shape[0], mats[c].shape[0])].sum(axis=0)
                     for c in CONDITIONS}
                deltas[b] = (js_distance(s[REF], s["H_LLM_roleplay_chat"])
                             - js_distance(s["HH_roleplay_chat"], s["H_LLM_roleplay_chat"]))
            point = (js_distance(pooled[REF], pooled["H_LLM_roleplay_chat"])
                     - js_distance(pooled["HH_roleplay_chat"], pooled["H_LLM_roleplay_chat"]))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            # Keep the draw above so the RNG stream stays identical across sets, then take the
            # baseline interval from the unit-ablation run so both tables report one number.
            if set_name == "full":
                lo, hi = unit_ablation_ci("Counselor" if speaker == "Counsellor" else speaker)
            results.append(dict(
                exclusion_set=set_name, n_excluded=len(ex), k_labels=k, comparison="ordering delta",
                speaker="Counselor" if speaker == "Counsellor" else speaker,
                jsd=round(point, 4), band_mean=np.nan, band_p95=np.nan, band_pctl=np.nan,
                ratio_to_floor=np.nan, nm_p95=round(float(lo), 4),
                cramers_v=round(float(hi), 4), v_large_threshold=np.nan))
            print(f"{set_name:9s} {speaker:11s} k={k:3d}  "
                  f"JSD(real,LLM)={js_distance(pooled[REF], pooled['H_LLM_roleplay_chat']):.3f}  "
                  f"delta={point:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"{'  *' if lo > 0 else '   (CI includes 0)'}")

            # ---- leave-one-out sweep (full support, no resampling) ----
            if set_name == "full":
                base = point
                for lab in labels:
                    one = {c: drop(per_cond[c], {lab}) for c in CONDITIONS}
                    labs2 = sorted({l for c in CONDITIONS for v in one[c].values() for l in v})
                    idx2 = {l: i for i, l in enumerate(labs2)}
                    pv = {c: count_matrix(one[c], idx2)[1].sum(axis=0) for c in CONDITIONS}
                    d = (js_distance(pv[REF], pv["H_LLM_roleplay_chat"])
                         - js_distance(pv["HH_roleplay_chat"], pv["H_LLM_roleplay_chat"]))
                    share = float(sum(pooled[c][idx[lab]] for c in CONDITIONS)
                                  / sum(pooled[c].sum() for c in CONDITIONS))
                    loo_rows.append(dict(
                        speaker="Counselor" if speaker == "Counsellor" else speaker,
                        label=lab, pooled_share=round(share, 5),
                        delta_without=round(d, 4), delta_full=round(base, 4),
                        change=round(d - base, 4),
                        in_S1=lab in S1, in_S2=lab in S2, in_S3=lab in S3))

    res = pd.DataFrame(results)
    res.to_csv(TABLES / "oncoco_category_sensitivity.csv", index=False)
    loo = pd.DataFrame(loo_rows).sort_values("change")
    loo.to_csv(TABLES / "oncoco_leave_one_out.csv", index=False)

    print("\nLargest single-category effects on the client-side ordering:")
    cl = loo[loo["speaker"] == "Client"]
    print(cl.nsmallest(5, "change")[["label", "pooled_share", "delta_without", "change"]]
          .to_string(index=False))
    print("  ... and the categories whose removal increases it:")
    print(cl.nlargest(3, "change")[["label", "pooled_share", "delta_without", "change"]]
          .to_string(index=False))

    # ---- supplement table ----
    lines = [r"\begin{tabular}{llrrrrc}", r"\toprule",
             r"Exclusion set & Role & $k$ & JSD real--LLM & Band mean & Pctl. & "
             r"$\Delta$ [95\% CI] \\", r"\midrule"]
    for set_name, _ in SETS:
        for sp in ("Client", "Counselor"):
            r_j = res[(res["exclusion_set"] == set_name) & (res["speaker"] == sp)
                      & (res["comparison"] == "H_LLM_roleplay_chat")]
            r_d = res[(res["exclusion_set"] == set_name) & (res["speaker"] == sp)
                      & (res["comparison"] == "ordering delta")]
            if not len(r_j) or not len(r_d):
                continue
            j, dd = r_j.iloc[0], r_d.iloc[0]
            star = r"$^{*}$" if dd["nm_p95"] > 0 else ""
            lines.append(f"{set_name} & {sp} & {int(j['k_labels'])} & {j['jsd']:.3f} & "
                         f"{j['band_mean']:.3f} & {j['band_pctl']:.1f} & "
                         f"{dd['jsd']:+.3f}{star} [{dd['nm_p95']:.3f}, {dd['cramers_v']:.3f}] " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "oncoco_category_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- leave-one-out figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
        for ax, sp in zip(axes, ("Client", "Counselor")):
            s = loo[loo["speaker"] == sp]
            base = s["delta_full"].iloc[0]
            ax.axhline(base, color="#666666", lw=1, ls="--")
            ax.scatter(s["pooled_share"], s["delta_without"], s=16, color="#0072B2", alpha=.75,
                       edgecolors="none")
            for _, r in s.reindex(s["change"].abs().nlargest(3).index).iterrows():
                ax.annotate(r["label"].split("-")[-2], (r["pooled_share"], r["delta_without"]),
                            fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("symlog", linthresh=1e-4)
            ax.set_xlabel("pooled share of the removed category")
            ax.set_title(sp, fontsize=10)
            ax.grid(alpha=.3, lw=.5)
        axes[0].set_ylabel(r"ordering $\Delta$ without that category")
        fig.tight_layout()
        FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGS / "leave_one_out_delta.pdf")
        fig.savefig(FIGS / "leave_one_out_delta.png", dpi=200)
        print(f"\nWrote {FIGS / 'leave_one_out_delta.pdf'}")
    except Exception as exc:
        print(f"\n(figure skipped: {exc})")

    print(f"Wrote {TABLES / 'oncoco_category_sensitivity.csv'} and companions")


if __name__ == "__main__":
    main()
