#!/usr/bin/env python3
"""Is classifier uncertainty correlated with condition? And does the RQ1 proximity gap
delta' = JSD(HH real, H-LLM) - JSD(HH real, HH roleplay) survive if we keep only the items
the classifier is confident about?

PRE-DECLARED INTERPRETATIONS (written before the sidecar had finished running, so that neither
outcome can be rationalised after the fact):

  (i)  If H-LLM spans carry LOWER confidence / HIGHER entropy than the human conditions, then LLM
       text is partly out of distribution for a classifier trained on human-written counseling
       messages. That is a genuine caveat: it would mean the absolute H-LLM label proportions carry
       extra noise, and we would report it as bounding the absolute magnitudes while leaving the
       ordering to the high-confidence sensitivity analysis in section 5.

  (ii) If HH REAL spans carry the lowest confidence -- which is what the OnCoCo 1.0 limitations
       predict, since its training data are synthetic, expert-written and idealised, and explicitly
       contain little informal or boundary-crossing language -- then the noisiest condition is the
       REFERENCE. Because every distance in the paper's Table 4 is scaled by HH real chat's own
       split-half band, noise in the reference inflates that band and therefore makes the reported
       percentile positions CONSERVATIVE rather than optimistic. We would report it that way, and
       section 5 is what keeps that from being a convenient story.

  (iii) If confidence does not separate the conditions at all, sections 1-4 are a null result and
       only section 5 carries weight.

  In every case section 5 is the decisive test: if the proximity gap survives restricting the
  analysis to high-confidence items, "classifier error drives the effect" is not viable.

MANDATORY CONTROL: confidence tracks item length, and the conditions differ in length. Every
between-condition contrast is therefore also reported within token-length deciles and under a
length-matched resample. Without that, the analysis only measures length.

Softmax probabilities are NOT calibrated. They are used only for relative comparisons between
conditions and for thresholding, never as a probability of correctness.

Input : data/processed/combined/normalized/oncoco_confidence_sat.csv.gz  (oncoco_confidence_sidecar.py)
Output: results/tables/oncoco_confidence_by_condition.csv / .tex
        results/tables/oncoco_confidence_length_strata.csv
        results/tables/oncoco_confidence_signature_labels.csv
        results/tables/oncoco_rq1_highconf_sensitivity.csv / .tex
        results/figures/oncoco/confidence_ecdf.pdf
"""
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

ROOT = Path(__file__).resolve().parents[3]
NORM = ROOT / "data/processed/combined/normalized"
TABLES = ROOT / "results/tables"
FIGS = ROOT / "results/figures/oncoco"

CHAT = ["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat"]
SHORT = {"HH_roleplay_chat": "HH roleplay", "HH_real_chat": "HH real", "H_LLM_roleplay_chat": "H--LLM"}
SPEAKERS = ["Client", "Counsellor"]
SIGNATURE = {
    "CL-IF-ACP-*-Rej-*": "Rejection",
    "CL-IF-ACP-*-Req-*": "General request",
    "CL-IF-ACP-*-PS-*": "Problem statement",
    "CL-IF-ACP-*-Cons-*": "Consent",
    "CL-IF-AO-*-Obj-*": "Objective of the assignment",
    "CL-O-*-*-O-*": "Other statements (client)",
    "CO-O-*-*-UCO-*": "Inappropriate remark (counselor)",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar", default=str(NORM / "oncoco_confidence_sat.csv.gz"))
    p.add_argument("--unit", choices=["span", "message"], default="span")
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--n-perm", type=int, default=2000,
                   help="Conversation-label permutations for the clustered p-values.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cliffs_delta(a, b):
    """Exact Cliff's delta from the Mann-Whitney U statistic (O(n log n), not O(n*m))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def cluster_boot_median_diff(df_a, df_b, col, rng, n_boot):
    """Bootstrap the difference in medians, resampling CONVERSATIONS not items."""
    ga = [g[col].to_numpy() for _, g in df_a.groupby("conv")]
    gb = [g[col].to_numpy() for _, g in df_b.groupby("conv")]
    if not ga or not gb:
        return np.nan, np.nan
    out = np.empty(n_boot)
    for i in range(n_boot):
        sa = np.concatenate([ga[j] for j in rng.integers(0, len(ga), len(ga))])
        sb = np.concatenate([gb[j] for j in rng.integers(0, len(gb), len(gb))])
        out[i] = np.median(sa) - np.median(sb)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def cluster_perm_p(df_a, df_b, col, rng, n_perm):
    """Two-sided p for the median difference, permuting CONDITION LABELS over conversations.

    The Mann-Whitney and Kruskal-Wallis statistics in this script are computed over spans,
    of which there are up to eleven thousand and which are strongly correlated within a
    conversation. Their p-values are therefore far too small: they count every span as
    independent evidence. This test keeps each conversation's spans together and permutes
    only which condition a conversation belongs to, which is the level at which the
    conditions actually differ. Report this one, not the item-level p.
    """
    ga = [g[col].to_numpy() for _, g in df_a.groupby("conv")]
    gb = [g[col].to_numpy() for _, g in df_b.groupby("conv")]
    if not ga or not gb:
        return np.nan
    pool = ga + gb
    n_a = len(ga)
    observed = abs(np.median(np.concatenate(ga)) - np.median(np.concatenate(gb)))
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(pool))
        sa = np.concatenate([pool[j] for j in perm[:n_a]])
        sb = np.concatenate([pool[j] for j in perm[n_a:]])
        if abs(np.median(sa) - np.median(sb)) >= observed:
            hits += 1
    return (1.0 + hits) / (n_perm + 1.0)


def js_distance(p, q):
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))

    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def pooled_vec(df, labels):
    c = df["pred_label"].value_counts()
    return np.array([float(c.get(l, 0)) for l in labels])


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    # separate stream for the clustered permutation test, so adding it does not shift the
    # bootstrap draws of section 5 and silently move the published CI bounds
    perm_rng = np.random.default_rng(args.seed + 1)

    d = pd.read_csv(args.sidecar)
    d = d[(d["unit"] == args.unit) & (d["source"].isin(CHAT))].copy()
    d["conv"] = d["source_file"].astype(str) + "::" + d["id"].astype(str)
    print(f"{len(d)} {args.unit} items across {d['conv'].nunique()} chat conversations\n")

    # --- 1. descriptive ------------------------------------------------------
    rows = []
    for sp in SPEAKERS:
        for cond in CHAT:
            s = d[(d["speaker_type"] == sp) & (d["source"] == cond)]
            rows.append(dict(speaker=sp, condition=cond, n=len(s),
                             conv=s["conv"].nunique(),
                             top1_median=s["top1_prob"].median(),
                             top1_q1=s["top1_prob"].quantile(.25),
                             top1_q3=s["top1_prob"].quantile(.75),
                             entropy_median=s["entropy_bits"].median(),
                             share_below_50=float((s["top1_prob"] < .5).mean()),
                             share_below_90=float((s["top1_prob"] < .9).mean()),
                             tokens_median=s["n_tokens"].median()))
    desc = pd.DataFrame(rows)
    desc.to_csv(TABLES / "oncoco_confidence_by_condition.csv", index=False)
    print("1. Confidence by condition")
    print(desc.round(4).to_string(index=False))

    # --- 2. tests ------------------------------------------------------------
    print("\n2. Between-condition tests (conversation-clustered; item-level p reported only "
          "for reference and NOT to be cited)")
    test_rows = []
    for sp in SPEAKERS:
        sub = d[d["speaker_type"] == sp]
        groups = [sub[sub["source"] == c]["top1_prob"].to_numpy() for c in CHAT]
        if min(len(g) for g in groups) == 0:
            continue
        h, p_kw = kruskal(*groups)
        print(f"  {sp}: Kruskal-Wallis over spans H={h:.1f}, p={p_kw:.3g} "
              f"(item level, spans are not independent)")
        pairs = [(CHAT[i], CHAT[j]) for i in range(3) for j in range(i + 1, 3)]
        pvals, recs = [], []
        for a, b in pairs:
            da, db = sub[sub["source"] == a], sub[sub["source"] == b]
            u, p_item = mannwhitneyu(da["top1_prob"], db["top1_prob"], alternative="two-sided")
            lo, hi = cluster_boot_median_diff(da, db, "top1_prob", rng, args.n_boot)
            p_cluster = cluster_perm_p(da, db, "top1_prob", perm_rng, args.n_perm)
            pvals.append(p_cluster)
            recs.append(dict(speaker=sp, a=a, b=b, n_a=len(da), n_b=len(db),
                             n_conv_a=da["conv"].nunique(), n_conv_b=db["conv"].nunique(),
                             median_a=da["top1_prob"].median(), median_b=db["top1_prob"].median(),
                             median_diff=da["top1_prob"].median() - db["top1_prob"].median(),
                             ci_lo=lo, ci_hi=hi,
                             p_cluster_perm=p_cluster, p_item_level=p_item,
                             cliffs_delta=cliffs_delta(da["top1_prob"], db["top1_prob"])))
        for r, ph in zip(recs, holm(pvals)):
            r["p_cluster_holm"] = ph
            test_rows.append(r)
            print(f"    {SHORT[r['a']]:12s} vs {SHORT[r['b']]:12s} "
                  f"median {r['median_a']:.4f} vs {r['median_b']:.4f} "
                  f"(diff {r['median_diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}])  "
                  f"p_cluster_Holm={ph:.3g}  (item-level p={r['p_item_level']:.2g})  "
                  f"delta={r['cliffs_delta']:+.3f}")
    tests = pd.DataFrame(test_rows)
    tests.to_csv(TABLES / "oncoco_confidence_tests.csv", index=False)

    # --- 3. length control ---------------------------------------------------
    print("\n3. Length control: the same contrast within token-length deciles")
    d["len_decile"] = pd.qcut(d["n_tokens"], 10, labels=False, duplicates="drop")
    strata = (d.groupby(["speaker_type", "len_decile", "source"])
                .agg(n=("top1_prob", "size"), top1_median=("top1_prob", "median"),
                     entropy_median=("entropy_bits", "median"), tokens=("n_tokens", "median"))
                .reset_index())
    strata.to_csv(TABLES / "oncoco_confidence_length_strata.csv", index=False)
    for sp in SPEAKERS:
        w = (strata[strata["speaker_type"] == sp]
             .pivot(index="len_decile", columns="source", values="top1_median"))
        w = w.reindex(columns=[c for c in CHAT if c in w.columns])
        w.columns = [SHORT[c] for c in w.columns]
        print(f"  {sp}: median top-1 probability by length decile")
        print(w.round(4).to_string())
        signs = np.sign((w["H--LLM"] - w["HH real"]).dropna())
        if len(signs):
            print(f"    H--LLM minus HH real is positive in {int((signs > 0).sum())} of "
                  f"{len(signs)} deciles")

    # length-matched resample of each condition to HH real's length distribution
    print("\n   Length-matched resample (each condition matched to HH real's token distribution)")
    lm_rows = []
    for sp in SPEAKERS:
        ref = d[(d["speaker_type"] == sp) & (d["source"] == "HH_real_chat")]
        ref_counts = ref["len_decile"].value_counts()
        for cond in CHAT:
            s = d[(d["speaker_type"] == sp) & (d["source"] == cond)]
            picks = []
            for dec, k in ref_counts.items():
                pool = s[s["len_decile"] == dec]
                if len(pool):
                    picks.append(pool.sample(n=int(k), replace=True, random_state=args.seed))
            if picks:
                m = pd.concat(picks)
                lm_rows.append(dict(speaker=sp, condition=cond, n=len(m),
                                    top1_median=m["top1_prob"].median(),
                                    entropy_median=m["entropy_bits"].median()))
    lm = pd.DataFrame(lm_rows)
    print(lm.round(4).to_string(index=False))

    # --- 4. signature labels -------------------------------------------------
    print("\n4. Confidence on the signature categories (items PREDICTED as each label)")
    sig_rows = []
    for code, name in SIGNATURE.items():
        for cond in CHAT:
            s = d[(d["pred_label"] == code) & (d["source"] == cond)]
            if not len(s):
                continue
            sig_rows.append(dict(label=code, label_text=name, condition=cond, n=len(s),
                                 top1_median=s["top1_prob"].median(),
                                 entropy_median=s["entropy_bits"].median(),
                                 share_below_90=float((s["top1_prob"] < .9).mean())))
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(TABLES / "oncoco_confidence_signature_labels.csv", index=False)
    if len(sig):
        piv = sig.pivot_table(index="label_text", columns="condition", values="top1_median")
        piv = piv.reindex(columns=[c for c in CHAT if c in piv.columns])
        piv.columns = [SHORT[c] for c in piv.columns]
        print(piv.round(4).to_string())

    # --- 5. the decisive test: RQ1 ordering on high-confidence items only -----
    print("\n5. RQ1 proximity gap restricted to high-confidence items")
    sens_rows = []
    for thr in (0.0, 0.7, 0.9, 0.99):
        f = d[d["top1_prob"] >= thr]
        for sp in SPEAKERS:
            s = f[f["speaker_type"] == sp]
            parts = {c: s[s["source"] == c] for c in CHAT}
            if min(len(v) for v in parts.values()) < 100:
                continue
            labels = sorted(s["pred_label"].dropna().unique())
            vec = {c: pooled_vec(v, labels) for c, v in parts.items()}
            d_real = js_distance(vec["HH_real_chat"], vec["H_LLM_roleplay_chat"])
            d_role = js_distance(vec["HH_real_chat"], vec["HH_roleplay_chat"])

            # one count row per conversation, so a resample is a row draw plus a column sum
            mats = {c: np.vstack([pooled_vec(g, labels) for _, g in v.groupby("conv")])
                    for c, v in parts.items()}
            deltas = np.empty(args.n_boot)
            for b in range(args.n_boot):
                vv = {}
                for c in CHAT:
                    n = mats[c].shape[0]
                    vv[c] = mats[c][rng.integers(0, n, n)].sum(axis=0)
                deltas[b] = (js_distance(vv["HH_real_chat"], vv["H_LLM_roleplay_chat"])
                             - js_distance(vv["HH_real_chat"], vv["HH_roleplay_chat"]))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            # The "all items" row is the same quantity as the SaT row of the main paper's unit
            # ablation. Two independent bootstraps of it disagree in the third decimal, so the
            # baseline interval is sourced from that run rather than redrawn here. The draw above
            # still runs, which keeps the RNG stream identical for the thresholded rows.
            if thr == 0:
                ua = pd.read_csv(TABLES / "oncoco_unit_ablation.csv")
                sp_disp = "Counselor" if sp == "Counsellor" else sp
                row = ua[(ua["unit"] == "SaT spans") & (ua["speaker"] == sp_disp)]
                if len(row) != 1:
                    raise SystemExit(f"unit ablation CSV: expected one 'SaT spans'/{sp_disp} row, found {len(row)}")
                lo, hi = float(row["ci_lo"].iloc[0]), float(row["ci_hi"].iloc[0])
            kept = len(s) / max(len(d[(d["speaker_type"] == sp)]), 1)
            sens_rows.append(dict(threshold=thr, speaker=sp, retained_share=kept,
                                  n_items=len(s), n_labels=len(labels),
                                  jsd_real_vs_hllm=d_real, jsd_real_vs_roleplay=d_role,
                                  delta=d_real - d_role, ci_lo=lo, ci_hi=hi,
                                  excludes_zero=bool(lo > 0)))
            print(f"  top1 >= {thr:.2f}  {sp:11s} retained {kept:5.1%} "
                  f"({len(s):6d} items)  delta'={d_real - d_role:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"{'  *' if lo > 0 else '   (CI includes 0)'}")
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(TABLES / "oncoco_rq1_highconf_sensitivity.csv", index=False)

    lines = [r"\begin{tabular}{lcrrrc}", r"\toprule",
             r"Confidence threshold & Role & Items retained & JSD real--LLM & JSD real--roleplay "
             r"& $\Delta$ [95\% CI] \\", r"\midrule"]
    for _, r in sens.iterrows():
        thr = "all items" if r["threshold"] == 0 else f"$\\geq$ {r['threshold']:.2f}"
        star = r"$^{*}$" if r["excludes_zero"] else ""
        # NB: escape the percent sign, otherwise it comments out the rest of the LaTeX row.
        lines.append(f"{thr} & {r['speaker'].replace('Counsellor', 'Counselor')} & "
                     f"{r['retained_share'] * 100:.0f}\\% & {r['jsd_real_vs_hllm']:.3f} & "
                     f"{r['jsd_real_vs_roleplay']:.3f} & {r['delta']:+.3f}{star} "
                     f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "oncoco_rq1_highconf_sensitivity.tex").write_text("\n".join(lines) + "\n",
                                                                encoding="utf-8")

    # --- ECDF figure ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"HH_roleplay_chat": "#0072B2", "HH_real_chat": "#009E73",
                  "H_LLM_roleplay_chat": "#D55E00"}
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
        for ax, sp in zip(axes, SPEAKERS):
            for cond in CHAT:
                v = np.sort(d[(d["speaker_type"] == sp) & (d["source"] == cond)]["entropy_bits"])
                if len(v):
                    ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=SHORT[cond].replace("--", "-"),
                            color=colors[cond], lw=1.6)
            ax.set_title("Counselor" if sp == "Counsellor" else sp, fontsize=10)
            ax.set_xlabel("prediction entropy (bits)")
            ax.set_xscale("symlog", linthresh=0.01)
            ax.grid(alpha=.3, lw=.5)
        axes[0].set_ylabel("cumulative share of items")
        axes[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGS / "confidence_ecdf.pdf")
        fig.savefig(FIGS / "confidence_ecdf.png", dpi=200)
        print(f"\nWrote {FIGS / 'confidence_ecdf.pdf'}")
    except Exception as exc:  # plotting must never take the analysis down
        print(f"\n(figure skipped: {exc})")

    print(f"Wrote {TABLES / 'oncoco_confidence_by_condition.csv'}, "
          f"{TABLES / 'oncoco_rq1_highconf_sensitivity.csv'} and companions")


if __name__ == "__main__":
    main()
