#!/usr/bin/env python3
"""Does the prompt configuration, not only the model, move the distance to real counseling?

The H-LLM conversations were generated under two template families (plain text vs. Llama 3
chat template with a structured answer-format block) and under personas that do or do not
carry a brevity cue in their profile (see oncoco_prompt_meta.py). This script reports, for
each prompt configuration, the client-side distance to HH real and the proximity gap
    delta' = JSD(HH real, group) - JSD(HH real, HH roleplay)
with conversation-level bootstrap intervals, on the paper's primary unit (SaT spans).

Confound warning, stated in the paper: template family is nested in model and semester
(GPT-OSS ran plain only, Mixtral logged no prompts). The one clean contrast is WITHIN Llama
3.3 70B, restricted to personas that appear under both families, which is the
"Llama, persona-matched" block.

A second block answers the volume-imbalance question: H-LLM has 414 conversations against
68 HH roleplay and 53 HH real. We draw 68 H-LLM conversations without replacement 1,000
times and recompute delta' on each draw, so the simulated client is compared at the human
roleplay's own size.

Input : data/processed/combined/normalized/oncoco_classification_all.json
        data/raw/human_llm/chats/**/*.json  (via oncoco_prompt_meta)
        results/tables/oncoco_unit_ablation.csv (baseline delta' CI, SaT row)
Output: results/tables/oncoco_prompt_variants.csv / .tex
        results/tables/oncoco_prompt_variants_sizematched.csv / .tex
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

from oncoco_noise_band_analysis import (  # noqa: E402
    count_matrix, cramers_v, js_distance, load_rows, span_labels_by_conversation,
)
from oncoco_prompt_meta import MODEL_LABEL, load_hllm_prompt_meta  # noqa: E402
from oncoco_seg_ordering_bootstrap import EXPECTED_CONVERSATIONS  # noqa: E402

NORM = ROOT / "data/processed/combined/normalized"
TABLES = ROOT / "results/tables"
REF, RP, LLM = "HH_real_chat", "HH_roleplay_chat", "H_LLM_roleplay_chat"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-json", default=str(NORM / "oncoco_classification_all.json"))
    p.add_argument("--speaker", default="Client", choices=["Client", "Counsellor"])
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--n-draw", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resample(mat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = mat.shape[0]
    return mat[rng.integers(0, n, n)].sum(axis=0)


def group_stats(name: str, block: str, ids: list[str], mats: dict, id_index: dict,
                words: pd.Series, n_boot: int, seed: int) -> dict:
    """JSD to HH real and delta' for a subset of H-LLM conversations, with bootstrap CIs."""
    rng = np.random.default_rng(seed)  # same stream per group, so groups are comparable
    rows = [id_index[i] for i in ids]
    g = mats[LLM][rows]
    pooled_g, pooled_ref, pooled_rp = g.sum(axis=0), mats[REF].sum(axis=0), mats[RP].sum(axis=0)
    jsd = js_distance(pooled_ref, pooled_g)
    jsd_rp = js_distance(pooled_ref, pooled_rp)
    boot_j, boot_d = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        r, p, gg = resample(mats[REF], rng), resample(mats[RP], rng), resample(g, rng)
        boot_j[b] = js_distance(r, gg)
        boot_d[b] = boot_j[b] - js_distance(r, p)
    return dict(
        block=block, group=name, n_conv=len(ids), n_spans=int(pooled_g.sum()),
        median_client_words=float(words.reindex(ids).median()),
        jsd_real=round(jsd, 4), jsd_lo=round(float(np.percentile(boot_j, 2.5)), 4),
        jsd_hi=round(float(np.percentile(boot_j, 97.5)), 4),
        jsd_real_vs_roleplay=round(jsd_rp, 4),
        delta_prime=round(jsd - jsd_rp, 4), dp_lo=round(float(np.percentile(boot_d, 2.5)), 4),
        dp_hi=round(float(np.percentile(boot_d, 97.5)), 4),
        frac_positive=round(float((boot_d > 0).mean()), 4),
        cramers_v=round(cramers_v(pooled_g, pooled_ref), 3), n_boot=n_boot,
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.data_json))
    per_cond = {c: span_labels_by_conversation(rows, c, args.speaker) for c in (REF, RP, LLM)}
    for c, n in EXPECTED_CONVERSATIONS.items():
        if len(per_cond[c]) != n:
            raise SystemExit(f"{c}: {len(per_cond[c])} conversations, expected {n}")
    labels = sorted({l for c in per_cond for labs in per_cond[c].values() for l in labs})
    idx = {l: i for i, l in enumerate(labels)}
    ids, mats = {}, {}
    for c in per_cond:
        ids[c], mats[c] = count_matrix(per_cond[c], idx)
    id_index = {k: i for i, k in enumerate(ids[LLM])}

    meta = load_hllm_prompt_meta().reindex(ids[LLM])
    if meta["model"].isna().any():
        missing = meta.index[meta["model"].isna()].tolist()
        raise SystemExit(f"{len(missing)} analysed H-LLM conversations have no raw metadata: {missing[:3]}")
    # median client words per conversation, from the message rows
    words = (pd.DataFrame([dict(k=f"{r.get('source_file', '')}::{r.get('id')}",
                                w=len(str(r.get("msg_content") or "").split()))
                           for r in rows if r.get("source") == LLM and r.get("speaker_type") == "Client"])
             .groupby("k")["w"].median())

    def sel(mask: pd.Series) -> list[str]:
        return meta.index[mask].tolist()

    groups = [("all H--LLM", "pooled", sel(meta["model"].notna()))]
    for fam, lab in (("plain", "plain template"), ("chat", "chat template"), ("not logged", "prompt not logged")):
        groups.append((lab, "template family", sel(meta["template_family"] == fam)))
    for flag, lab in ((True, "persona asks for brevity"), (False, "persona silent on length")):
        groups.append((lab, "persona brevity cue", sel(meta["brevity_persona"] == flag)))
    for m, ml in MODEL_LABEL.items():
        for fam in ("plain", "chat", "not logged"):
            s = sel((meta["model"] == m) & (meta["template_family"] == fam))
            if s:
                groups.append((f"{ml}, {fam}", "model x template family", s))
        for flag in (True, False):
            s = sel((meta["model"] == m) & (meta["brevity_persona"] == flag))
            if s:
                groups.append((f"{ml}, brevity {'yes' if flag else 'no'}", "model x persona brevity cue", s))
    llama = meta[meta["model"] == "LLama3_3_70B"]
    dual = sorted(p for p, g in llama.groupby("persona_name")
                  if {"plain", "chat"} <= set(g["template_family"]))
    for fam in ("plain", "chat"):
        s = sel((meta["model"] == "LLama3_3_70B") & (meta["template_family"] == fam)
                & meta["persona_name"].isin(dual))
        groups.append((f"Llama 3.3 70B, {fam}, persona-matched", "Llama, persona-matched", s))
    print(f"persona-matched Llama contrast uses {len(dual)} personas: {dual}")

    results = [group_stats(name, block, g, mats, id_index, words, args.n_boot, args.seed)
               for name, block, g in groups if len(g) >= 5]
    res = pd.DataFrame(results)
    res.to_csv(TABLES / "oncoco_prompt_variants.csv", index=False)
    print(res[["block", "group", "n_conv", "median_client_words", "jsd_real", "jsd_lo", "jsd_hi",
               "delta_prime", "dp_lo", "dp_hi", "frac_positive", "cramers_v"]].to_string(index=False))

    # persona-matched contrast: difference between the two Llama families, same resampling
    pm = {r["group"].split(", ")[1]: r for r in results if r["block"] == "Llama, persona-matched"}
    if {"plain", "chat"} <= set(pm):
        rng = np.random.default_rng(args.seed)
        gp = mats[LLM][[id_index[i] for i in groups[[g[0] for g in groups].index("Llama 3.3 70B, plain, persona-matched")][2]]]
        gc = mats[LLM][[id_index[i] for i in groups[[g[0] for g in groups].index("Llama 3.3 70B, chat, persona-matched")][2]]]
        diff = np.empty(args.n_boot)
        for b in range(args.n_boot):
            r = resample(mats[REF], rng)
            diff[b] = js_distance(r, resample(gc, rng)) - js_distance(r, resample(gp, rng))
        point = pm["chat"]["jsd_real"] - pm["plain"]["jsd_real"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"\nPersona-matched Llama contrast, JSD(real, chat) - JSD(real, plain) = {point:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]")
        pd.DataFrame([dict(contrast="chat - plain (Llama, persona-matched)", point=round(point, 4),
                           ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4), n_boot=args.n_boot)]
                     ).to_csv(TABLES / "oncoco_prompt_variants_contrast.csv", index=False)

    # ---- size-matched subsample: 68 H-LLM conversations per draw ----
    rng = np.random.default_rng(args.seed)
    n_rp = mats[RP].shape[0]
    pooled_ref, pooled_rp = mats[REF].sum(axis=0), mats[RP].sum(axis=0)
    jsd_rp = js_distance(pooled_ref, pooled_rp)
    sm_rows = []
    pools = [("all H--LLM", ids[LLM])] + [(ml, sel(meta["model"] == m)) for m, ml in MODEL_LABEL.items()]
    for name, pool in pools:
        if len(pool) < n_rp:
            continue
        rows_ = np.array([id_index[i] for i in pool])
        d = np.empty(args.n_draw)
        for b in range(args.n_draw):
            pick = rng.choice(rows_, size=n_rp, replace=False)
            d[b] = js_distance(pooled_ref, mats[LLM][pick].sum(axis=0)) - jsd_rp
        sm_rows.append(dict(pool=name, n_pool=len(pool), n_per_draw=n_rp, n_draw=args.n_draw,
                            delta_prime_mean=round(float(d.mean()), 4),
                            p2_5=round(float(np.percentile(d, 2.5)), 4),
                            p97_5=round(float(np.percentile(d, 97.5)), 4),
                            frac_positive=round(float((d > 0).mean()), 4)))
    sm = pd.DataFrame(sm_rows)
    sm.to_csv(TABLES / "oncoco_prompt_variants_sizematched.csv", index=False)
    print("\nSize-matched subsample (68 H-LLM conversations per draw, without replacement):")
    print(sm.to_string(index=False))

    # ---- supplement tables ----
    lines = [r"\begin{tabular}{llrrcc}", r"\toprule",
             r"Block & Prompt configuration & Conv. & Med. words & JSD real--LLM [95\% CI] & "
             r"$\Delta$ [95\% CI] \\", r"\midrule"]
    last = None
    for _, r in res.iterrows():
        if r["block"] == "pooled":
            lead = "all"
        else:
            lead = r["block"] if r["block"] != last else ""
        if r["block"] != last and last is not None:
            lines.append(r"\addlinespace")
        last = r["block"]
        star = r"$^{*}$" if r["dp_lo"] > 0 else ""
        lines.append(f"{lead} & {r['group']} & {int(r['n_conv'])} & {r['median_client_words']:.0f} & "
                     f"{r['jsd_real']:.3f} [{r['jsd_lo']:.3f}, {r['jsd_hi']:.3f}] & "
                     f"{r['delta_prime']:+.3f}{star} [{r['dp_lo']:.3f}, {r['dp_hi']:.3f}] " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "oncoco_prompt_variants.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{lrrcr}", r"\toprule",
             r"Pool & Conv. in pool & Mean $\Delta$ & [P2.5, P97.5] & Share $>0$ \\", r"\midrule"]
    for _, r in sm.iterrows():
        lines.append(f"{r['pool']} & {int(r['n_pool'])} & {r['delta_prime_mean']:+.3f} & "
                     f"[{r['p2_5']:.3f}, {r['p97_5']:.3f}] & {r['frac_positive'] * 100:.1f}\\% " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "oncoco_prompt_variants_sizematched.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {TABLES / 'oncoco_prompt_variants.csv'}, .tex, _sizematched.csv/.tex, _contrast.csv")


if __name__ == "__main__":
    main()
