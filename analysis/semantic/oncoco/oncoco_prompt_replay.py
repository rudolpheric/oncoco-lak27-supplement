#!/usr/bin/env python3
"""Does a prompt written against the measured signatures move the simulated client toward real clients?

Prompt-replay experiment (analysis/prompt_replay/): the client turns of the 90 GPT-OSS-120B
conversations were regenerated with the original conversation history held fixed, under
  A  the logged prompt, unchanged            (harness / sampling baseline)
  B  a generic "write like a real client" cue  (naive-prompting control)
  C  rules written against the five signatures (findings-based prompt)
This script puts every arm next to the original deployment and the two human conditions on
the paper's client-side measures (SaT spans):
  * JSD to HH real with conversation-level bootstrap CI, delta' = JSD(real, arm) - JSD(real, HH roleplay)
  * Cramer's V, and the HH-real split-half noise band
  * JSD of each arm to the original GPT-OSS deployment (is arm A inside the deployment's own band?)
  * the signature metrics of Section 5.2: problem share in decile 1, solution-space share
    (pooled, decile 10), share of conversations with a General request / a Rejection,
    negative-to-positive feedback ratio, median words per client message, share without
    final punctuation
  * compliance / echo diagnostics for the generated arms

Input : data/processed/combined/normalized/oncoco_classification_all.json     (paper corpus)
        data/processed/prompt_replay/oncoco_classification_replay.json         (replay arms)
        data/processed/prompt_replay/replay_chat_turn_status.csv               (generated / missing)
        data/processed/prompt_replay/replay_requests.jsonl                     (example lines for echo check)
Output: results/tables/oncoco_prompt_replay.csv / .tex
        results/tables/oncoco_prompt_replay_labels.csv   (per-label client shares per group)
        results/figures/oncoco/prompt_replay_deciles.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from oncoco_noise_band_analysis import (  # noqa: E402
    count_matrix, cramers_v, js_distance, load_rows, span_labels_by_conversation, split_half_band,
)
from oncoco_prompt_meta import client_word_count, has_final_punct  # noqa: E402
from oncoco_signature_shares import NEG_FEEDBACK, POS_FEEDBACK  # noqa: E402

NORM = ROOT / "data/processed/combined/normalized"
REPLAY = ROOT / "data/processed/prompt_replay"
TABLES = ROOT / "results/tables"
FIGS = ROOT / "results/figures/oncoco"
LABEL_MAP = json.load(open(HERE / "label_text_map.json"))
REF, RP, LLM, MODEL = "HH_real_chat", "HH_roleplay_chat", "H_LLM_roleplay_chat", "GPT_OSS_120B"
ARM_NAMES = {"O": "O: original text, current pipeline", "A": "A: replay of logged prompt",
             "B": "B: generic realism prompt", "C": "C: findings-based prompt",
             "D": "D: findings-based prompt, dosed",
             "E": "E: as D, positive feedback allowed", "F": "F: phase-aware prompt (HH-real HMM)",
             "G": "G: phase-aware, F2 (no examples, no commitments)"}
# S1 setting-bound exclusion set of the supplement (client side), for a robustness column
S1_CLIENT = {"CL-IF-AO-*-Obj-*", "CL-IF-AO-*-Ext-*", "CL-FB-*-*-*-*", "CL-O-*-*-UCO-*"}
# Labels the findings-based prompt instructs directly and overshoots (post-hoc diagnostic column):
# General request, and the two positive-feedback labels it suppresses.
INSTRUCTED = {"CL-IF-ACP-*-Req-*", "CL-IF-HP-*-PosF-*", "CL-IF-HP-*-PosFR-*"}
META_RE = re.compile(r"\b(als KI|Sprachmodell|KI-Modell|ich bin ein(e)? (KI|Assistent)|as an AI|language model)\b", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default=str(NORM / "oncoco_classification_all.json"))
    p.add_argument("--replay", nargs="+", default=[str(REPLAY / "oncoco_classification_replay.json")],
                   help="one or more classification JSONs of replay arms")
    p.add_argument("--turn-status", nargs="+", default=[str(REPLAY / "replay_chat_turn_status.csv")],
                   help="turn-status CSVs written by replay_to_chat_csv.py (any number)")
    p.add_argument("--requests", default=str(REPLAY / "replay_requests.jsonl"))
    p.add_argument("--n-boot", type=int, default=20000)
    p.add_argument("--n-splits", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", default=str(TABLES / "oncoco_prompt_replay"))
    p.add_argument("--fig", default=str(FIGS / "prompt_replay_deciles.pdf"))
    p.add_argument("--unit", choices=["span", "message"], default="span",
                   help="label unit; 'message' uses the message-level label and is immune to segmentation")
    return p.parse_args()


def as_message_unit(rows: list[dict]) -> list[dict]:
    """Replace the span list of every message by one pseudo-span carrying the message label."""
    out = []
    for r in rows:
        lab = (r.get("message_classification") or {}).get("predicted_label")
        r2 = dict(r)
        text = r.get("msg_content") or ""
        r2["sentence_classification"] = ([dict(sentence_index=0, start_offset=0, end_offset=len(text),
                                               text=text, predicted_label=lab)] if lab else [])
        out.append(r2)
    return out


def bucket(code: str) -> str | None:
    """Same buckets as oncoco_behavioral_evolution.py."""
    t = LABEL_MAP.get(code, code).lower()
    if "problem" in t:
        return "Problem"
    if any(k in t for k in ["recommendation", "resource activation", "implementation"]):
        return "Solution"
    if "general request" in t:
        return "Request"
    return None


def conv_key(r: dict) -> str:
    return f"{r.get('source_file', '')}::{r.get('id')}"


def drop_ungenerated(rows: list[dict], status_paths: list[Path]) -> list[dict]:
    """Client messages the generator did not produce keep the original text in the CSV;
    remove them from the arm so every client span of an arm is generated text."""
    bad = set()
    for status_path in status_paths:
        if not status_path.exists():
            continue
        with status_path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["status"] != "generated":
                    bad.add((f"H_LLM_replay_{r['arm']}_chat", str(r["conv_id"]), int(r["message_number"])))
    kept = [r for r in rows if not (r.get("speaker_type") == "Client" and
                                    (r.get("source"), str(r.get("id")), int(r.get("msg_message_number") or 0)) in bad)]
    print(f"dropped {len(rows) - len(kept)} ungenerated client messages from the replay arms")
    return kept


def client_rows(rows: list[dict], source: str, model: str | None = None) -> list[dict]:
    return [r for r in rows if r.get("source") == source and r.get("speaker_type") == "Client"
            and (model is None or r.get("model") == model)]


def deciles(crows: list[dict]) -> pd.DataFrame:
    """Span-level deciles per conversation, as in oncoco_sentence_analysis.add_deciles."""
    recs = []
    for r in crows:
        for s in r.get("sentence_classification") or []:
            if s.get("predicted_label"):
                recs.append((conv_key(r), int(r.get("msg_message_number") or 0),
                             s.get("sentence_index", s.get("start_offset", 0)), s["predicted_label"]))
    df = pd.DataFrame(recs, columns=["conv", "msg", "sent", "label"])
    if df.empty:
        return df
    df = df.sort_values(["conv", "msg", "sent"])
    df["order"] = df.groupby("conv").cumcount()
    n = df.groupby("conv")["order"].transform("max") + 1
    df["rel"] = np.where(n <= 1, 0.0, df["order"] / (n - 1).clip(lower=1))
    df["decile"] = (df["rel"] * 10).astype(int).clip(0, 9)
    df["bucket"] = df["label"].map(bucket)
    return df


def signature_metrics(crows: list[dict]) -> dict:
    d = deciles(crows)
    out = {}
    if d.empty:
        return out
    out["problem_d1"] = float((d[d.decile == 0].bucket == "Problem").mean())
    out["problem_pooled"] = float((d.bucket == "Problem").mean())
    out["solution_pooled"] = float((d.bucket == "Solution").mean())
    out["solution_d10"] = float((d[d.decile == 9].bucket == "Solution").mean())
    out["request_pooled"] = float((d.bucket == "Request").mean())
    per_conv = defaultdict(lambda: dict(request=False, rejection=False, qmark=False))
    neg = pos = 0
    words, punct = [], []
    for r in crows:
        k = conv_key(r)
        per_conv[k]
        text = r.get("msg_content") or ""
        if "?" in text:
            per_conv[k]["qmark"] = True
        words.append(client_word_count(text))
        punct.append(has_final_punct(text))
        for s in r.get("sentence_classification") or []:
            lab = s.get("predicted_label") or ""
            t = LABEL_MAP.get(lab, lab).lower()
            if "general request" in t:
                per_conv[k]["request"] = True
            if "rejection" in t:
                per_conv[k]["rejection"] = True
            neg += lab in NEG_FEEDBACK
            pos += lab in POS_FEEDBACK
    n = len(per_conv)
    out["share_conv_request"] = sum(v["request"] for v in per_conv.values()) / n
    out["share_conv_rejection"] = sum(v["rejection"] for v in per_conv.values()) / n
    out["share_conv_qmark"] = sum(v["qmark"] for v in per_conv.values()) / n
    out["neg_pos_ratio"] = neg / pos if pos else float("nan")
    out["neg_feedback"], out["pos_feedback"] = neg, pos
    out["median_words"] = float(np.median(words))
    out["share_no_final_punct"] = 1.0 - float(np.mean(punct))
    out["n_messages"] = len(words)
    return out


def segmentation_profile(crows: list[dict]) -> dict:
    """How the segmenter treats a group's client text: spans per message, span length, share of
    one-word spans, spans starting inside a word, share labelled Other statements."""
    spans = [(r.get("msg_content") or "", s) for r in crows for s in r.get("sentence_classification") or []]
    if not spans or not crows:
        return {}
    words = np.array([len((s.get("text") or "").split()) for _, s in spans])
    mid = [t[s["start_offset"] - 1].isalpha() if s.get("start_offset", 0) > 0 and s["start_offset"] <= len(t) else False
           for t, s in spans]
    return dict(
        spans_per_msg=len(spans) / len(crows),
        words_per_span=float(words.mean()),
        share_span_le1_word=float((words <= 1).mean()),
        share_span_midword_start=float(np.mean(mid)),
        share_other_statements=float(np.mean([s.get("predicted_label") == "CL-O-*-*-O-*" for _, s in spans])),
    )


def compliance(crows: list[dict], examples: list[str]) -> dict:
    texts = [(r.get("msg_content") or "").strip() for r in crows]
    if not texts:
        return {}
    ex = {e.lower() for e in examples}
    return dict(
        share_over_20_words=float(np.mean([client_word_count(t) > 20 for t in texts])),
        share_multi_paragraph=float(np.mean(["\n" in t for t in texts])),
        share_meta=float(np.mean([bool(META_RE.search(t)) for t in texts])),
        share_example_echo=float(np.mean([t.lower().rstrip(".!?") in ex for t in texts])),
    )


def example_lines(requests_path: Path) -> list[str]:
    """The form examples in the arm-C prompts, for the echo check."""
    if not requests_path.exists():
        return []
    ex = set()
    with requests_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["arm"] != "C":
                continue
            block = r["prompt"].split("Beispiele nur für die Form")[-1]
            for l in block.splitlines():
                if l.startswith("- "):
                    ex.add(l[2:].strip())
            if len(ex) >= 7:
                break
    return sorted(ex)


def bootstrap(mats: dict, g: str, n_boot: int, seed: int) -> dict:
    """JSD(real, g) and delta' with conversation-level resampling, same stream for every group."""
    rng = np.random.default_rng(seed)
    ref, rp, gg = mats[REF], mats[RP], mats[g]

    def rs(m):
        return m[rng.integers(0, m.shape[0], m.shape[0])].sum(axis=0)

    bj, bd = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        r, p, x = rs(ref), rs(rp), rs(gg)
        bj[b] = js_distance(r, x)
        bd[b] = bj[b] - js_distance(r, p)
    jsd = js_distance(ref.sum(axis=0), gg.sum(axis=0))
    return dict(jsd_real=jsd, jsd_real_lo=float(np.percentile(bj, 2.5)), jsd_real_hi=float(np.percentile(bj, 97.5)),
                delta=jsd - js_distance(ref.sum(axis=0), rp.sum(axis=0)),
                delta_lo=float(np.percentile(bd, 2.5)), delta_hi=float(np.percentile(bd, 97.5)))


def main() -> None:
    args = parse_args()
    master = load_rows(Path(args.master))
    replay = drop_ungenerated([r for p in args.replay for r in load_rows(Path(p))],
                              [Path(p) for p in args.turn_status])
    arms = sorted({r["source"] for r in replay if str(r.get("source", "")).startswith("H_LLM_replay_")})
    rows = master + replay
    if args.unit == "message":
        rows = as_message_unit(rows)
        args.out_prefix = args.out_prefix + "_message"
        args.fig = str(Path(args.fig).with_name(Path(args.fig).stem + "_message" + Path(args.fig).suffix))

    groups = [(REF, REF, None, "HH real"), (RP, RP, None, "HH roleplay"),
              ("orig", LLM, MODEL, "GPT-OSS-120B deployment")]
    for a in arms:
        code = a.replace("H_LLM_replay_", "").replace("_chat", "")
        groups.append((a, a, None, ARM_NAMES.get(code, a)))

    labels_by_group = {g: span_labels_by_conversation(rows, src, "Client", model) for g, src, model, _ in groups}
    all_labels = sorted({l for d in labels_by_group.values() for labs in d.values() for l in labs})
    idx = {l: i for i, l in enumerate(all_labels)}
    mats = {g: count_matrix(labels_by_group[g], idx)[1] for g in labels_by_group}
    s1_cols = np.array([l not in S1_CLIENT for l in all_labels])
    ex_cols = np.array([l not in INSTRUCTED for l in all_labels])

    rng = np.random.default_rng(args.seed)
    band_ref = split_half_band(mats[REF], args.n_splits, rng)
    band_orig = split_half_band(mats["orig"], args.n_splits, rng)
    examples = example_lines(Path(args.requests))

    records, label_rows = [], []
    for g, src, model, name in groups:
        crows = client_rows(rows, src, model)
        rec = dict(group=g, name=name, n_conv=mats[g].shape[0], n_spans=int(mats[g].sum()))
        if g != REF:
            rec.update(bootstrap(mats, g, args.n_boot, args.seed))
            rec["jsd_real_s1"] = js_distance(mats[REF].sum(axis=0)[s1_cols], mats[g].sum(axis=0)[s1_cols])
            rec["jsd_real_ex_instructed"] = js_distance(mats[REF].sum(axis=0)[ex_cols], mats[g].sum(axis=0)[ex_cols])
            rec["cramers_v"] = cramers_v(mats[REF].sum(axis=0), mats[g].sum(axis=0))
            rec["ref_band_mean"], rec["ref_band_p95"] = float(band_ref.mean()), float(np.percentile(band_ref, 95))
        if g not in (REF, RP, "orig"):
            rec["jsd_to_orig"] = js_distance(mats["orig"].sum(axis=0), mats[g].sum(axis=0))
            rec["orig_band_p95"] = float(np.percentile(band_orig, 95))
            rec["within_orig_band"] = bool(rec["jsd_to_orig"] <= rec["orig_band_p95"])
            rec.update(compliance(crows, examples))
        rec.update(signature_metrics(crows))
        rec.update(segmentation_profile(crows))
        records.append(rec)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})
        share = mats[g].sum(axis=0) / max(mats[g].sum(), 1)
        for l, s in zip(all_labels, share):
            label_rows.append(dict(group=g, label=l, label_text=LABEL_MAP.get(l, l), share=float(s)))

    df = pd.DataFrame(records)
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    pd.DataFrame(label_rows).to_csv(out.with_name(out.name + "_labels.csv"), index=False)
    write_label_shifts(pd.DataFrame(label_rows), [g for g, *_ in groups], out.with_name(out.name + "_label_shifts.csv"))
    write_tex(df, out.with_suffix(".tex"))
    plot_deciles(rows, groups, Path(args.fig))
    print(f"wrote {out.with_suffix('.csv')}, {out.with_suffix('.tex')}, {args.fig}")


def write_label_shifts(labels: pd.DataFrame, groups: list[str], path: Path) -> None:
    """Per-label client shares side by side, plus each label's contribution (bits) to the
    Jensen-Shannon divergence from HH real, so the reader sees which labels carry an arm's distance."""
    wide = labels.pivot(index=["label", "label_text"], columns="group", values="share").fillna(0.0)
    wide = wide.reindex(columns=[g for g in groups if g in wide.columns])
    ref = wide[REF].values
    for g in wide.columns:
        if g == REF:
            continue
        q = wide[g].values
        m = (ref + q) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            c = 0.5 * (np.where(ref > 0, ref * np.log2(ref / m), 0) + np.where(q > 0, q * np.log2(q / m), 0))
        wide[f"jsd_bits_{g}"] = c
    order = [c for c in wide.columns if c.startswith("jsd_bits_")]
    wide = wide.sort_values(order[-1], ascending=False) if order else wide
    wide.reset_index().to_csv(path, index=False)


def _f(v, nd=3):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def write_tex(df: pd.DataFrame, path: Path) -> None:
    lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             r"Group & Conv. & JSD real [95\% CI] & $\delta'$ [95\% CI] & $V$ & Problem d1 & Solution & Req.\ conv. & Rej.\ conv. & Neg:Pos & Med.\ words \\",
             r"\midrule"]
    for _, r in df.iterrows():
        jsd = "" if r["group"] == REF else f"{_f(r['jsd_real'])} [{_f(r['jsd_real_lo'])}, {_f(r['jsd_real_hi'])}]"
        dl = "" if r["group"] == REF else f"{r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
        lines.append(f"{r['name']} & {int(r['n_conv'])} & {jsd} & {dl} & {_f(r.get('cramers_v'), 2)} & "
                     f"{_f(r.get('problem_d1'))} & {_f(r.get('solution_pooled'))} & {_f(r.get('share_conv_request'), 2)} & "
                     f"{_f(r.get('share_conv_rejection'), 2)} & {_f(r.get('neg_pos_ratio'), 2)} & {_f(r.get('median_words'), 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_deciles(rows: list[dict], groups: list[tuple], fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    styles = {REF: dict(color="#009E73", ls="--"), RP: dict(color="#0072B2", ls=":"), "orig": dict(color="#D55E00", ls="-")}
    palette = ["#56B4E9", "#E69F00", "#CC79A7", "#000000", "#F0E442", "#8B0000", "#009E73", "#999999"]  # A..G, O
    for i, (g, src, model, name) in enumerate(groups):
        d = deciles(client_rows(rows, src, model))
        if d.empty:
            continue
        st = styles.get(g, dict(color=palette[(i - 3) % len(palette)], ls="-"))
        for ax, bk in zip(axes, ["Problem", "Solution", "Request"]):
            ys = [float((d[d.decile == k].bucket == bk).mean()) for k in range(10)]
            ax.plot(range(1, 11), ys, marker="o", ms=3, lw=1.6, label=name, **st)
    for ax, title in zip(axes, ["Problem disclosure", "Solution-space engagement", "Reciprocal requests"]):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Conversation decile")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Share of client spans")
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
