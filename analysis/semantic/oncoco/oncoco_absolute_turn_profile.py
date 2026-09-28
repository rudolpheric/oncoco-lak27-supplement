#!/usr/bin/env python3
"""Client-side signature profiles by absolute turn, aligned at the start and at the end.

The paper's temporal claims use relative deciles of the client's spans. The conditions differ
in length (median 55 messages in HH real, 35 in HH roleplay, 31 in H-LLM), so a decile covers
more turns in a real conversation than in a simulated one. This script recomputes the three
temporal signatures on absolute client turns: early problem disclosure (start-aligned),
solution-space engagement toward the end (end-aligned), and disagreement in every bin.

A client turn is a maximal run of consecutive client messages, because the HH real export
merges consecutive lines of one speaker while the other conditions keep them apart.
Shares are pooled over the client spans in a bin, intervals resample conversations.

Outputs results/tables/oncoco_absolute_turn_profile.{csv,tex}.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NORM = ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"
OUT = ROOT / "results/tables/oncoco_absolute_turn_profile"
CONDS = [("HH_real_chat", "HH real"), ("HH_roleplay_chat", "HH roleplay"), ("H_LLM_roleplay_chat", "H--LLM")]
BUCKETS = {
    "problem": {"CL-IF-ACP-*-PS-*", "CL-IF-ACP-*-PD-*"},
    "solution": {"CL-IF-HP-*-NegFR-*", "CL-IF-HP-*-PosFR-*", "CL-IF-HP-*-RepRA-*",
                 "CL-IF-RA-*-RF-*", "CL-IF-RA-*-RP-*"},
    "rejection": {"CL-IF-ACP-*-Rej-*"},
}
START_BINS = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 10), (11, 15), (16, 20), (21, 10**6)]
END_BINS = [(1, 1), (2, 3), (4, 6), (7, 10)]
# pooled bins quoted in the text, written to the csv only
SUMMARY_BINS = {"start": [(1, 3)], "end": [(1, 3)]}


def bin_label(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10**6 else f"{lo}--{hi}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = json.loads(NORM.read_text(encoding="utf-8"))

    msgs = collections.defaultdict(list)          # (cond, conv) -> [(msg_no, speaker, labels)]
    for r in rows:
        if r.get("source") not in dict(CONDS):
            continue
        key = (r["source"], f'{r.get("source_file")}::{r.get("id")}')
        labels = [s["predicted_label"] for s in r.get("sentence_classification") or [] if s.get("predicted_label")]
        msgs[key].append((int(float(r.get("msg_message_number") or 0)), r.get("speaker_type"), labels))

    # per conversation: list of client turns, each a list of span labels
    turns = collections.defaultdict(dict)         # cond -> conv -> [labels per client turn]
    n_msgs = collections.defaultdict(list)
    for (cond, conv), ms in msgs.items():
        ms.sort(key=lambda m: m[0])
        n_msgs[cond].append(len(ms))
        out, prev = [], None
        for _, spk, labels in ms:
            if spk == "Client":
                if prev != "Client":
                    out.append([])
                out[-1].extend(labels)
            prev = spk
        out = [t for t in out if t]
        if out:
            turns[cond][conv] = out

    def counts(conv_turns, align, lo, hi, bucket):
        """(bucket spans, all client spans) of one conversation inside the bin."""
        n = len(conv_turns)
        hit = tot = 0
        for i, labels in enumerate(conv_turns):
            pos = i + 1 if align == "start" else n - i
            if lo <= pos <= hi:
                tot += len(labels)
                hit += sum(l in BUCKETS[bucket] for l in labels)
        return hit, tot

    rng = np.random.default_rng(args.seed)
    out = []
    for align, bins in (("start", START_BINS + SUMMARY_BINS["start"]), ("end", END_BINS + SUMMARY_BINS["end"])):
        for lo, hi in bins:
            for bucket in BUCKETS:
                for cond, cname in CONDS:
                    convs = sorted(turns[cond])
                    c = np.array([counts(turns[cond][k], align, lo, hi, bucket) for k in convs], float)
                    reach = int((c[:, 1] > 0).sum())
                    if c[:, 1].sum() == 0:
                        continue
                    share = c[:, 0].sum() / c[:, 1].sum()
                    idx = rng.integers(0, len(c), (args.B, len(c)))
                    num, den = c[idx, 0].sum(1), c[idx, 1].sum(1)
                    boot = num[den > 0] / den[den > 0]
                    lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
                    summary = (lo, hi) in SUMMARY_BINS[align]
                    out.append(dict(align=align + ("-pooled" if summary else ""), bin=bin_label(lo, hi), bucket=bucket, condition=cname,
                                    conversations=reach, spans=int(c[:, 1].sum()), share=share,
                                    ci_lo=lo_ci, ci_hi=hi_ci))
    for cond, cname in CONDS:
        nt = [len(v) for v in turns[cond].values()]
        print(f"{cname:12s} conversations {len(nt)}, client turns median {np.median(nt):.0f} "
              f"(IQR {np.percentile(nt, 25):.0f}-{np.percentile(nt, 75):.0f}), messages median {np.median(n_msgs[cond]):.0f}")

    with open(OUT.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    # tex: one block per alignment, rows = bins, columns = condition x bucket
    names = [c for _, c in CONDS]
    lines = [r"\begin{tabular}{ll" + "r" * 9 + "}", r"\toprule",
             r" & & \multicolumn{3}{c}{Problem disclosure} & \multicolumn{3}{c}{Solution space} & \multicolumn{3}{c}{Rejection} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             r"Aligned at & Client turn & " + " & ".join(names * 3) + r" \\", r"\midrule"]
    get = {(o["align"], o["bin"], o["bucket"], o["condition"]): o for o in out}
    for align, bins in (("start", START_BINS), ("end", END_BINS)):
        for j, (lo, hi) in enumerate(bins):
            b = bin_label(lo, hi)
            cells = []
            for bucket in BUCKETS:
                for n in names:
                    o = get.get((align, b, bucket, n))
                    cells.append(f"{o['share']:.3f}" if o else "")
            first = ("Start" if align == "start" else "End") if j == 0 else ""
            turn = b if align == "start" else ("last" if b == "1" else f"{b} from end")
            lines.append(f"{first} & {turn} & " + " & ".join(cells) + r" \\")
        if align == "start":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT.with_suffix(".csv"), OUT.with_suffix(".tex"))


if __name__ == "__main__":
    main()
