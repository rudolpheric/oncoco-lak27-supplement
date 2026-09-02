#!/usr/bin/env python3
"""Temporal 'behavioral signatures' figure for the three chat conditions (Client side):
problem disclosure, solution-space engagement, and reciprocal requests by decile.
Reads the OnCoCo temporal distribution; writes a 1x3 panel PNG."""
import csv, json, collections
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
lm = json.load(open(ROOT / "analysis/semantic/oncoco/label_text_map.json"))
rows = list(csv.DictReader(open(ROOT / "results/tables/oncoco_temporal_distribution.csv")))

CONDS = [("HH_roleplay_chat", "HH roleplay", "#1f77b4"),
         ("HH_real_chat", "HH real", "#2ca02c"),
         ("H_LLM_roleplay_chat", "H–LLM", "#d62728")]
BUCKETS = ["Problem disclosure", "Solution-space engagement", "Reciprocal requests"]

def bucket(code):
    t = lm.get(code, code).lower()
    if "problem" in t: return "Problem disclosure"
    if any(k in t for k in ["recommendation", "resource activation", "implementation"]):
        return "Solution-space engagement"
    if "general request" in t: return "Reciprocal requests"
    return None

agg = collections.defaultdict(lambda: collections.defaultdict(float))  # (cond,decile)->bucket->sum
for r in rows:
    if r["speaker_type"] != "Client": continue
    b = bucket(r["label"])
    if b: agg[(r["condition"], int(r["decile"]))][b] += float(r["proportion"])

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
x = list(range(10))
for ax, bkt in zip(axes, BUCKETS):
    for cond, lab, col in CONDS:
        ax.plot([d * 10 + 5 for d in x], [agg[(cond, d)].get(bkt, 0) for d in x],
                marker="o", ms=3, lw=1.8, color=col, label=lab)
    ax.set_title(bkt, fontsize=11)
    ax.set_xlabel("Conversation progress (%)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Share of client spans")
axes[0].legend(fontsize=9, loc="upper left")
fig.tight_layout()
out = ROOT / "results/figures/oncoco/behavioral_signatures_temporal.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
