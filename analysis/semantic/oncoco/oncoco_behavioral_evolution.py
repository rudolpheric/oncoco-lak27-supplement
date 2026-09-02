#!/usr/bin/env python3
"""Per-model evolution of the three behavioral signatures (Client side), vs human baselines.
Chronological models Mixtral -> LLama3.3-70B -> GPT-OSS-120B. Writes a 1x3 panel PNG."""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
lm = json.load(open(ROOT / "analysis/semantic/oncoco/label_text_map.json"))
rows = json.loads((ROOT / "data/processed/combined/normalized/oncoco_classification_all.json").read_text(encoding="utf-8"))

def bucket(code):
    t = lm.get(code, code).lower()
    if "problem" in t: return "Problem"
    if any(k in t for k in ["recommendation", "resource activation", "implementation"]): return "Solution"
    if "general request" in t: return "Request"
    return None

recs = []
for r in rows:
    if r.get("speaker_type") != "Client": continue
    src, model = r.get("source"), r.get("model") or ""
    grp = model if src == "H_LLM_roleplay_chat" else (src if src in ("HH_real_chat", "HH_roleplay_chat") else None)
    if grp is None: continue
    for s in r.get("sentence_classification") or []:
        recs.append((grp, bucket(s.get("predicted_label"))))
df = pd.DataFrame(recs, columns=["grp", "bucket"])

MODELS = [("Mixtral", "Mixtral 8x7B\n2024-10"), ("LLama3_3_70B", "Llama 3.3 70B\n2025-07"), ("GPT_OSS_120B", "GPT-OSS-120B\n2025-12")]
BUCKETS = [("Problem", "Problem disclosure"), ("Solution", "Solution-space engagement"), ("Request", "Reciprocal requests")]
def share(grp, b): sub = df[df.grp == grp]; return (sub.bucket == b).mean() if len(sub) else 0.0

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
xs = list(range(len(MODELS)))
for ax, (bk, title) in zip(axes, BUCKETS):
    ax.plot(xs, [share(m, bk) for m, _ in MODELS], marker="o", ms=7, lw=2, color="#d62728", label="H–LLM (by model)")
    ax.axhline(share("HH_real_chat", bk), ls="--", color="#2ca02c", lw=1.6, label="HH real")
    ax.axhline(share("HH_roleplay_chat", bk), ls=":", color="#1f77b4", lw=1.6, label="HH roleplay")
    ax.set_xticks(xs); ax.set_xticklabels([lbl for _, lbl in MODELS], fontsize=9)
    ax.set_title(title, fontsize=11); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
axes[0].set_ylabel("Share of client spans")
axes[0].legend(fontsize=8, loc="best")
fig.suptitle("Behavioral signatures across model generations (Client)", fontsize=12)
fig.tight_layout()
out = ROOT / "results/figures/oncoco/behavioral_signatures_evolution.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
for m, _ in MODELS:
    print(f"  {m:<14} " + "  ".join(f"{b}={share(m,b):.3f}" for b, _ in BUCKETS))
for h in ("HH_roleplay_chat", "HH_real_chat"):
    print(f"  {h:<14} " + "  ".join(f"{b}={share(h,b):.3f}" for b, _ in BUCKETS))
