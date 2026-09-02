#!/usr/bin/env python3
"""Publication-ready figures for the OnCoCo paper.
Unified Okabe-Ito palette (colorblind-safe), clean labels/names, vector PDF + PNG.
Regenerates every figure used in the paper. Run from repo root:
    .venv/bin/python analysis/semantic/oncoco/pub_figures.py
"""
import csv, json, collections
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "results/figures/oncoco"
LM = json.load(open(ROOT / "analysis/semantic/oncoco/label_text_map.json"))

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12.5, "axes.titlesize": 13, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
})

# ---- Okabe-Ito palette ----
COND = {"HH_roleplay_chat": "#0072B2", "HH_real_chat": "#009E73", "H_LLM_roleplay_chat": "#D55E00",
        "HH_roleplay_mail": "#56B4E9", "HH_real_mail": "#CC79A7", "HH_CAIA_mail": "#E69F00"}
CLABEL = {"HH_roleplay_chat": "HH roleplay", "HH_real_chat": "HH real", "H_LLM_roleplay_chat": "H–LLM",
          "HH_roleplay_mail": "HH roleplay (mail)", "HH_real_mail": "HH real (mail)", "HH_CAIA_mail": "HH+assistant (mail)"}
SPKC = {"Client": "#0072B2", "Counsellor": "#E69F00"}
MODEL = {"GPT_OSS_120B": "GPT-OSS-120B", "LLama3_3_70B": "Llama 3.3 70B", "Mixtral": "Mixtral 8x7B"}
MORDER = ["Mixtral", "LLama3_3_70B", "GPT_OSS_120B"]
MSEEN = {"Mixtral": "2024-10", "LLama3_3_70B": "2025-07", "GPT_OSS_120B": "2025-12"}


def rows(p):
    return list(csv.DictReader(open(ROOT / "results/tables" / p)))

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig); print("  wrote", name)

def bucket(code):
    t = LM.get(code, code).lower()
    if "problem" in t: return "Problem"
    if any(k in t for k in ("recommendation", "resource activation", "implementation")): return "Solution"
    if "general request" in t: return "Request"
    return None

# ---------------------------------------------------------------- 1. triad
def triad():
    d = {(r["comparison"], r["speaker_type"]): float(r["jsd"]) for r in rows("oncoco_rq1_distances.csv")}
    C_RL, C_RE, RL_RE = ("HH_roleplay_chat vs H_LLM_roleplay_chat", "HH_real_chat vs H_LLM_roleplay_chat",
                         "HH_real_chat vs HH_roleplay_chat")
    fig, ax = plt.subplots(figsize=(6.6, 5.2)); ax.set_axis_off(); ax.set_aspect("equal")
    pos = {"H_LLM_roleplay_chat": (0, 1.05), "HH_roleplay_chat": (-1.25, -0.7), "HH_real_chat": (1.25, -0.7)}
    lab = {"H_LLM_roleplay_chat": "H–LLM\nroleplay chat", "HH_roleplay_chat": "HH roleplay\nchat", "HH_real_chat": "HH real\nchat"}
    for a, b, comp in [("HH_roleplay_chat", "H_LLM_roleplay_chat", C_RL),
                       ("HH_real_chat", "H_LLM_roleplay_chat", C_RE),
                       ("HH_roleplay_chat", "HH_real_chat", RL_RE)]:
        (x1, y1), (x2, y2) = pos[a], pos[b]; cl, co = d[(comp, "Client")], d[(comp, "Counsellor")]
        ax.plot([x1, x2], [y1, y2], color="0.6", lw=1.2 + 11 * cl, solid_capstyle="round", zorder=1)
        ax.text((x1+x2)/2, (y1+y2)/2, f"Client {cl:.3f}\nCounselor {co:.3f}", ha="center", va="center", fontsize=10,
                zorder=3, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", lw=0.8))
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.26, fc=COND[k], ec="white", lw=2, zorder=2))
        ax.text(x, y + (0.42 if y > 0 else -0.42), lab[k], ha="center", va="bottom" if y > 0 else "top",
                fontsize=11, fontweight="bold", color="0.15")
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-1.7, 1.9); save(fig, "main_contribution_triad_jsd")

# ---------------------------------------------------------------- 2. model proximity
def realness():
    d = {(r["model"], r["speaker_type"]): float(r["jsd"]) for r in rows("oncoco_rq1b_model_realness.csv")}
    models = [m for m in MORDER if (m, "Client") in d]; x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for i, sp in enumerate(["Client", "Counsellor"]):
        b = ax.bar(x + (i - 0.5) * w, [d[(m, sp)] for m in models], w,
                   label="Counselor" if sp == "Counsellor" else sp, color=SPKC[sp], edgecolor="white")
        ax.bar_label(b, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([MODEL[m] for m in models])
    ax.set_ylabel("Distributional proximity to HH real chat\n(Jensen–Shannon distance; lower = closer)")
    ax.set_ylim(0, max(d.values()) * 1.18)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.09)); save(fig, "model_realness_chat_jsd")

# ---------------------------------------------------------------- 3. behavioral temporal
def behavioral_temporal():
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows("oncoco_temporal_distribution.csv"):
        if r["speaker_type"] != "Client": continue
        b = bucket(r["label"])
        if b: agg[(r["condition"], int(r["decile"]))][b] += float(r["proportion"])
    conds = ["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat"]
    buckets = [("Problem", "Problem disclosure"), ("Solution", "Solution-space engagement"), ("Request", "Reciprocal requests")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, (bk, title) in zip(axes, buckets):
        for c in conds:
            ax.plot([d*10+5 for d in range(10)], [agg[(c, d)].get(bk, 0) for d in range(10)],
                    marker="o", ms=3, lw=2, color=COND[c], label=CLABEL[c])
        ax.set_title(title); ax.set_xlabel("Conversation progress (%)")
    axes[0].set_ylabel("Share of client spans"); axes[0].legend(loc="upper left")
    fig.tight_layout(); save(fig, "behavioral_signatures_temporal")

# ---------------------------------------------------------------- 4. behavioral evolution
def behavioral_evolution():
    R = json.loads((ROOT / "data/processed/combined/normalized/oncoco_classification_all.json").read_text())
    df = collections.defaultdict(lambda: collections.Counter())
    for r in R:
        if r.get("speaker_type") != "Client": continue
        src, m = r.get("source"), r.get("model") or ""
        g = m if src == "H_LLM_roleplay_chat" else (src if src in ("HH_real_chat", "HH_roleplay_chat") else None)
        if g is None: continue
        for s in r.get("sentence_classification") or []:
            df[g][bucket(s.get("predicted_label"))] += 1
    def share(g, b):
        tot = sum(df[g].values()); return df[g][b] / tot if tot else 0
    buckets = [("Problem", "Problem disclosure"), ("Solution", "Solution-space engagement"), ("Request", "Reciprocal requests")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7)); xs = range(len(MORDER))
    for ax, (bk, title) in zip(axes, buckets):
        ax.plot(list(xs), [share(m, bk) for m in MORDER], marker="o", ms=7, lw=2, color="#D55E00", label="H–LLM (by model)")
        ax.axhline(share("HH_real_chat", bk), ls="--", lw=1.6, color="#009E73", label="HH real")
        ax.axhline(share("HH_roleplay_chat", bk), ls=":", lw=1.8, color="#0072B2", label="HH roleplay")
        ax.set_xticks(list(xs)); ax.set_xticklabels([f"{MODEL[m]}\n{MSEEN[m]}" for m in MORDER], fontsize=9)
        ax.set_title(title); ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Share of client spans"); axes[0].legend(loc="best")
    fig.suptitle("Behavioral signatures across model generations (Client)", y=1.0)
    fig.tight_layout(); save(fig, "behavioral_signatures_evolution")

# ---------------------------------------------------------------- 5. monthly evolution
def monthly_evolution():
    data = rows("oncoco_hllm_monthly_realness.csv")
    months = sorted({r["month"] for r in data})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, sp in zip(axes, ["Client", "Counsellor"]):
        for base, col, ls in [("HH_real_chat", "#009E73", "-"), ("HH_roleplay_chat", "#0072B2", "--")]:
            pts = {r["month"]: float(r["jsd"]) for r in data if r["speaker_type"] == sp and r["baseline"] == base}
            ax.plot(months, [pts.get(m, np.nan) for m in months], marker="o", ms=5, lw=2, color=col, ls=ls,
                    label=f"vs {CLABEL[base]}")
        ax.set_title("Counselor" if sp == "Counsellor" else sp); ax.set_xlabel("Month")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("JSD to baseline"); axes[0].legend(loc="best")
    fig.tight_layout(); save(fig, "hllm_evolution_monthly_jsd")

# ---------------------------------------------------------------- 6. CAIA shifts
def caia_shifts():
    data = [r for r in rows("oncoco_rq5_caia_label_effects.csv") if r["speaker_type"] == "Counsellor"]
    data.sort(key=lambda r: -abs(float(r["diff"]))); top = data[:8][::-1]
    labs = [LM.get(r["label"], r["label"]) for r in top]
    labs = [l if len(l) <= 42 else l[:40] + "…" for l in labs]
    vals = [float(r["diff"]) for r in top]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(range(len(top)), vals, color=["#CC79A7" if v > 0 else "#999999" for v in vals], edgecolor="white")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labs, fontsize=9)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("Δ proportion (HH+assistant − HH roleplay), counselor")
    ax.grid(axis="y", visible=False); fig.tight_layout(); save(fig, "rq5_caia_top_label_shifts")

# ---------------------------------------------------------------- 7. transition heatmaps
def transition_heatmaps():
    MACRO = ["Opening", "Empathy", "Clarify", "Objectives", "Motivation", "Resources", "Help", "Moderation", "Other", "Closing"]
    order = [f"{r}:{m}" for r in ("CL", "CO") for m in MACRO]
    bycond = collections.defaultdict(dict)
    for r in rows("oncoco_transition_matrix.csv"):
        bycond[r["condition"]][(r["from"], r["to"])] = float(r["prob"])
    nice = {"HH_roleplay_chat": "HH roleplay chat", "H_LLM_roleplay_chat": "H–LLM roleplay chat", "HH_real_chat": "HH real chat"}
    for cond, fname in [("HH_roleplay_chat", "transition_HH_roleplay_chat"),
                        ("H_LLM_roleplay_chat", "transition_H_LLM_roleplay_chat"),
                        ("HH_real_chat", "transition_HH_real_chat")]:
        m = bycond[cond]
        cats = [c for c in order if any(k[0] == c or k[1] == c for k in m)]
        M = np.array([[m.get((a, b), 0.0) for b in cats] for a in cats])
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        im = ax.imshow(M, cmap="cividis", aspect="auto", vmin=0, vmax=min(1.0, M.max() or 1))
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=90, fontsize=8.5)
        ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=8.5)
        ax.set_xlabel("to"); ax.set_ylabel("from"); ax.set_title(f"Transitions — {nice[cond]} (row-normalized)")
        ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(to | from)")
        fig.tight_layout(); save(fig, fname)

# ---------------------------------------------------------------- 8. t-SNE
def tsne():
    data = rows("oncoco_tsne_sample.csv")
    for sp, fname in [("Client", "tsne_client"), ("Counsellor", "tsne_counsellor")]:
        pts = [r for r in data if r["speaker_type"] == sp]
        if not pts: continue
        fig, ax = plt.subplots(figsize=(7.0, 5.6))
        for cond in COND:
            xs = [float(r["x"]) for r in pts if r["condition"] == cond]
            ys = [float(r["y"]) for r in pts if r["condition"] == cond]
            if xs: ax.scatter(xs, ys, s=8, alpha=0.55, color=COND[cond], label=CLABEL[cond], edgecolors="none")
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        ax.set_title(f"Sentence embeddings — {'Counselor' if sp=='Counsellor' else sp}")
        ax.grid(True, alpha=0.3); ax.legend(markerscale=2, loc="best", fontsize=9.5)
        fig.tight_layout(); save(fig, fname)

# ---------------------------------------------------------------- 9. within-condition JSD
def within_jsd():
    data = rows("oncoco_within_condition_jsd_per_conversation.csv")
    order = ["HH_roleplay_chat", "HH_real_chat", "H_LLM_roleplay_chat", "HH_roleplay_mail", "HH_CAIA_mail"]
    for sp, fname in [("Client", "within_jsd_client"), ("Counsellor", "within_jsd_counselor")]:
        groups, used = [], []
        for c in order:
            vals = [float(r["jsd"]) for r in data if r["speaker_type"] == sp and r["condition"] == c]
            if vals: groups.append(vals); used.append(c)
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        bp = ax.boxplot(groups, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops=dict(color="0.15", lw=1.5))
        for patch, c in zip(bp["boxes"], used):
            patch.set_facecolor(COND[c]); patch.set_alpha(0.85); patch.set_edgecolor("white")
        ax.set_xticks(range(1, len(used) + 1)); ax.set_xticklabels([CLABEL[c] for c in used], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Conversation-level JSD to condition mean")
        ax.set_title(f"Within-condition variability — {'Counselor' if sp=='Counsellor' else sp}")
        fig.tight_layout(); save(fig, fname)


if __name__ == "__main__":
    for fn in (triad, realness, behavioral_temporal, behavioral_evolution, monthly_evolution,
               caia_shifts, transition_heatmaps, tsne, within_jsd):
        try:
            fn()
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
