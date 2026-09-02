#!/usr/bin/env python3
"""Derive chat-only variants of paper tables/figures (the LAK version drops the mail/CAIA conditions).

Reads the canonical CSVs/tex (which keep all six conditions for the archival pipeline)
and writes *_chat.tex / *_chat.pdf files that the paper inputs.
"""
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "results/tables"
F = ROOT / "results/figures/oncoco"
CHAT = [("HH_roleplay_chat", "HH roleplay chat", "#0072B2"),
        ("HH_real_chat", "HH real chat", "#009E73"),
        ("H_LLM_roleplay_chat", "H--LLM roleplay chat", "#D55E00")]
CHAT_KEYS = [c for c, _, _ in CHAT]


# In a chat-only paper the " chat" qualifier on every condition is noise and widens
# every table, so the derived variants carry the short names the prose uses. Longest
# label first, so a shorter pattern can never bite off part of a longer one.
SHORTEN = [("H--LLM roleplay chat", "H--LLM roleplay"), ("HH roleplay chat", "HH roleplay"),
           ("HH real chat", "HH real"), ("H--LLM chat", "H--LLM")]


def normalise_first_col(dst):
    """Half-width minipages make fixed p{x\\linewidth} first columns wrap; use natural width
    and let \\fitwidth shrink the table instead."""
    f = T / dst
    txt = re.sub(r"p\{0\.\d+\\linewidth\}", "l", f.read_text())
    for long, short in SHORTEN:
        txt = txt.replace(long, short)
    f.write_text(txt)


def keep_rows(src, dst, patterns):
    """Keep header/footer plus body rows matching any pattern."""
    out = []
    for line in (T / src).read_text().splitlines():
        body = " & " in line and line.rstrip().endswith("\\\\")
        if not body or any(re.match(p, line.strip()) for p in patterns):
            out.append(line)
    (T / dst).write_text("\n".join(out) + "\n")
    normalise_first_col(dst)


CHATP = [r"HH roleplay chat", r"HH real chat", r"H--LLM", r"Comparison", r"Condition",
         r"Label", r"Model", r"Semester", r" & ", r"\\multicolumn"]

# 1) noise band: drop CAIA rows
keep_rows("oncoco_noise_band.tex", "oncoco_noise_band_chat.tex",
          [r"(HH roleplay chat|HH real chat|H--LLM)", r"Comparison"])
# 2) within summary: chat conditions only
keep_rows("oncoco_within_condition_jsd_summary.tex", "oncoco_within_condition_jsd_summary_chat.tex",
          [r"(HH roleplay chat|HH real chat|H--LLM roleplay chat) &", r"Condition"])
# 3) transition summary
keep_rows("oncoco_transition_summary.tex", "oncoco_transition_summary_chat.tex",
          [r"(HH roleplay chat|HH real chat|H--LLM roleplay chat) &", r"Condition"])
# 4) HMM states: chat blocks (state rows start with '&'; keep those following chat blocks)
src = (T / "oncoco_hmm_state_summary.tex").read_text().splitlines()
out, keep = [], False
for line in src:
    s = line.strip()
    if s.endswith("\\\\") and " & " in s and not s.startswith("&"):
        keep = any(s.startswith(d) for _, d, _ in [(0, n, 0) for _, n, _ in CHAT]) or \
               any(s.startswith(n) for _, n, _ in CHAT)
    if s.endswith("\\\\") and " & " in s:
        if keep or s.startswith("Condition"):
            out.append(line)
    elif s == "\\midrule":
        if keep or not out or out[-1].strip().startswith("Condition"):
            out.append(line)
    else:
        out.append(line)
# collapse possible duplicate midrules
txt = re.sub(r"(\\midrule\n)+", r"\\midrule\n", "\n".join(out) + "\n")
txt = txt.replace("\\midrule\n\\bottomrule", "\\bottomrule")
(T / "oncoco_hmm_state_summary_chat.tex").write_text(txt)
normalise_first_col("oncoco_hmm_state_summary_chat.tex")
# 5) HMM model selection compact
keep_rows("oncoco_hmm_model_selection_compact.tex", "oncoco_hmm_model_selection_compact_chat.tex",
          [r"(HH roleplay chat|HH real chat|H--LLM roleplay chat) &", r"Condition", r"^&", r"\\multicolumn"])
# 6) seg ablation: chat-vs-chat comparisons only
keep_rows("oncoco_rq1_segmentation_ablation.tex", "oncoco_rq1_segmentation_ablation_chat.tex",
          [r"HH real chat vs H--LLM chat", r"HH real chat vs HH roleplay chat",
           r"HH roleplay chat vs H--LLM chat", r"Comparison"])
# 6b) persistence: chat-vs-chat comparisons only (drops the mail block)
keep_rows("oncoco_persistence_summary.tex", "oncoco_persistence_summary_chat.tex",
          [r"(HH roleplay chat|HH real chat|H--LLM roleplay chat) vs", r"Comparison"])

# 7) within-condition figures, chat only ---------------------------------------
pc = pd.read_csv(T / "oncoco_within_condition_jsd_per_conversation.csv")
for sp, fname, title in [("Client", "within_jsd_client_chat.pdf", "Client"),
                         ("Counsellor", "within_jsd_counselor_chat.pdf", "Counselor")]:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    data, labels, colors = [], [], []
    for cond, disp, col in CHAT:
        v = pc[(pc.condition == cond) & (pc.speaker_type == sp)]["jsd"]
        data.append(v.values); labels.append(disp.replace("--", "\u2013")); colors.append(col)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55, showfliers=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("Conversation JSD to condition mean")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(F / fname, bbox_inches="tight")
    plt.close(fig)

# 8) t-SNE figures, chat only ---------------------------------------------------
# Reads the chat-only sample written by oncoco_tsne_embeddings.py with
# --exclude-conditions HH_real_mail HH_roleplay_mail HH_CAIA_mail. The joint
# six-condition embedding stays canonical; the paper figures use this rerun so
# the projection and the silhouette are computed on the chat corpus alone.
tsne_csv = T / "oncoco_tsne_sample_chat.csv"
if tsne_csv.exists():
    ts = pd.read_csv(tsne_csv)
    for sp, fname, title in [("Client", "tsne_client_chat.pdf", "Client"),
                             ("Counsellor", "tsne_counsellor_chat.pdf", "Counselor")]:
        sub = ts[ts.speaker_type == sp]
        fig, ax = plt.subplots(figsize=(7.0, 5.6))
        for cond, disp, col in CHAT:
            g = sub[sub.condition == cond]
            ax.scatter(g.x, g.y, s=8, alpha=0.55, color=col, edgecolors="none",
                       label=disp.replace(" chat", "").replace("--", "–"))
        ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
        ax.set_title(f"Sentence embeddings — {title}")
        ax.grid(alpha=0.3); ax.legend(markerscale=2, loc="best", fontsize=9.5)
        fig.tight_layout()
        fig.savefig(F / fname, bbox_inches="tight")
        plt.close(fig)
else:
    print("oncoco_tsne_sample_chat.csv fehlt; t-SNE-Chat-Figuren uebersprungen")

print("chat-only Tabellen + Figuren geschrieben")
for f in ["oncoco_noise_band_chat.tex", "oncoco_within_condition_jsd_summary_chat.tex",
          "oncoco_transition_summary_chat.tex", "oncoco_hmm_state_summary_chat.tex",
          "oncoco_hmm_model_selection_compact_chat.tex", "oncoco_rq1_segmentation_ablation_chat.tex",
          "oncoco_persistence_summary_chat.tex"]:
    print("---", f)
    print((T / f).read_text())
