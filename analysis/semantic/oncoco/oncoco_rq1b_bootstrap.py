#!/usr/bin/env python3
"""Bootstrap confidence intervals and persona coverage for RQ1b model comparison."""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ1b bootstrap and persona coverage.")
    parser.add_argument(
        "--data-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="Path to OnCoCo sentence classification JSON.",
    )
    parser.add_argument(
        "--chat-raw-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "human_llm" / "chats"),
        help="Path to raw H-LLM chat JSON directory (for persona metadata).",
    )
    parser.add_argument(
        "--out-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_rq1b_model_realness_bootstrap.csv"),
        help="Output CSV for bootstrap CIs.",
    )
    parser.add_argument(
        "--out-tex",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_rq1b_model_realness_bootstrap.tex"),
        help="Output TeX summary table.",
    )
    parser.add_argument(
        "--out-persona-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_model_persona_coverage.csv"),
        help="Output CSV for persona coverage by model.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=20000,  # the published Table S13 / Figure 2 intervals
        help="Number of bootstrap draws.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_hllm_personas(chat_raw_dir: Path) -> pd.DataFrame:
    """Persona metadata per conversation, keyed on (source_file, id).

    The raw exports reuse conversation ids across files, so a frame keyed on the id alone
    kept whichever file happened to be read first and attached its persona to every other
    conversation carrying that id.
    """
    records = []
    root = chat_raw_dir.resolve().parents[3]
    for fp in sorted(chat_raw_dir.rglob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        rel = str(fp.resolve().relative_to(root))

        if isinstance(data, list):
            convs = [c for c in data if isinstance(c, dict)]
        elif isinstance(data, dict):
            convs = None
            for key in ("counsellings", "counselings", "conversations", "data"):
                if isinstance(data.get(key), list):
                    convs = [c for c in data[key] if isinstance(c, dict)]
                    break
            if convs is None:
                convs = [data]
        else:
            continue

        for conv in convs:
            cid = conv.get("id")
            if cid is None:
                continue
            persona_id = conv.get("persona_id")
            persona_name = ""
            persona = conv.get("persona")
            if isinstance(persona, dict):
                persona_name = str(persona.get("name") or "")
            records.append(
                {
                    "conversation_id": f"{rel}::{cid}",
                    "persona_id": str(persona_id) if persona_id is not None else "",
                    "persona_name": persona_name,
                }
            )
    return pd.DataFrame(records).drop_duplicates(subset=["conversation_id"])


def build_sentence_df(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        condition = r.get("source")
        speaker = r.get("speaker_type")
        model = r.get("model") or "unknown"
        # (source_file, id): ids repeat across raw exports. The conversation is the bootstrap
        # resampling unit, so a merged key would silently shrink the number of independent
        # units and narrow the intervals.
        conv_id = f"{r.get('source_file', '')}::{r.get('id')}" if r.get("id") is not None else None
        if not condition or not speaker:
            continue
        for sent in r.get("sentence_classification", []) or []:
            label = sent.get("predicted_label")
            if not label:
                continue
            records.append(
                {
                    "condition": condition,
                    "speaker_type": speaker,
                    "model": model,
                    "conversation_id": str(conv_id) if conv_id is not None else "",
                    "label": label,
                }
            )
    return pd.DataFrame(records)


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))

    return sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def probs_from_labels(labels: pd.Series, universe: List[str]) -> np.ndarray:
    counts = labels.value_counts()
    arr = np.array([counts.get(lab, 0.0) for lab in universe], dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.zeros(len(universe), dtype=float)
    return arr / total


def _conversation_matrix(df: pd.DataFrame, universe: List[str]) -> np.ndarray:
    """Conversations x labels count matrix, one row per conversation."""
    counts = (
        df.groupby(["conversation_id", "label"]).size().unstack(fill_value=0)
        .reindex(columns=universe, fill_value=0)
    )
    return counts.to_numpy(dtype=float)


def bootstrap_jsd(
    model_df: pd.DataFrame,
    base_df: pd.DataFrame,
    universe: List[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Conversation-level bootstrap of the JSD between a model and the baseline.

    Conversations are the independent sampling unit; spans within a conversation are
    strongly correlated, so resampling spans (or drawing a multinomial at the observed
    span count) treats each span as independent evidence and yields intervals that are
    far too narrow. Both arms are resampled with replacement.
    """
    m = _conversation_matrix(model_df, universe)
    b = _conversation_matrix(base_df, universe)
    draws = []
    for _ in range(n_bootstrap):
        pm = m[rng.integers(0, len(m), len(m))].sum(axis=0)
        pb = b[rng.integers(0, len(b), len(b))].sum(axis=0)
        if pm.sum() <= 0 or pb.sum() <= 0:
            continue
        draws.append(js_distance(pm / pm.sum(), pb / pb.sum()))
    return np.asarray(draws, dtype=float)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    rows = load_rows(Path(args.data_json))
    df = build_sentence_df(rows)
    baseline = df[df["condition"] == "HH_real_chat"].copy()
    hllm = df[df["condition"] == "H_LLM_roleplay_chat"].copy()

    if baseline.empty or hllm.empty:
        raise SystemExit("Missing baseline or H-LLM rows in sentence data.")

    all_labels = sorted(set(df["label"]))
    out_rows = []

    for model in sorted(hllm["model"].unique()):
        for speaker in sorted(hllm["speaker_type"].unique()):
            sub_model = hllm[(hllm["model"] == model) & (hllm["speaker_type"] == speaker)]
            sub_base = baseline[baseline["speaker_type"] == speaker]
            if sub_model.empty or sub_base.empty:
                continue

            p_model = probs_from_labels(sub_model["label"], all_labels)
            p_base = probs_from_labels(sub_base["label"], all_labels)
            point = js_distance(p_model, p_base)
            draws = bootstrap_jsd(
                model_df=sub_model,
                base_df=sub_base,
                universe=all_labels,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
            )
            out_rows.append(
                {
                    "model": model,
                    "speaker_type": "Counselor" if speaker == "Counsellor" else speaker,
                    "n_model_sentences": int(len(sub_model)),
                    "n_baseline_sentences": int(len(sub_base)),
                    "n_model_conversations": int(sub_model["conversation_id"].nunique()),
                    "n_baseline_conversations": int(sub_base["conversation_id"].nunique()),
                    "jsd": float(point),
                    "jsd_ci_low": float(np.quantile(draws, 0.025)),
                    "jsd_ci_high": float(np.quantile(draws, 0.975)),
                }
            )

    out_df = pd.DataFrame(out_rows).sort_values(["speaker_type", "jsd", "model"])
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # Persona coverage by model (conversation-level)
    persona_df = parse_hllm_personas(Path(args.chat_raw_dir))
    conv_model = (
        rows_to_conv_model(rows)
    )
    coverage = conv_model.merge(persona_df, on="conversation_id", how="left")
    coverage["persona_id"] = coverage["persona_id"].fillna("")
    coverage_rows = []
    for model, sub in coverage.groupby("model"):
        counts = sub["persona_id"].value_counts()
        probs = counts / max(float(counts.sum()), 1.0)
        entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum())
        n_personas = int(sub[sub["persona_id"] != ""]["persona_id"].nunique())
        coverage_rows.append(
            {
                "model": model,
                "n_conversations": int(sub["conversation_id"].nunique()),
                "n_personas": n_personas,
                "persona_entropy_bits": entropy,
                "top_persona_share": float(probs.max() if len(probs) else np.nan),
            }
        )
    coverage_df = pd.DataFrame(coverage_rows).sort_values("model")
    out_persona = Path(args.out_persona_csv)
    out_persona.parent.mkdir(parents=True, exist_ok=True)
    out_persona.write_text(coverage_df.to_csv(index=False), encoding="utf-8")

    # TeX: client/counselor with CIs
    pivot = out_df.pivot(index="model", columns="speaker_type", values=["jsd", "jsd_ci_low", "jsd_ci_high"])
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model & Client JSD [95\\% CI] & Counselor JSD [95\\% CI] \\\\",
        "\\midrule",
    ]
    for model in sorted(pivot.index):
        c = fmt_ci(pivot, model, "Client")
        co = fmt_ci(pivot, model, "Counselor")
        lines.append(f"{tex_escape(MODEL_DISPLAY.get(model, model))} & {c} & {co} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    Path(args.out_tex).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {args.out_tex}")
    print(f"Wrote {args.out_persona_csv}")


def rows_to_conv_model(rows: List[Dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        if r.get("source") != "H_LLM_roleplay_chat":
            continue
        cid = r.get("id")
        if cid is None:
            continue
        recs.append(
            {
                "conversation_id": f"{r.get('source_file', '')}::{cid}",
                "model": r.get("model") or "unknown",
            }
        )
    return pd.DataFrame(recs).drop_duplicates()


MODEL_DISPLAY = {
    "GPT_OSS_120B": "GPT-OSS-120B",
    "LLama3_3_70B": "Llama 3.3 70B",
    "Mixtral": "Mixtral 8x7B",
}


def fmt_ci(pivot: pd.DataFrame, model: str, speaker: str) -> str:
    try:
        jsd = float(pivot.loc[model, ("jsd", speaker)])
        lo = float(pivot.loc[model, ("jsd_ci_low", speaker)])
        hi = float(pivot.loc[model, ("jsd_ci_high", speaker)])
        return f"{jsd:.3f} [{lo:.3f}, {hi:.3f}]"
    except Exception:
        return "--"


def tex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


if __name__ == "__main__":
    main()
