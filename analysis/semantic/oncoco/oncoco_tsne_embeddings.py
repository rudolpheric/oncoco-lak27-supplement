#!/usr/bin/env python3
"""Compute embedding-based t-SNE plots for OnCoCo sentence data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE embedding analysis for OnCoCo sentences.")
    parser.add_argument(
        "--data-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="Path to OnCoCo sentence classification JSON.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="HF model for sentence embeddings.",
    )
    parser.add_argument("--sample-per-condition", type=int, default=600, help="Max sentences per condition per speaker.")
    parser.add_argument(
        "--exclude-conditions",
        nargs="*",
        default=[],
        help="Condition names (source values) to drop before sampling, e.g. HH_real_mail.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "figures" / "oncoco"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_tsne_sample.csv"),
        help="CSV with sampled sentences and t-SNE coordinates.",
    )
    parser.add_argument(
        "--out-silhouette",
        default=str(PROJECT_ROOT / "results" / "tables" / "oncoco_embedding_silhouette.csv"),
        help="CSV with silhouette scores per speaker.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_sentences(rows: List[Dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        condition = r.get("source")
        speaker_type = r.get("speaker_type")
        for sent in r.get("sentence_classification", []) or []:
            text = sent.get("text")
            if not (condition and speaker_type and text):
                continue
            records.append(
                {
                    "condition": condition,
                    "speaker_type": speaker_type,
                    "text": text.strip(),
                }
            )
    df = pd.DataFrame.from_records(records)
    return df.dropna(subset=["condition", "speaker_type", "text"]).drop_duplicates()


def sample_by_condition(df: pd.DataFrame, max_per_condition: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = []
    for (condition, speaker), group in df.groupby(["condition", "speaker_type"]):
        if len(group) <= max_per_condition:
            sampled.append(group)
            continue
        idx = rng.choice(group.index.to_numpy(), size=max_per_condition, replace=False)
        sampled.append(group.loc[idx])
    return pd.concat(sampled, ignore_index=True)


def embed_texts(texts: List[str], tokenizer, model, batch_size: int, device: torch.device) -> np.ndarray:
    embeddings = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # Mask-weighted mean pooling. A plain mean over dim=1 averages the padding
            # positions too, so each embedding depends on the longest text in its batch
            # and short spans are pulled towards the pad representation.
            mask = inputs["attention_mask"].unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def plot_tsne(df: pd.DataFrame, speaker_type: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10", n_colors=df["condition"].nunique())
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="condition",
        palette=palette,
        s=20,
        alpha=0.7,
        linewidth=0,
    )
    plt.title(f"t-SNE of Sentence Embeddings ({speaker_type})")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.data_json))
    df = collect_sentences(rows)
    if args.exclude_conditions:
        df = df[~df["condition"].isin(set(args.exclude_conditions))].copy()
        print(f"Excluded conditions {args.exclude_conditions}; remaining: {sorted(df['condition'].unique())}")
    sampled = sample_by_condition(df, args.sample_per_condition, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    all_frames = []
    silhouette_rows = []

    for speaker in sorted(sampled["speaker_type"].unique()):
        sub = sampled[sampled["speaker_type"] == speaker].copy()
        texts = sub["text"].tolist()
        embeddings = embed_texts(texts, tokenizer, model, args.batch_size, device)

        norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings_norm = embeddings / norm

        if sub["condition"].nunique() > 1 and len(sub) >= 10:
            try:
                sil = float(silhouette_score(embeddings_norm, sub["condition"], metric="cosine"))
            except Exception:
                sil = float("nan")
        else:
            sil = float("nan")
        silhouette_rows.append({"speaker_type": speaker, "silhouette_cosine": sil})

        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30, max_iter=1000)
        coords = tsne.fit_transform(embeddings_norm)
        sub = sub.reset_index(drop=True)
        sub["x"] = coords[:, 0]
        sub["y"] = coords[:, 1]

        out_path = Path(args.out_dir) / f"tsne_{speaker.lower()}.png"
        plot_tsne(sub, speaker, out_path)
        all_frames.append(sub)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_frames, ignore_index=True).to_csv(out_csv, index=False)

    out_sil = Path(args.out_silhouette)
    out_sil.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(silhouette_rows).to_csv(out_sil, index=False)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_sil}")


if __name__ == "__main__":
    main()
