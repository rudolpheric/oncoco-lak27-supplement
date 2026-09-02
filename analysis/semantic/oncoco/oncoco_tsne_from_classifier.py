#!/usr/bin/env python3
"""Extract embeddings from trained OnCoCo classifier and create interactive t-SNE plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE from OnCoCo classifier embeddings.")
    parser.add_argument(
        "--data-json",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="Path to OnCoCo sentence classification JSON.",
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "analysis" / "models" / "segmentation" / "SCC-OnCoCo" / "model" / "oncoco"),
        help="Path to trained OnCoCo XLM-RoBERTa model.",
    )
    parser.add_argument(
        "--sample-per-condition",
        type=int,
        default=500,
        help="Max sentences per condition per speaker.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "figures" / "oncoco" / "interactive"),
        help="Output directory for interactive plots.",
    )
    parser.add_argument(
        "--max-hover-chars",
        type=int,
        default=200,
        help="Max characters in hover text.",
    )
    return parser.parse_args()


def load_sentences(json_path: Path) -> pd.DataFrame:
    """Load sentences with labels from JSON."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    records = []
    for msg in data:
        condition = msg.get("source", "")
        speaker_type = msg.get("speaker_type", "")

        for sent in msg.get("sentence_classification", []) or []:
            text = sent.get("text", "").strip()
            label_full = sent.get("predicted_label", "unknown")

            if text and condition and speaker_type and label_full != "unknown":
                # Extract 5th layer label
                parts = label_full.split("-")
                label_5th = parts[4] if len(parts) >= 5 and parts[4] != "*" else parts[1] if len(parts) >= 2 else "unknown"

                records.append({
                    "condition": condition,
                    "speaker_type": speaker_type,
                    "text": text,
                    "label_full": label_full,
                    "label_5th": label_5th,
                })

    return pd.DataFrame(records)


def sample_sentences(df: pd.DataFrame, max_per_condition: int, seed: int) -> pd.DataFrame:
    """Sample sentences per condition and speaker."""
    rng = np.random.default_rng(seed)
    sampled = []

    for (condition, speaker), group in df.groupby(["condition", "speaker_type"]):
        if len(group) <= max_per_condition:
            sampled.append(group)
        else:
            idx = rng.choice(group.index.to_numpy(), size=max_per_condition, replace=False)
            sampled.append(group.loc[idx])

    return pd.concat(sampled, ignore_index=True)


def extract_embeddings(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract hidden layer representations from OnCoCo classifier."""
    model.eval()
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use the [CLS] token embedding from last hidden layer
            # hidden_states[-1] shape: (batch_size, seq_len, hidden_size)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # Take [CLS] token
            embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(embeddings)


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def create_interactive_plot(
    df: pd.DataFrame,
    speaker: str,
    modality: str,
    out_path: Path,
    max_chars: int,
) -> None:
    """Create interactive Plotly scatter plot."""
    df_filtered = df[
        (df["speaker_type"] == speaker) &
        (df["modality"] == modality)
    ].copy()

    if len(df_filtered) == 0:
        print(f"[WARN] No data for {speaker} {modality}")
        return

    df_filtered["text_display"] = df_filtered["text"].apply(
        lambda x: truncate_text(x, max_chars)
    )

    fig = px.scatter(
        df_filtered,
        x="tsne_x",
        y="tsne_y",
        color="label_5th",
        symbol="condition",
        hover_data={
            "tsne_x": False,
            "tsne_y": False,
            "label_5th": True,
            "label_full": True,
            "condition": True,
            "text_display": True,
        },
        title=f"OnCoCo t-SNE: {speaker} - {modality.upper()} (from classifier embeddings)",
        labels={
            "label_5th": "Label (Layer 5)",
            "label_full": "Full Label",
            "condition": "Condition",
            "text_display": "Text",
        },
        width=1920,
        height=1080,
    )

    fig.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title_text="Label",
        hovermode="closest",
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color="DarkSlateGrey")),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"[SAVED] {speaker} {modality}: {out_path}")


def main() -> None:
    args = parse_args()

    # Load data
    print("[INFO] Loading sentences...")
    df = load_sentences(Path(args.data_json))
    print(f"[INFO] Loaded {len(df)} sentences")

    # Sample
    print("[INFO] Sampling sentences...")
    df_sampled = sample_sentences(df, args.sample_per_condition, args.seed)
    print(f"[INFO] Sampled {len(df_sampled)} sentences")

    # Load OnCoCo model
    print(f"[INFO] Loading OnCoCo classifier from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)

    # Extract embeddings per speaker type
    all_frames = []

    for speaker in sorted(df_sampled["speaker_type"].unique()):
        print(f"\n[INFO] Processing speaker: {speaker}")
        df_speaker = df_sampled[df_sampled["speaker_type"] == speaker].copy()

        texts = df_speaker["text"].tolist()
        embeddings = extract_embeddings(texts, tokenizer, model, args.batch_size, device)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings_norm = embeddings / norms

        # Run t-SNE
        print(f"[INFO] Running t-SNE for {speaker}...")
        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=30, max_iter=1000)
        coords = tsne.fit_transform(embeddings_norm)

        df_speaker = df_speaker.reset_index(drop=True)
        df_speaker["tsne_x"] = coords[:, 0]
        df_speaker["tsne_y"] = coords[:, 1]

        all_frames.append(df_speaker)

    # Combine all data
    df_all = pd.concat(all_frames, ignore_index=True)

    # Add modality column
    df_all["modality"] = df_all["condition"].apply(
        lambda x: "chat" if "chat" in x.lower() else "mail"
    )

    # Create 4 plots
    out_dir = Path(args.out_dir)

    for speaker in ["Client", "Counsellor"]:
        for modality in ["mail", "chat"]:
            out_path = out_dir / f"tsne_oncoco_{speaker.lower()}_{modality}.html"
            create_interactive_plot(df_all, speaker, modality, out_path, args.max_hover_chars)

    # Save data
    out_csv = out_dir.parent.parent / "tables" / "oncoco_tsne_from_classifier.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Data with OnCoCo embeddings: {out_csv}")


if __name__ == "__main__":
    main()
