#!/usr/bin/env python3
"""Store per-item classifier probabilities as a sidecar, without touching any existing artifact.

Why: the paper's measurements inherit the OnCoCo classifier's errors, and a reviewer will ask
whether those errors are correlated with condition -- if LLM text were out of distribution for a
classifier trained on human-written counseling messages, its confidence should degrade relative to
the human conditions. That is testable, and this script produces the data to test it.

Design decisions:

  * The spans are READ BACK from the classification artifact rather than re-segmented. This
    guarantees byte-identical classifier inputs (no risk of segmentation drift) and avoids needing
    the LoRA/peft stack, which is not installed in the analysis environment.
  * Items are processed in the artifact's stored order at the same batch size as the original run,
    so batch composition -- and therefore padding, and therefore the exact float results on MPS --
    matches. The script asserts that the recomputed argmax reproduces the stored label and reports
    the agreement rate rather than silently repairing disagreements.
  * Nothing is written into the classification JSONs. The sidecar is a separate parquet keyed on
    (source_file, id, msg_message_number, unit, sentence_index).

Usage:
    python oncoco_confidence_sidecar.py                     # SaT artifact, spans + messages
    python oncoco_confidence_sidecar.py --limit 200         # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
NORM = ROOT / "data/processed/combined/normalized"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-json", default=str(NORM / "oncoco_classification_all.json"))
    p.add_argument("--model-path",
                   default=str(ROOT / "analysis/models/oncoco/xlm-roberta-large-OnCoCo-DE-EN"))
    # gzipped CSV rather than parquet: pyarrow is not installed in the analysis environment,
    # and adding a dependency for ~57k rows is not worth it. The two list columns are JSON-encoded.
    p.add_argument("--out-csv", default=str(NORM / "oncoco_confidence_sat.csv.gz"))
    p.add_argument("--out-manifest", default=str(NORM / "oncoco_confidence_manifest.json"))
    p.add_argument("--batch-size", type=int, default=16,
                   help="Must match the original classification run (16) for exact reproduction.")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="Process only the first N items (smoke test).")
    return p.parse_args()


def apply_role_prefix(text: str, speaker_type: str) -> str:
    """Mirrors analysis/semantic/classification/classification_all.py::apply_role_prefix (prefix always on)."""
    st = (speaker_type or "").strip().lower()
    if st.startswith("couns"):
        return f"Counselor: {text}"
    if st.startswith("client"):
        return f"Client: {text}"
    return text


def build_items(rows, unit):
    """Reconstruct the classifier inputs in the artifact's stored order."""
    items = []
    for r in rows:
        st = r.get("speaker_type", "")
        if unit == "message":
            mc = r.get("message_classification") or {}
            txt = mc.get("text")
            if txt is None:
                continue
            items.append(dict(source_file=r.get("source_file", ""), id=str(r.get("id")),
                              msg_message_number=str(r.get("msg_message_number", "")),
                              source=r.get("source"), speaker_type=st, model=r.get("model", ""),
                              unit="message", sentence_index=-1,
                              stored_label=mc.get("predicted_label"),
                              text=txt, input_text=apply_role_prefix(txt, st)))
        else:
            for s in r.get("sentence_classification") or []:
                txt = s.get("text")
                if txt is None:
                    continue
                items.append(dict(source_file=r.get("source_file", ""), id=str(r.get("id")),
                                  msg_message_number=str(r.get("msg_message_number", "")),
                                  source=r.get("source"), speaker_type=st, model=r.get("model", ""),
                                  unit="span", sentence_index=int(s.get("sentence_index", -1)),
                                  stored_label=s.get("predicted_label"),
                                  text=txt, input_text=apply_role_prefix(txt, st)))
    return items


def main():
    args = parse_args()
    data_json = Path(args.data_json)
    rows = json.loads(data_json.read_text(encoding="utf-8"))
    print(f"{len(rows)} message records from {data_json.name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    id2label = model.config.id2label
    label_of = (lambda i: id2label[str(i)]) if (isinstance(id2label, dict) and "0" in id2label) \
        else (lambda i: id2label[i])
    print(f"model on {device}, {len(id2label)} labels")

    out_frames = []
    # Message pass first, then the span pass: the same order as the original run, so batch
    # composition (and hence padding, and hence exact float results) is reproduced.
    for unit in ("message", "span"):
        items = build_items(rows, unit)
        if args.limit:
            items = items[: args.limit]
        print(f"\n{unit}: {len(items)} items")

        recs = []
        for i in range(0, len(items), args.batch_size):
            batch = items[i: i + args.batch_size]
            enc = tokenizer([b["input_text"] for b in batch], padding=True, truncation=True,
                            return_tensors="pt")
            n_tokens = enc["attention_mask"].sum(dim=1).tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

            k = min(args.top_k, probs.shape[1])
            order = np.argsort(-probs, axis=1)[:, :k]
            for j, b in enumerate(batch):
                p = probs[j]
                top = order[j]
                ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
                recs.append({
                    "source_file": b["source_file"], "id": b["id"],
                    "msg_message_number": b["msg_message_number"], "source": b["source"],
                    "speaker_type": b["speaker_type"], "model": b["model"],
                    "unit": b["unit"], "sentence_index": b["sentence_index"],
                    "stored_label": b["stored_label"], "pred_label": label_of(int(top[0])),
                    "top1_prob": float(p[top[0]]), "top2_prob": float(p[top[1]]),
                    "margin": float(p[top[0]] - p[top[1]]), "entropy_bits": ent,
                    "top_labels": json.dumps([label_of(int(t)) for t in top], ensure_ascii=False),
                    "top_probs": json.dumps([round(float(p[t]), 6) for t in top]),
                    "n_tokens": int(n_tokens[j]), "n_chars": len(b["text"]),
                    "truncated": bool(n_tokens[j] >= tokenizer.model_max_length or n_tokens[j] >= 512),
                })
            if (i // args.batch_size) % 200 == 0:
                print(f"  {i + len(batch)}/{len(items)}", flush=True)

        df = pd.DataFrame(recs)
        agree = float((df["pred_label"] == df["stored_label"]).mean())
        print(f"  agreement with the stored labels: {agree:.5f} "
              f"({int((df['pred_label'] != df['stored_label']).sum())} disagreements)")
        print(f"  truncated at the model limit: {int(df['truncated'].sum())} "
              f"({df['truncated'].mean():.4%}); mean {df['n_tokens'].mean():.1f} tokens")
        # Report chat and mail separately: mail items are an order of magnitude longer, so a
        # pooled rate is dominated by conditions a chat-only analysis never touches.
        if "source" in df.columns:
            is_chat = df["source"].astype(str).str.endswith("_chat")
            for name, mask in (("chat", is_chat), ("mail", ~is_chat)):
                sub = df[mask]
                if len(sub):
                    print(f"    {name}: {int(sub['truncated'].sum())}/{len(sub)} "
                          f"({sub['truncated'].mean():.4%})")
        out_frames.append(df)

    out = pd.concat(out_frames, ignore_index=True)
    out_path = Path(args.out_csv)
    out.to_csv(out_path, index=False, compression="gzip")
    print(f"\nWrote {out_path} ({len(out)} rows, {out_path.stat().st_size / 1e6:.1f} MB)")

    manifest = {
        "data_json": data_json.name,
        "model_path": str(Path(args.model_path).relative_to(ROOT)),
        "model_sha256_config": hashlib.sha256(
            (Path(args.model_path) / "config.json").read_bytes()).hexdigest(),
        "batch_size": args.batch_size,
        "device": str(device),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "n_rows": int(len(out)),
        "agreement_by_unit": {u: float((g["pred_label"] == g["stored_label"]).mean())
                              for u, g in out.groupby("unit")},
        "note": "Probabilities are raw softmax and are NOT calibrated; use them only for relative "
                "between-condition comparisons, never as a probability of correctness.",
    }
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_manifest}")


if __name__ == "__main__":
    main()
