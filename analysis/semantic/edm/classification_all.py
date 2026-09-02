#!/usr/bin/env python3
"""
Run sentence-level OnCoCo classification over combined chat + mail datasets.

Inputs:
  - data/processed/chat/chat_all.csv
  - data/processed/mail/mail_all_with_saeule4.csv

Output:
  - data/processed/combined/normalized/oncoco_classification_all.json

This mirrors the structure of edm_classification_merged.json but uses
OnCoCo labels and adds condition metadata for extended analyses.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OnCoCo classification for chat+mail.")
    parser.add_argument(
        "--chat_csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "chat" / "chat_all.csv"),
        help="Normalized chat CSV path.",
    )
    parser.add_argument(
        "--mail_csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "mail" / "mail_all_with_saeule4.csv"),
        help="Merged mail CSV path.",
    )
    parser.add_argument(
        "--include_chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include chat messages.",
    )
    parser.add_argument(
        "--include_mail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include mail messages.",
    )
    parser.add_argument(
        "--model_path",
        default=str(PROJECT_ROOT / "analysis" / "models" / "oncoco" / "xlm-roberta-large-OnCoCo-DE-EN"),
        help="Local OnCoCo model path.",
    )
    parser.add_argument(
        "--segmentation",
        choices=["regex", "sat6l"],
        default="regex",
        help="Sentence segmentation method.",
    )
    parser.add_argument(
        "--sat_model",
        default="segment-any-text/sat-6l",
        help="Base SaT model name or HF path.",
    )
    parser.add_argument(
        "--lora_path",
        default=str(PROJECT_ROOT / "analysis" / "models" / "segmentation" / "adapter"),
        help="Path to LoRA adapter for SaT.",
    )
    parser.add_argument("--sat_threshold", type=float, default=0.5)
    parser.add_argument("--min_span_len", type=int, default=1)
    parser.add_argument(
        "--use_role_prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Prefix inputs with "Counselor:" or "Client:" before classification.',
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "combined" / "normalized" / "oncoco_classification_all.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output_jsonl",
        default="",
        help="Optional JSONL output path (appends per message).",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_rows", type=int, default=0, help="Limit rows for quick test.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N rows.")
    parser.add_argument("--limit", type=int, default=0, help="Process only N rows after offset.")
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def map_mail_condition(dataset: str) -> str:
    ds = dataset.lower()
    if "säule_1" in ds or "saeule_1" in ds:
        return "HH_roleplay_mail"
    if "säule_3" in ds or "saeule_3" in ds:
        return "HH_real_mail"
    if "säule_5" in ds or "saeule_5" in ds:
        return "HH_CAIA_mail"
    if "säule_4" in ds or "saeule_4" in ds:
        return "LLM_LLM_mail"
    return "mail_unknown"


def map_chat_condition(dataset_group: str, source_file: str) -> str:
    if dataset_group == "human_human":
        src = (source_file or "").lower()
        if "realcounsellings" in src:
            return "HH_real_chat"
        return "HH_roleplay_chat"
    if dataset_group == "human_llm":
        return "H_LLM_roleplay_chat"
    return "chat_unknown"


def role_to_speaker_type(role: str) -> str:
    if role == "Ratsuchende":
        return "Client"
    if role == "Beratende":
        return "Counsellor"
    return ""


def apply_role_prefix(text: str, speaker_type: str, use_role_prefix: bool) -> str:
    if not use_role_prefix:
        return text
    if not text:
        return text
    st = (speaker_type or "").lower()
    if st.startswith("counsellor") or st.startswith("counselor"):
        return f"Counselor: {text}"
    if st.startswith("client"):
        return f"Client: {text}"
    return text


def split_sentences_with_offsets(text: str) -> List[Tuple[int, int, str]]:
    if not isinstance(text, str) or not text.strip():
        return []
    sentences: List[Tuple[int, int, str]] = []
    # Simple regex-based splitter that keeps offsets.
    # NOTE: this class previously read r"[^.!?\\n]+[.!?]?". In a raw string, "\\n" is
    # backslash + the letter "n", so the negated class excluded the letter "n" and split
    # German text on every "n" (~16 spans/message, mean 10 chars). Fixed 2026-08-09.
    for match in re.finditer(r"[^.!?\n]+[.!?]?", text, flags=re.MULTILINE):
        start, end = match.start(), match.end()
        segment = match.group()
        if not segment.strip():
            continue
        # Trim whitespace and adjust offsets
        lstrip_len = len(segment) - len(segment.lstrip())
        rstrip_len = len(segment) - len(segment.rstrip())
        s = start + lstrip_len
        e = end - rstrip_len
        sent = text[s:e]
        if sent.strip():
            sentences.append((s, e, sent))
    return sentences


def load_sat_with_lora(model_name: str, lora_path: str, threshold: float):
    # Lazy import to avoid dependency unless needed.
    from wtpsplit.models import SubwordXLMConfig, SubwordXLMForTokenClassification
    from transformers import AutoTokenizer
    from tokenizers import AddedToken

    try:
        import adapters
        from adapters.models import MODEL_MIXIN_MAPPING
        from adapters.models.bert.mixin_bert import BertModelAdaptersMixin
        MODEL_MIXIN_MAPPING["SubwordXLMRobertaModel"] = BertModelAdaptersMixin
    except ImportError as exc:
        raise ImportError("adapters library required for LoRA. Install: pip install adapters") from exc

    config = SubwordXLMConfig.from_pretrained(model_name)
    model = SubwordXLMForTokenClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )
    model.config.base_model = "xlm-roberta-base"

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\\n")]})

    original_model_type = model.config.model_type
    model.config.model_type = "xlm-roberta"
    adapters.init(model)
    model.load_adapter(lora_path, set_active=True)
    model.config.model_type = original_model_type

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return LoRASplitter(model, tokenizer, threshold=threshold)


class LoRASplitter:
    def __init__(self, model, tokenizer, threshold: float = 0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.device = next(model.parameters()).device

    def _split_window(self, text: str) -> Tuple[List[str], int, int]:
        """Boundaries inside one <=512-token window.

        Returns (segments, characters consumed up to the last boundary, characters the
        window covered at all). The two differ for the trailing fragment after the last
        boundary, which is deliberately left for the next window.
        """
        encoding = self.tokenizer(
            text, truncation=True, max_length=512,
            return_tensors="pt", return_offsets_mapping=True
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, :, 0]
            probs = torch.sigmoid(logits).cpu().numpy()

        segments: List[str] = []
        current_start = 0
        covered = 0
        for prob, (start, end) in zip(probs, offset_mapping):
            if start == end:
                continue
            covered = max(covered, end)
            if prob > self.threshold and end > current_start:
                sentence = text[current_start:end].strip()
                if sentence:
                    segments.append(sentence)
                current_start = end
        consumed = current_start if segments else 0
        return segments, consumed, covered

    def split(self, text: str) -> List[str]:
        """Segment `text`, processing it in <=512-token windows.

        The model input is capped at 512 tokens. This used to run once over the truncated
        input and then append everything past the last boundary as a single span, so any
        message longer than the window came out with one unsegmented tail. Chat is barely
        affected (0.04-0.25% of messages exceed ~1,200 characters) but 36-58% of mail
        messages do, so the mail conditions were partly unsegmented. We now slide the
        window over the remainder instead. Changed 2026-08-21.
        """
        if not text.strip():
            return []

        sentences: List[str] = []
        pos = 0
        while pos < len(text):
            chunk = text[pos:]
            segments, consumed, covered = self._split_window(chunk)
            sentences.extend(segments)
            if consumed <= 0:
                # no boundary inside this window: emit what the window covered and move on
                take = covered if covered > 0 else len(chunk)
                tail = chunk[:take].strip()
                if tail:
                    sentences.append(tail)
                consumed = take
            if consumed <= 0:  # defensive: never loop forever
                break
            pos += consumed
        if not sentences:
            return [text.strip()]
        return sentences


def split_sentences_with_sat(text: str, splitter, min_span_len: int) -> List[Tuple[int, int, str]]:
    if not isinstance(text, str) or not text.strip():
        return []
    segments = splitter.split(text)
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for seg in segments:
        seg = seg.strip()
        if not seg or len(seg) < min_span_len:
            cursor += len(seg)
            continue
        try:
            start = text.index(seg, cursor)
        except ValueError:
            cursor += len(seg)
            continue
        end = start + len(seg)
        cursor = end
        spans.append((start, end, seg))
    return spans


def classify_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int,
) -> List[str]:
    labels: List[str] = []
    id2label = model.config.id2label
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        labels.extend([id2label[str(p)] if isinstance(id2label, dict) and str(p) in id2label else id2label[p] for p in preds])
    return labels


def main() -> None:
    args = parse_args()
    chat_csv = Path(args.chat_csv)
    mail_csv = Path(args.mail_csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.include_chat and not chat_csv.exists():
        raise FileNotFoundError(f"Missing chat CSV: {chat_csv}")
    if args.include_mail and not mail_csv.exists():
        raise FileNotFoundError(f"Missing mail CSV: {mail_csv}")

    chat_rows = load_csv(chat_csv) if args.include_chat else []
    mail_rows = load_csv(mail_csv) if args.include_mail else []

    # Normalize to unified message rows
    rows: List[Dict[str, str]] = []

    for r in chat_rows:
        condition = map_chat_condition(r.get("dataset_group", ""), r.get("source_file", ""))
        rows.append(
            {
                "id": r.get("conversation_id", ""),
                "title": "",
                "source": condition,
                "modality": "chat",
                "dataset": r.get("dataset_group", ""),
                "model": r.get("model", ""),
                "msg_learn_counselling_id": r.get("conversation_id", ""),
                "msg_message_number": r.get("message_id", ""),
                "msg_content": r.get("content", ""),
                "msg_author": r.get("author", ""),
                "msg_created_at": r.get("timestamp", ""),
                "speaker_type": role_to_speaker_type(r.get("role", "")),
                "speaker_origin": r.get("speaker_origin", ""),
                "source_file": r.get("source_file", ""),
            }
        )

    for r in mail_rows:
        condition = map_mail_condition(r.get("dataset", ""))
        rows.append(
            {
                "id": r.get("conversation_id", ""),
                "title": r.get("subject", ""),
                "source": condition,
                "modality": "mail",
                "dataset": r.get("dataset", ""),
                "model": "",
                "msg_learn_counselling_id": r.get("conversation_id", ""),
                "msg_message_number": r.get("message_id", ""),
                "msg_content": r.get("content", ""),
                "msg_author": r.get("role", ""),
                "msg_created_at": r.get("timestamp", ""),
                "speaker_type": role_to_speaker_type(r.get("role", "")),
                "speaker_origin": r.get("generated_by", ""),
                "source_file": r.get("source_file", ""),
            }
        )

    # offset first, then limit, then the max_rows cap. Applying max_rows first truncated the
    # list before the offset was taken, so --offset combined with --max_rows silently
    # processed the wrong slice (or nothing at all).
    if args.offset and args.offset > 0:
        rows = rows[args.offset:]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    # Load OnCoCo model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Optional SaT segmentation
    splitter = None
    if args.segmentation == "sat6l":
        sat_model = args.sat_model
        if "/" not in sat_model:
            sat_model = f"segment-any-text/{sat_model}"
        splitter = load_sat_with_lora(sat_model, args.lora_path, args.sat_threshold)

    # Message-level classification
    texts = [
        apply_role_prefix(r["msg_content"], r.get("speaker_type", ""), args.use_role_prefix)
        for r in rows
    ]
    msg_labels = classify_texts(texts, tokenizer, model, device, args.batch_size)

    # Sentence-level classification (flatten for batching)
    sentence_records: List[Tuple[int, int, int, int, str]] = []
    for idx, r in enumerate(rows):
        if splitter is not None:
            spans = split_sentences_with_sat(r["msg_content"], splitter, args.min_span_len)
        else:
            spans = split_sentences_with_offsets(r["msg_content"])
        for sent_idx, (start, end, sent) in enumerate(spans):
            sentence_records.append((idx, sent_idx, start, end, sent))

    sent_labels: List[str] = []
    if sentence_records:
        sent_texts = [
            apply_role_prefix(rec[4], rows[rec[0]].get("speaker_type", ""), args.use_role_prefix)
            for rec in sentence_records
        ]
        sent_labels = classify_texts(sent_texts, tokenizer, model, device, args.batch_size)

    # Attach classifications
    sentence_groups: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for (rec, label) in zip(sentence_records, sent_labels):
        msg_idx, sent_idx, start, end, sent = rec
        sentence_groups[msg_idx].append(
            {
                "sentence_index": sent_idx,
                "start_offset": start,
                "end_offset": end,
                "text": sent,
                "predicted_label": label,
            }
        )

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else None
    if output_jsonl:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("a", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                record = {
                    **r,
                    "message_classification": {
                        "text": r["msg_content"],
                        "predicted_label": msg_labels[i] if i < len(msg_labels) else None,
                    },
                    "sentence_classification": sentence_groups.get(i, []),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Appended {len(rows)} rows to {output_jsonl}")
    else:
        output_rows: List[Dict[str, object]] = []
        for i, r in enumerate(rows):
            output_rows.append(
                {
                    **r,
                    "message_classification": {
                        "text": r["msg_content"],
                        "predicted_label": msg_labels[i] if i < len(msg_labels) else None,
                    },
                    "sentence_classification": sentence_groups.get(i, []),
                }
            )
        output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
