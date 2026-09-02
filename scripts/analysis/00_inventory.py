#!/usr/bin/env python3
"""
Inventory counts for mail and chat datasets.
Outputs CSV + markdown summary in results/.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_SUMMARIES = PROJECT_ROOT / "results" / "summaries"
RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
RESULTS_SUMMARIES.mkdir(parents=True, exist_ok=True)

MAIL_MERGED = PROJECT_ROOT / "data" / "raw" / "mail" / "kia-data" / "data" / "merged" / "all_data.csv"
SAEULE4_DIR = PROJECT_ROOT / "data" / "raw" / "mail" / "kia-data" / "data" / "saeule_4"

CHAT_HH_DIR = PROJECT_ROOT / "data" / "raw" / "human_human" / "chats"
CHAT_HL_DIR = PROJECT_ROOT / "data" / "raw" / "human_llm" / "chats"


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _iter_json_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.json"):
        if "prompts" in p.parts:
            continue
        # skip notebooks caches etc.
        if p.name.startswith("."):
            continue
        yield p


def _load_json_any(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def inventory_mail() -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    notes: List[str] = []

    if MAIL_MERGED.exists():
        mail_rows = _read_csv_rows(MAIL_MERGED)
        conv_sets = defaultdict(set)
        msg_counts = Counter()
        for r in mail_rows:
            ds = r.get("dataset", "")
            msg_counts[ds] += 1
            conv_sets[ds].add(r.get("conversation_id"))
        for ds, msg_count in msg_counts.items():
            rows.append(
                {
                    "modality": "mail",
                    "dataset_group": "kia",
                    "dataset": ds,
                    "model": "",
                    "conversations": str(len(conv_sets[ds])),
                    "messages": str(msg_count),
                }
            )
    else:
        notes.append(f"Missing mail merged CSV: {MAIL_MERGED}")

    # Saeule 4 synthetic (LLM-LLM)
    if SAEULE4_DIR.exists():
        conv_count = 0
        msg_count = 0
        for p in _iter_json_files(SAEULE4_DIR):
            data = _load_json_any(p)
            if data is None:
                continue
            # Handle list or single object
            convs = data if isinstance(data, list) else [data]
            for conv in convs:
                if not isinstance(conv, dict):
                    continue
                conv_count += 1
                messages = conv.get("messages", [])
                if isinstance(messages, list):
                    msg_count += len(messages)
        if conv_count > 0:
            rows.append(
                {
                    "modality": "mail",
                    "dataset_group": "kia",
                    "dataset": "saeule_4",
                    "model": "",
                    "conversations": str(conv_count),
                    "messages": str(msg_count),
                }
            )
    else:
        notes.append(f"Missing saeule_4 dir: {SAEULE4_DIR}")

    return rows, notes


def _infer_model_from_path(path: Path) -> str:
    # model is usually the immediate parent folder for human_llm files
    if "human_llm" in path.parts:
        parent = path.parent.name
        # ignore top-level chats dir
        if parent == "chats":
            return ""
        return parent
    return ""


def _load_chat_conversations(path: Path) -> List[dict]:
    data = _load_json_any(path)
    if data is None:
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        # try common keys
        for key in ("counsellings", "counselings", "conversations", "data"):
            if key in data and isinstance(data[key], list):
                return [d for d in data[key] if isinstance(d, dict)]
        # fallback single
        return [data]
    return []


def inventory_chat() -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    notes: List[str] = []

    def scan_dir(base: Path, group: str):
        if not base.exists():
            notes.append(f"Missing chat dir: {base}")
            return
        for p in base.rglob("*.json"):
            data = _load_chat_conversations(p)
            if not data:
                continue
            conv_count = len(data)
            msg_count = 0
            for conv in data:
                msgs = conv.get("learn_counselling_messages", [])
                if isinstance(msgs, list):
                    msg_count += len(msgs)
            rows.append(
                {
                    "modality": "chat",
                    "dataset_group": group,
                    "dataset": p.name,
                    "model": _infer_model_from_path(p),
                    "conversations": str(conv_count),
                    "messages": str(msg_count),
                }
            )

    scan_dir(CHAT_HH_DIR, "human_human")
    scan_dir(CHAT_HL_DIR, "human_llm")

    return rows, notes


def write_outputs(rows: List[Dict[str, str]], notes: List[str]) -> None:
    out_csv = RESULTS_TABLES / "data_inventory.csv"
    fields = ["modality", "dataset_group", "dataset", "model", "conversations", "messages"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # markdown summary
    out_md = RESULTS_SUMMARIES / "data_inventory.md"
    lines = ["# Data inventory", "", "Generated by scripts/analysis/00_inventory.py", ""]
    if notes:
        lines.append("## Notes")
        lines.extend([f"- {n}" for n in notes])
        lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append("modality | dataset_group | dataset | model | conversations | messages")
    lines.append("---|---|---|---|---|---")
    for r in rows:
        lines.append(
            f"{r['modality']} | {r['dataset_group']} | {r['dataset']} | {r['model']} | {r['conversations']} | {r['messages']}"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = []
    notes: List[str] = []

    mail_rows, mail_notes = inventory_mail()
    rows.extend(mail_rows)
    notes.extend(mail_notes)

    chat_rows, chat_notes = inventory_chat()
    rows.extend(chat_rows)
    notes.extend(chat_notes)

    write_outputs(rows, notes)
    print(f"Wrote {len(rows)} rows to results/tables/data_inventory.csv")


if __name__ == "__main__":
    main()
