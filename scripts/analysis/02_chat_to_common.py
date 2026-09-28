#!/usr/bin/env python3
"""
Normalize chat JSON files to a common CSV schema.

SOURCE OF TRUTH (2026-08-21)
----------------------------
H-LLM is read from ``chats/<model>/``, and the quality filter is applied here.

Three things used to go wrong, and all three survived silently into a rebuild:

* the eleven conversations dropped by the quality review are still present in the raw
  export, so a rebuild reinstated them. The drop set is imported from
  ``analysis/quality_review/apply_quality_filter.py`` rather than duplicated.
* the H-LLM directory also holds unattributed platform exports at the top level
  (``Counsellings.json`` plus six ``vk_*.json``). They carry 605 conversations with
  messages, 125 of them duplicates of ids that the model subdirectories already
  contain, and they have no model attribution (``_infer_model`` returned ""). Taking
  them in inflates H-LLM from 414 to over a thousand conversations and breaks the
  per-model split of RQ1b. Only the model subdirectories are analysed.
* ``chats_filtered/`` is NOT the right source, despite the name. It is an older,
  message-level filtered export: it holds the same conversations but 114 fewer
  messages inside 28 of them. The analysed corpus was built from ``chats/``. It is
  still the right place to look up persona metadata, which is all that
  ``make_persona_table.py`` and ``tam_convergence.py`` use it for.

The HH-real role repair and the platform-boilerplate removal are NOT done here.
They are owned by ``analysis/quality_review/repair_hh_real_roles.py``, which runs on
the classification JSON and segments the client stream at the ``Beigetreten`` join
events. Dropping that boilerplate at this stage would destroy the very boundaries the
repair needs, and it would leave the reference condition's roles uncorrected.

Pipeline order: this script -> analysis/semantic/classification/classification_all.py ->
analysis/quality_review/repair_hh_real_roles.py.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAT_HH_DIR = PROJECT_ROOT / "data" / "raw" / "human_human" / "chats"
CHAT_HL_DIR = PROJECT_ROOT / "data" / "raw" / "human_llm" / "chats"
HL_MODEL_DIRS = ("GPT_OSS_120B", "LLama3_3_70B", "Mixtral")

# the quality-review drop set, imported by path (neither directory is a package)
_QF_PATH = PROJECT_ROOT / "analysis" / "quality_review" / "apply_quality_filter.py"
_QF_SPEC = importlib.util.spec_from_file_location("apply_quality_filter", _QF_PATH)
_QF = importlib.util.module_from_spec(_QF_SPEC)
_QF_SPEC.loader.exec_module(_QF)  # type: ignore[union-attr]
QUALITY_DROPS = _QF.DROPS
STRUCTURAL_DROPS = _QF.STRUCTURAL_DROPS
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "chat" / "chat_all.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Conversation counts the paper reports. A silent source-path or filter regression must
# fail here rather than three scripts downstream.
EXPECTED_CONVERSATIONS = {"HH_real_chat": 53, "HH_roleplay_chat": 68, "H_LLM_roleplay_chat": 414}


def _load_json_any(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_conversations(path: Path) -> List[dict]:
    data = _load_json_any(path)
    if data is None:
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for key in ("counsellings", "counselings", "conversations", "data"):
            if key in data and isinstance(data[key], list):
                return [d for d in data[key] if isinstance(d, dict)]
        return [data]
    return []


def _infer_model(path: Path) -> str:
    if "human_llm" in path.parts:
        parent = path.parent.name
        if parent in HL_MODEL_DIRS:
            return parent
        raise SystemExit(
            f"{path} is not inside a model subdirectory {HL_MODEL_DIRS}; H-LLM rows must carry a "
            "model attribution. Do not widen the H-LLM ingestion without fixing this.")
    return ""


def _map_role(author: str, source_file: Path) -> str:
    if not author:
        return ""
    a = str(author).strip().lower()
    # RealCounsellings exports use "user/system" author IDs with reversed semantics:
    # user = client side, system = counselor side.
    if source_file.name.lower() == "realcounsellings.json":
        if a in {"user", "virtual_client", "client", "ratsuchende"}:
            return "Ratsuchende"
        if a in {"system", "berater", "beratende", "counsellor", "counselor"}:
            return "Beratende"
        return "Beratende"
    if a in {"virtual_client", "client", "ratsuchende"}:
        return "Ratsuchende"
    return "Beratende"


def _speaker_origin(dataset_group: str, role: str) -> str:
    if dataset_group == "human_human":
        return "human"
    if dataset_group == "human_llm":
        # in H-LLM data, client is LLM, counselor is human
        if role == "Ratsuchende":
            return "llm"
        if role == "Beratende":
            return "human"
    return ""


def _iter_chat_files(base: Path, group: str) -> List[Path]:
    """JSON exports to ingest.

    For H-LLM this is deliberately restricted to the per-model subdirectories: the
    top-level exports in the same directory are unattributed supersets that overlap the
    model directories by 125 conversation ids (see the module docstring).
    """
    if not base.exists():
        raise SystemExit(f"Missing chat source directory: {base}")
    if group == "human_llm":
        paths = [p for d in HL_MODEL_DIRS for p in sorted((base / d).glob("*.json"))]
        skipped = [p.name for p in sorted(base.glob("*.json")) if not p.name.startswith(".")]
        if skipped:
            print(f"  [skip] {len(skipped)} unattributed top-level export(s): {', '.join(skipped)}")
        if not paths:
            raise SystemExit(f"No H-LLM model exports under {base}/{{{','.join(HL_MODEL_DIRS)}}}")
    else:
        paths = sorted(base.rglob("*.json"))
    return [p for p in paths if not p.name.startswith(".")]


def _condition(group: str, source_file: str) -> str:
    """Mirrors analysis/semantic/classification/classification_all.py::map_chat_condition."""
    if group == "human_llm":
        return "H_LLM_roleplay_chat"
    return "HH_real_chat" if "realcounsellings" in source_file.lower() else "HH_roleplay_chat"


def main() -> None:
    rows: List[Dict[str, str]] = []

    seen: Dict[str, set] = defaultdict(set)
    n_dropped = 0
    n_structural = 0

    for base, group in [(CHAT_HH_DIR, "human_human"), (CHAT_HL_DIR, "human_llm")]:
        for p in _iter_chat_files(base, group):
            convs = _load_conversations(p)
            if not convs:
                continue
            model = _infer_model(p)
            rel = str(p.relative_to(PROJECT_ROOT))
            for conv in convs:
                conv_id = conv.get("id", "")
                messages = conv.get("learn_counselling_messages", [])
                if not isinstance(messages, list):
                    continue
                if (model, str(conv_id)) in QUALITY_DROPS:
                    n_dropped += 1
                    continue
                cond = _condition(group, rel)
                if (cond, str(conv_id)) in STRUCTURAL_DROPS:
                    n_structural += 1
                    continue
                seen[cond].add((rel, str(conv_id)))
                conv_rows = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    author = msg.get("author", "")
                    role = _map_role(author, p)
                    conv_rows.append(
                        {
                            "conversation_id": str(conv_id),
                            "message_id": str(msg.get("message_number", "")),
                            "role": role,
                            "author": str(author),
                            "timestamp": msg.get("created_at", ""),
                            "content": msg.get("content", ""),
                            "dataset_group": group,
                            "model": model,
                            "source_file": str(p.relative_to(PROJECT_ROOT)),
                            "modality": "chat",
                            "speaker_origin": _speaker_origin(group, role),
                        }
                    )
                rows.extend(conv_rows)

    if not rows:
        raise SystemExit("No chat rows collected. Check input paths.")

    if n_dropped != len(QUALITY_DROPS):
        raise SystemExit(
            f"quality filter removed {n_dropped} of {len(QUALITY_DROPS)} declared conversations; "
            "the raw export and the drop set have diverged.")
    for condition, expected in EXPECTED_CONVERSATIONS.items():
        got = len(seen.get(condition, ()))
        if got != expected:
            raise SystemExit(
                f"{condition}: collected {got} conversations, expected {expected}. Check the "
                f"source directories and the quality filter before writing a corpus the "
                "downstream tables cannot reproduce.")
    if n_structural != len(STRUCTURAL_DROPS):
        raise SystemExit(
            f"structural filter removed {n_structural} of {len(STRUCTURAL_DROPS)} declared "
            "conversations; the raw export and the drop set have diverged.")
    print(f"quality filter: {n_dropped} conversations dropped at ingestion "
          f"({n_structural} structural)")
    print("conversation counts: " + ", ".join(
        f"{c}={len(seen[c])}" for c in sorted(seen)))
    print("NOTE: platform boilerplate and the HH-real role repair are applied downstream by "
          "analysis/quality_review/repair_hh_real_roles.py, not here.")

    fieldnames = list(rows[0].keys())
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
