#!/usr/bin/env python3
"""Turn replay outputs into a chat CSV in the chat_all.csv schema, one condition per arm.

Purpose: feed the regenerated client turns through the unchanged OnCoCo pipeline
(analysis/semantic/classification/classification_all.py --segmentation sat6l). Counselor turns are the
original human turns; client turns are the generated text of the arm. Rows carry
``dataset_group=replay:<arm>``, which classification_all.map_chat_condition maps to the
condition ``H_LLM_replay_<arm>_chat``.

Turns the generator did not produce (no logged prompt, or a failed request) keep the original
client text and are flagged in the sidecar CSV so the analysis can exclude them.

    python replay_to_chat_csv.py --outputs replay_outputs.jsonl --out replay_chat.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/quality_review"))
from apply_quality_filter import DROPS  # noqa: E402

RAW_DIR = ROOT / "data/raw/human_llm/chats/GPT_OSS_120B"
OUT_DIR = ROOT / "data/processed/prompt_replay"
MODEL = "GPT_OSS_120B"
COLUMNS = ["conversation_id", "message_id", "role", "author", "timestamp", "content",
           "dataset_group", "model", "source_file", "modality", "speaker_origin"]

_NAME_PREFIX = re.compile(r"^\s*(?:\*\*)?[A-ZÄÖÜ][\wäöüß .-]{1,30}?(?:\*\*)?\s*:\s*")


def clean(text: str, persona_name: str) -> str:
    """Strip a leading speaker label ("Boris: ...") and surrounding quotes/whitespace."""
    t = (text or "").strip()
    for n in (persona_name, "virtual_client", "Klient", "Klientin", "Client"):
        if n and t.lower().startswith(n.lower() + ":"):
            t = t[len(n) + 1:].strip()
            break
    if len(t) >= 2 and t[0] in "\"„“'" and t[-1] in "\"“”'":
        t = t[1:-1].strip()
    return t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", default=str(OUT_DIR / "replay_outputs.jsonl"))
    p.add_argument("--out", default=str(OUT_DIR / "replay_chat.csv"))
    p.add_argument("--arms", default="", help="comma list; empty = all arms present in outputs")
    p.add_argument("--raw-files", nargs="+", default=None, help="raw export JSONs; default all of RAW_DIR")
    p.add_argument("--model", default=MODEL, help="deployment model key (quality DROPS, model column)")
    p.add_argument("--group-suffix", default="",
                   help="appended to the arm in dataset_group, e.g. _llama -> condition H_LLM_replay_A_llama_chat")
    p.add_argument("--only-generated", action="store_true",
                   help="keep only conversations with at least one generated turn (smoke runs)")
    args = p.parse_args()

    gen: dict[tuple[str, str, int], dict] = {}
    with Path(args.outputs).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                gen[(r["arm"], r["conv_id"], int(r["message_number"]))] = r
    arms = sorted({a for a, _, _ in gen}) if not args.arms else [a.strip() for a in args.arms.split(",")]

    rows, flags = [], []
    files = [ROOT / f for f in args.raw_files] if args.raw_files else sorted(RAW_DIR.glob("*.json"))
    for fp in files:
        src = str(fp.relative_to(ROOT))
        for conv in json.loads(fp.read_text(encoding="utf-8")):
            cid = str(conv["id"])
            if (args.model, cid) in DROPS:
                continue
            for arm in arms:
                if args.only_generated and not any(k[0] == arm and k[1] == cid for k in gen):
                    continue
                for m in sorted(conv["learn_counselling_messages"], key=lambda m: int(m["message_number"])):
                    num = int(m["message_number"])
                    is_client = m.get("author") == "virtual_client"
                    content = m.get("content") or ""
                    status = "original"
                    if is_client:
                        g = gen.get((arm, cid, num))
                        if g is None:
                            status = "missing"
                        elif not clean(g["content"], g.get("persona_name", "")):
                            status = "empty"
                        else:
                            content = clean(g["content"], g.get("persona_name", ""))
                            status = "generated"
                        flags.append(dict(arm=arm, conv_id=cid, message_number=num, status=status,
                                          finish_reason=(g or {}).get("finish_reason", "")))
                    rows.append(dict(
                        conversation_id=cid, message_id=num,
                        role="Ratsuchende" if is_client else "Beratende",
                        author=m.get("author", ""), timestamp=m.get("created_at", ""),
                        content=content, dataset_group=f"replay:{arm}{args.group_suffix}", model=args.model,
                        source_file=src, modality="chat",
                        speaker_origin="llm" if is_client else "human",
                    ))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with out.with_name(out.stem + "_turn_status.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "conv_id", "message_number", "status", "finish_reason"])
        w.writeheader()
        w.writerows(flags)
    summary = {}
    for f in flags:
        summary.setdefault(f["arm"], {}).setdefault(f["status"], 0)
        summary[f["arm"]][f["status"]] += 1
    print(f"wrote {len(rows)} rows for arms {arms} -> {out}; client-turn status per arm: {summary}")


if __name__ == "__main__":
    main()
