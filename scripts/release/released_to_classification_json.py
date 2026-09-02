#!/usr/bin/env python3
"""Rebuild the analysis scripts' input from the released corpus.

The scripts under analysis/ read data/processed/combined/normalized/oncoco_classification_all.json,
a list with one record per message. This script writes that file from data/conversations/*.jsonl
for the two released conditions (HH roleplay chat, H-LLM roleplay chat). Offsets in
``sentence_classification`` refer to the released, masked text.

    python scripts/release/released_to_classification_json.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/conversations"
OUT = REPO / "data/processed/combined/normalized/oncoco_classification_all.json"

MODEL_CODE = {"Mixtral 8x7B": "Mixtral", "Llama 3.3 70B": "LLama3_3_70B", "GPT-OSS-120B": "GPT_OSS_120B", "": ""}
DATASET = {"HH_roleplay_chat": "human_human", "H_LLM_roleplay_chat": "human_llm"}
SPEAKER = {"Client": "Client", "Counselor": "Counsellor"}  # the analyses use the British spelling


def main() -> None:
    rows = []
    for fn in ["hh_roleplay_chat.jsonl", "h_llm_roleplay_chat.jsonl"]:
        for line in (SRC / fn).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            conv = json.loads(line)
            export, _, cid = conv["conversation_id"].partition("-")
            for m in conv["messages"]:
                text = m["content"]
                rows.append(dict(
                    id=cid,
                    title="",
                    source=conv["condition"],
                    modality="chat",
                    dataset=DATASET[conv["condition"]],
                    model=MODEL_CODE[conv.get("model", "")],
                    course_id=conv.get("course_id", ""),
                    msg_learn_counselling_id=cid,
                    msg_message_number=str(m["message_number"]),
                    msg_content=text,
                    msg_author="vikl" if m["role"] == "Client" else "user",
                    msg_created_at=m.get("created_at"),
                    speaker_type=SPEAKER[m["role"]],
                    speaker_origin=m["speaker_origin"],
                    source_file=f"released/{export}.jsonl",
                    message_classification=dict(text=text, predicted_label=m.get("oncoco_message_label")),
                    sentence_classification=[
                        dict(sentence_index=i, start_offset=s["start"], end_offset=s["end"],
                             text=text[s["start"]:s["end"]], predicted_label=s["label"])
                        for i, s in enumerate(m.get("oncoco_spans", []))],
                ))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)} message records to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
