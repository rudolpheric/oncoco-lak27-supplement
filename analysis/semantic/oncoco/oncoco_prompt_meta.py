#!/usr/bin/env python3
"""Per-conversation prompt and persona metadata for the H-LLM condition.

The normalized classification artifact (oncoco_classification_all.json) carries no prompt
or persona fields. Both live only in the raw platform exports under
data/raw/human_llm/chats/<MODEL>/*.json, keyed here on the same (source_file, id) pair the
analysis scripts use as conversation key.

What the raw export gives us per conversation:
  persona.name, persona.properties{Steckbrief, Hauptanliegen, Nebenanliegen,
                                   Sprachliche Merkmale, ...}
  learn_counselling_messages[].additions.prompt   the full system prompt that produced
                                                  each LLM turn (Llama and GPT-OSS only;
                                                  the Mixtral deployment did not log it)

Two prompt-side flags are derived:

  chat_template    the template family. Prompts that start with "<|begin_of_text|>" wrap the
                   persona in the Llama 3 chat format with a structured "## Antwortformat"
                   block; the others are plain-text templates ("# Rollenanweisung ..." for
                   the first turn, "Du bist ..." for later turns). The family is constant
                   within a conversation and confounded with model and semester.
  brevity_persona  whether the persona profile itself asks for short replies. Every logged
                   template already contains a brevity instruction at template level
                   ("Maximal 1-2 kurze Sätze", "1-4 Sätze maximal", "kurz und präzise"), so
                   the only brevity cue that VARIES between conversations is the persona's
                   "Sprachliche Merkmale" field (e.g. "Kurze prägnante Rückmeldungen").
                   That field is embedded verbatim in every template of the persona, so the
                   flag is persona-level.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CHATS = PROJECT_ROOT / "data" / "raw" / "human_llm" / "chats"

MODEL_DIRS = {"Mixtral": "Mixtral", "LLama3_3_70B": "LLama3_3_70B", "GPT_OSS_120B": "GPT_OSS_120B"}
MODEL_LABEL = {"Mixtral": "Mixtral 8x7B", "LLama3_3_70B": "Llama 3.3 70B", "GPT_OSS_120B": "GPT-OSS-120B"}

# Persona-level brevity cue in "Sprachliche Merkmale". Deliberately narrow: a cue must talk
# about message length, not about register ("einfache Sprache") or precision ("präzise").
BREVITY_RE = re.compile(
    r"\bkurz\w*\b|\bknapp\w*\b|prägnant|praegnant|1-2 Sätze|wenige Worte|einsilbig|wortkarg",
    re.IGNORECASE,
)

# Template families, same prefixes as scripts/release/build_release_corpus.py::strip_history
def prompt_variant(prompt: str) -> str:
    if prompt.startswith("# Rollenanweisung"):
        return "first_message"
    if prompt.startswith("Du bist"):
        return "later_turn"
    if prompt.startswith("<|begin_of_text|>"):
        return "first_message_chat_template" if "Du beginnst" in prompt else "later_turn_chat_template"
    return "unknown"


_FINAL_PUNCT = re.compile(r"[.!?…]+[\"'»“”)\]]*\s*$")


def client_word_count(text: str) -> int:
    return len(str(text or "").split())


def has_final_punct(text: str) -> bool:
    return bool(_FINAL_PUNCT.search(str(text or "").rstrip()))


def _iter_raw_conversations():
    for model_dir in sorted(RAW_CHATS.iterdir()):
        if not model_dir.is_dir():
            continue
        for fp in sorted(model_dir.glob("*.json")):
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, list):
                convs = [c for c in data if isinstance(c, dict)]
            else:
                convs = None
                for key in ("counsellings", "counselings", "conversations", "data"):
                    if isinstance(data.get(key), list):
                        convs = [c for c in data[key] if isinstance(c, dict)]
                        break
                if convs is None:
                    convs = [data]
            rel = str(fp.resolve().relative_to(PROJECT_ROOT))
            for conv in convs:
                yield model_dir.name, rel, conv


def _merkmale(persona: Optional[dict]) -> str:
    props = (persona or {}).get("properties") or {}
    v = props.get("Sprachliche Merkmale")
    if isinstance(v, list):
        return " | ".join(str(x) for x in v if x is not None)
    return "" if v is None else str(v)


def load_hllm_prompt_meta() -> pd.DataFrame:
    """One row per raw H-LLM conversation, indexed by conv_key = "<source_file>::<id>"."""
    records: List[Dict] = []
    for model, rel, conv in _iter_raw_conversations():
        cid = conv.get("id")
        if cid is None:
            continue
        persona = conv.get("persona") or {}
        merk = _merkmale(persona)
        msgs = conv.get("learn_counselling_messages") or []
        llm_msgs = [m for m in msgs if str(m.get("author", "")).lower() == "virtual_client"]
        prompts = []
        for m in llm_msgs:
            add = m.get("additions")
            p = add.get("prompt") if isinstance(add, dict) else None
            if p:
                prompts.append(p)
        variants = sorted({prompt_variant(p) for p in prompts})
        records.append(
            {
                "conv_key": f"{rel}::{cid}",
                "model": model,
                "model_label": MODEL_LABEL.get(model, model),
                "persona_id": str(conv.get("persona_id") or ""),
                "persona_name": str(persona.get("name") or ""),
                "sprachliche_merkmale": merk,
                "brevity_persona": bool(BREVITY_RE.search(merk)),
                "template_logged": bool(prompts),
                "n_prompts_logged": len(prompts),
                "chat_template": any(p.startswith("<|begin_of_text|>") for p in prompts),
                "variants": ";".join(variants),
                "n_llm_msgs": len(llm_msgs),
            }
        )
    df = pd.DataFrame(records).drop_duplicates(subset=["conv_key"]).set_index("conv_key")
    # Mixtral logged no prompts: the family is unknown there, not "plain".
    df["template_family"] = df.apply(
        lambda r: "not logged" if not r["template_logged"] else ("chat" if r["chat_template"] else "plain"),
        axis=1,
    )
    return df


if __name__ == "__main__":
    meta = load_hllm_prompt_meta()
    print(meta.groupby(["model", "template_family"]).size())
    print("\nbrevity_persona by persona name:")
    print(meta.groupby("persona_name")["brevity_persona"].agg(["first", "size"]).sort_values("size", ascending=False))
