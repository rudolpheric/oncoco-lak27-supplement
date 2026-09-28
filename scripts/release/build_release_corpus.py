#!/usr/bin/env python3
"""Build the releasable corpus (HH roleplay chat + H-LLM roleplay chat) from the platform exports.

Stage 1 of the data release. Reads the same exports and applies the same quality filter as
scripts/analysis/02_chat_to_common.py, attaches the OnCoCo labels of the analysed corpus to every
message, extracts the LLM prompt templates with the conversation history stripped, and writes
an UNREDACTED intermediate corpus to --work-dir (never committed). Stage 2, redact_corpus.py,
runs the PII filter over that intermediate and writes data/conversations/.

HH real chat is not part of the release: those are real help-seekers.

Usage (from the working tree that holds the raw exports):
    python scripts/release/build_release_corpus.py --project-root <working tree> --work-dir <scratch>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HL_MODEL_DIRS = ["Mixtral", "LLama3_3_70B", "GPT_OSS_120B"]
HH_FILES = ["E01.json", "E02.json"]
MODEL_LABEL = {"Mixtral": "Mixtral 8x7B", "LLama3_3_70B": "Llama 3.3 70B", "GPT_OSS_120B": "GPT-OSS-120B"}

HISTORY = "{{CONVERSATION_HISTORY}}"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def role_of(author: str) -> str:
    return "Client" if str(author or "").strip().lower() in {"virtual_client", "client", "ratsuchende"} else "Counselor"


# ---------------------------------------------------------------- prompt templates
def strip_history(prompt: str) -> tuple[str, str]:
    """Return (variant, template) with the conversation history replaced by a placeholder."""
    if prompt.startswith("# Rollenanweisung"):
        assert "virtual_client:" not in prompt and "Gesprächsverlauf" not in prompt
        return "first_message", prompt
    if prompt.startswith("Du bist"):
        m = re.search(r"(Kontext der Beratungssitzung:)(.*?)(\n+Deine Aufgabe:)", prompt, re.S)
        assert m, prompt[:100]
        return "later_turn", prompt[: m.start(2)] + " " + HISTORY + prompt[m.end(2):]
    if prompt.startswith("<|begin_of_text|>"):
        if "Du beginnst" in prompt:
            assert "Gesprächsverlauf" not in prompt
            return "first_message_chat_template", prompt
        m = re.search(r"(## Bisheriger Gesprächsverlauf:\n)(.*?)(\n+## Verboten:)", prompt, re.S)
        assert m, prompt[:100]
        return "later_turn_chat_template", prompt[: m.start(2)] + HISTORY + prompt[m.end(2):]
    raise ValueError("unknown prompt family: " + prompt[:80])


class Templates:
    def __init__(self) -> None:
        self.by_text: dict[str, dict] = {}

    def register(self, prompt: str, persona: str, model: str) -> str:
        variant, tpl = strip_history(prompt)
        assert "\nvikl:" not in tpl and "\nuser:" not in tpl, "history leaked into template"
        key = hashlib.sha1(tpl.encode()).hexdigest()[:10]
        rec = self.by_text.setdefault(tpl, dict(key=key, variant=variant, persona=persona,
                                                models=collections.Counter(), n_messages=0))
        rec["models"][model] += 1
        rec["n_messages"] += 1
        return key

    def finalize(self, out_dir: Path) -> dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        order = sorted(self.by_text.items(),
                       key=lambda kv: (kv[1]["persona"].lower(), kv[1]["variant"], -kv[1]["n_messages"]))
        key2id, index = {}, []
        for n, (tpl, rec) in enumerate(order, 1):
            tid = f"T{n:03d}"
            key2id[rec["key"]] = tid
            models = {MODEL_LABEL[m]: c for m, c in rec["models"].items()}
            header = (f"---\ntemplate_id: {tid}\nvariant: {rec['variant']}\npersona: {rec['persona']}\n"
                      f"used_by_models: {json.dumps(models, ensure_ascii=False)}\n"
                      f"n_messages_generated: {rec['n_messages']}\n---\n\n")
            (out_dir / f"{tid}.md").write_text(header + tpl, encoding="utf-8")
            index.append(dict(template_id=tid, variant=rec["variant"], persona=rec["persona"],
                              used_by_models=models, n_messages_generated=rec["n_messages"]))
        (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
        return key2id


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True, help="working tree with data/raw and data/processed")
    ap.add_argument("--work-dir", required=True, help="scratch dir for the UNREDACTED intermediate")
    args = ap.parse_args()
    root, work = Path(args.project_root).resolve(), Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    qf = load_module(REPO / "analysis/quality_review/apply_quality_filter.py", "apply_quality_filter")
    DROPS = qf.DROPS

    # OnCoCo labels of the analysed corpus, keyed like 02_chat_to_common keys conversations.
    labels: dict[tuple, dict] = {}
    for r in json.loads((root / "data/processed/combined/normalized/oncoco_classification_all.json").read_text()):
        if r["source"] not in ("HH_roleplay_chat", "H_LLM_roleplay_chat"):
            continue
        labels[(r["source_file"], str(r["id"]), str(r["msg_message_number"]))] = r
    label_convs = {k[:2] for k in labels}

    templates = Templates()
    course_codes: dict[str, str] = {}
    export_codes: dict[str, str] = {}
    persona_names: dict[str, set] = collections.defaultdict(set)
    stats = collections.Counter()
    content_mismatch = []
    out: dict[str, list] = {"HH_roleplay_chat": [], "H_LLM_roleplay_chat": []}

    sources = [(root / "data/raw/human_human/chats" / f, "HH_roleplay_chat", "") for f in HH_FILES]
    for m in HL_MODEL_DIRS:
        sources += [(p, "H_LLM_roleplay_chat", m) for p in sorted((root / "data/raw/human_llm/chats" / m).glob("*.json"))]

    for path, condition, model in sources:
        rel = str(path.relative_to(root))
        export_codes.setdefault(rel, f"E{len(export_codes) + 1:02d}")
        for conv in json.loads(path.read_text(encoding="utf-8")):
            cid = str(conv["id"])
            msgs = conv.get("learn_counselling_messages") or []
            if not msgs:
                stats["skipped_empty"] += 1
                continue
            if (model, cid) in DROPS:
                stats["dropped_quality_filter"] += 1
                continue
            if (rel, cid) not in label_convs:
                stats["not_in_analysed_corpus"] += 1
                continue
            persona = conv.get("persona") if isinstance(conv.get("persona"), dict) else {}
            pname = (persona.get("name") or "").strip()
            cm = conv.get("course_member") if isinstance(conv.get("course_member"), dict) else {}
            course = ((cm.get("course") or {}) if isinstance(cm, dict) else {}).get("name") or ""
            if course:
                course_codes.setdefault(course, f"C{len(course_codes) + 1:02d}")
            rec = dict(
                conversation_id=f"{export_codes[rel]}-{cid}",
                condition=condition,
                model=MODEL_LABEL.get(model, ""),
                course_id=course_codes.get(course, ""),
                created_at=conv.get("created_at"),
                persona=dict(name=pname, profile=persona.get("properties") or {}),
                messages=[],
            )
            if pname:
                persona_names[rec["conversation_id"]].add(pname)
            for msg in msgs:
                content = msg.get("content") or ""
                role = role_of(msg.get("author"))
                lab = labels.get((rel, cid, str(msg.get("message_number"))))
                item = dict(
                    message_number=int(msg["message_number"]),
                    role=role,
                    speaker_origin="llm" if (condition == "H_LLM_roleplay_chat" and role == "Client") else "human",
                    created_at=msg.get("created_at"),
                    content=content,
                )
                add = msg.get("additions") or {}
                if condition == "H_LLM_roleplay_chat" and role == "Client":
                    li = add.get("llm_info") or {}
                    if li.get("model"):
                        item["llm_model_id"] = li["model"]
                    if add.get("prompt"):
                        item["prompt_template_key"] = templates.register(add["prompt"], pname, model)
                if lab is not None:
                    if lab["msg_content"] != content:
                        content_mismatch.append(rec["conversation_id"])
                    item["oncoco_message_label"] = (lab.get("message_classification") or {}).get("predicted_label")
                    item["oncoco_spans"] = [dict(start=s["start_offset"], end=s["end_offset"], label=s["predicted_label"])
                                            for s in lab.get("sentence_classification") or []]
                    stats["messages_labelled"] += 1
                else:
                    stats["messages_unlabelled"] += 1
                rec["messages"].append(item)
            out[condition].append(rec)
            stats[condition] += 1

    key2id = templates.finalize(REPO / "prompts/templates")
    for recs in out.values():
        for rec in recs:
            for item in rec["messages"]:
                if "prompt_template_key" in item:
                    item["prompt_template_id"] = key2id[item.pop("prompt_template_key")]

    for condition, recs in out.items():
        p = work / f"{condition.lower()}.unredacted.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # local-only mappings (identifying): never copied into the repository
    (work / "LOCAL_ONLY_code_mappings.json").write_text(json.dumps(
        dict(export_codes=export_codes, course_codes=course_codes), ensure_ascii=False, indent=1), encoding="utf-8")
    (work / "persona_names_by_conversation.json").write_text(json.dumps(
        {k: sorted(v) for k, v in persona_names.items()}, ensure_ascii=False), encoding="utf-8")

    print(dict(stats))
    print("templates:", len(key2id), "| content mismatches:", len(set(content_mismatch)))
    print("export codes:", export_codes)
    print("course codes:", course_codes)


if __name__ == "__main__":
    main()
