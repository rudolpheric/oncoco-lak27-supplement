#!/usr/bin/env python3
"""
Apply the manual quality filter for H-LLM roleplay chats.

Removes 10 conversations identified as non-genuine (joke/trolling/test/AI-pasted)
during the sub-agent review + manual re-read (see drop_decisions.csv).
Filters BOTH the raw source (chats_filtered) and the cached labeled artifacts
(SaT-6l + regex), so no re-segmentation/re-classification is needed.

Every modified file gets a .bak_prefilter backup (idempotent: never overwrites
an existing backup). Run once.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (model, conversation_id) -> every conversation dropped by the quality review.
#
# This set is the SINGLE SOURCE OF TRUTH for the filter and is imported by
# scripts/analysis/02_chat_to_common.py, so that a rebuild from the raw exports produces
# the filtered corpus instead of silently reinstating these conversations. The two
# drop_decisions CSVs record only the 9 cases decided as DROP outright; the two marked
# BORDERLINE below were dropped on the manual re-read and exist only here.
DROPS = {
    ("GPT_OSS_120B", "48"),    # sexual harassment of client
    ("GPT_OSS_120B", "68"),    # joke name + contemptuous
    ("LLama3_3_70B", "1012"),  # sexual innuendo + "Stroh ist die antwort"
    ("LLama3_3_70B", "1024"),  # lazy "Ja/Ja/Ja" + mock blessing (borderline->drop)
    ("LLama3_3_70B", "1102"),  # "Ich leite einen Zirkus"
    ("LLama3_3_70B", "1155"),  # "Sie langweilen mich"
    ("LLama3_3_70B", "1167"),  # pasted AI text + "1" spam
    ("LLama3_3_70B", "1251"),  # explicit bot-directing (borderline->drop)
    ("LLama3_3_70B", "1560"),  # one-word mocking "wow"/"stark"
    ("LLama3_3_70B", "1565"),  # bot-testing "du bist nun geheilt"
    # QUOB26 review (quob26_drop_decisions.csv), added with that course's merge:
    ("GPT_OSS_120B", "176"),   # counselor turns B14-B26 pasted from an external chatbot
}
DROP_IDS = {i for _, i in DROPS}

# Structural drops outside the H-LLM quality review, keyed by (source, id) because
# conversation ids repeat across conditions (id 1052 exists in HH real *and* in HH
# roleplay). Same single-source-of-truth contract as DROPS: imported by
# scripts/analysis/02_chat_to_common.py so a rebuild from raw does not reinstate them.
STRUCTURAL_DROPS = {
    # One counselor message, no client turn at all, so the conversation carries no
    # client spans. It made HH real 54 conversations counselor-side and 53 client-side,
    # which put two different reference sizes in the same paper.
    ("HH_real_chat", "1052"),
}

# 425 raw H-LLM conversations minus these leaves the 414 the paper reports.
EXPECTED_HLLM_AFTER_FILTER = 414


def _keep(r: dict) -> bool:
    """False for any labeled record belonging to a dropped conversation."""
    rid = str(r.get("id"))
    if (str(r.get("model") or ""), rid) in DROPS:
        return False
    return (str(r.get("source") or ""), rid) not in STRUCTURAL_DROPS


def backup(p: Path):
    b = p.with_suffix(p.suffix + ".bak_prefilter")
    if not b.exists():
        shutil.copy2(p, b)
        return True
    return False


def filter_source():
    """Source chats_filtered/<model>/*.json : arrays of conversation objects."""
    base = ROOT / "data/raw/human_llm/chats_filtered"
    total_removed = 0
    for model in ("GPT_OSS_120B", "LLama3_3_70B", "Mixtral"):
        for f in sorted((base / model).glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            kept = [c for c in data if (model, str(c.get("id"))) not in DROPS]
            removed = len(data) - len(kept)
            if removed:
                backup(f)
                f.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
                total_removed += removed
                print(f"  [source] {model}/{f.name}: removed {removed}")
    return total_removed


def filter_json_artifact(rel):
    """Master labeled JSON: list of per-message records with model+id."""
    p = ROOT / rel
    if not p.exists():
        print(f"  [skip] {rel} (missing)")
        return 0
    rows = json.loads(p.read_text(encoding="utf-8"))
    kept = [r for r in rows if _keep(r)]
    removed = len(rows) - len(kept)
    if removed:
        backup(p)
        p.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    convs = {(str(r.get("model") or ""), str(r.get("id"))) for r in rows} - \
            {(str(r.get("model") or ""), str(r.get("id"))) for r in kept}
    print(f"  [labeled] {p.name}: removed {removed} records / {len(convs)} conversations")
    return removed


def filter_jsonl_artifact(rel):
    """Intermediate JSONL: one per-message record per line."""
    p = ROOT / rel
    if not p.exists():
        print(f"  [skip] {rel} (missing)")
        return 0
    out, removed = [], 0
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            r = json.loads(line)
            if not _keep(r):
                removed += 1
            else:
                out.append(line)
    if removed:
        backup(p)
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  [labeled] {p.name}: removed {removed} records")
    return removed


def filter_csv_gz_artifact(rel):
    """Gzipped per-span sidecar: one row per classified span, keyed like the JSON."""
    import csv
    import gzip
    p = ROOT / rel
    if not p.exists():
        print(f"  [skip] {rel} (missing)")
        return 0
    with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    kept = [r for r in rows if _keep(r)]
    removed = len(rows) - len(kept)
    if removed:
        backup(p)
        with gzip.open(p, "wt", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
    print(f"  [labeled] {p.name}: removed {removed} rows")
    return removed


if __name__ == "__main__":
    print(f"Dropping {len(DROPS)} H-LLM conversations (ids: {sorted(DROP_IDS)})")
    print(f"Dropping {len(STRUCTURAL_DROPS)} structural: {sorted(STRUCTURAL_DROPS)}\n")
    print("1) Source chats_filtered:")
    src = filter_source()
    print("\n2) Master labeled JSON (read by all result scripts):")
    filter_json_artifact("data/processed/combined/normalized/oncoco_classification_all.json")
    filter_json_artifact("data/processed/combined/normalized/oncoco_classification_all_regex.json")
    # the corrected regex corpus, read as unit "regex" by the ablation scripts; it was
    # missing from this list, so drops reached the other two corpora only
    filter_json_artifact("data/processed/combined/normalized/oncoco_classification_all_regex_v2.json")
    print("\n3) Per-span confidence sidecar:")
    filter_csv_gz_artifact("data/processed/combined/normalized/oncoco_confidence_sat.csv.gz")
    print("\n4) Intermediate JSONL (kept consistent):")
    for rel in (
        "data/processed/combined/normalized/oncoco_chat_sat6l.jsonl",
        "data/processed/combined/normalized/oncoco_chat_regex.jsonl",
    ):
        filter_jsonl_artifact(rel)
    print(f"\nDone. Source conversations removed this run: {src} "
          f"(0 on a re-run; the filter is idempotent, {len(DROPS)} drops in total).")
