# data/

`conversations/` holds the released corpus (see the top-level README for the record format and
the masking procedure):

| File | Conversations | Condition |
|---|---|---|
| `hh_roleplay_chat.jsonl` | 68 | human counselor trainee + human playing a persona |
| `h_llm_roleplay_chat.jsonl` | 414 | human counselor trainee + LLM-simulated client |
| `REDACTION_REPORT.json` | | masking counts per condition and label |

The analysis scripts read `data/processed/combined/normalized/oncoco_classification_all.json`, one
record per message. `scripts/release/released_to_classification_json.py` rebuilds that file from the
two JSONL files above:

```bash
python scripts/release/released_to_classification_json.py
```

The rebuilt file contains the two roleplay conditions only. HH real, the reference condition of
most analyses, is not released (real help-seekers).
