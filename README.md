# Companion repository: Measuring Counseling Interaction Patterns Across Human–Human and Human–LLM Dialogues

Anonymized companion repository for a LAK 2027 Research Track submission (Journal of Learning Analytics format).
It contains the Supplementary Material referenced from the main text, the two roleplay corpora with personal data masked and OnCoCo labels attached, the prompt templates of the simulated client, the analysis code that produced every table and figure in the paper, and the aggregated result tables the code writes.

It does **not** contain the real-counseling conversations (HH real), survey responses, or classifier weights. See [Data availability](#data-availability).

## Contents

| Path | What it is |
|---|---|
| `supplementary_material.pdf` | Supplementary Material (Tables S1–S31, Figures S1–S11, Sections S1–S12) as cited in the main text |
| `oncoco_supplementary.tex`, `JLA_article.cls`, `header.png` | LaTeX source of the supplement; compiles from the repository root (see [Building the supplement](#building-the-supplement)) |
| `data/conversations/hh_roleplay_chat.jsonl` | 68 human–human roleplay chats (trainee counselor, trainee playing a persona), masked, labelled |
| `data/conversations/h_llm_roleplay_chat.jsonl` | 414 human–LLM roleplay chats (trainee counselor, LLM-simulated client), masked, labelled |
| `prompts/templates/` | 70 prompt templates of the simulated client (one per persona × variant), conversation history replaced by a placeholder; `index.json` lists them |
| `results/tables/*.tex` | Every table `\input` by the main paper or the supplement |
| `results/tables/*.csv` | Aggregated numeric outputs of the analysis scripts |
| `results/figures/oncoco/` | Every figure included by the main paper or the supplement, plus the TikZ sources of the composite figures |
| `analysis/models/segmentation/` | LoRA adapter for the SaT-6l segmenter used to build the spans, and the script that trained it |
| `analysis/`, `scripts/analysis/` | Analysis code, corpus normalization, quality filter |
| `scripts/release/` | The two scripts that produced `data/conversations/` from the platform exports (extraction, then PII masking) |

## The released conversations

One JSON object per line, one line per conversation:

```json
{
  "conversation_id": "E10-7",
  "condition": "H_LLM_roleplay_chat",
  "model": "GPT-OSS-120B",
  "course_id": "C11",
  "created_at": "2025-12-09T14:18:00.000000Z",
  "persona": {"name": "Jessica Bergmann", "profile": {"Steckbrief": {"Alter": 17, "...": "..."}, "Hauptanliegen": "...", "Nebenanliegen": ["..."], "Sprachliche Merkmale": ["..."]}},
  "messages": [
    {
      "message_number": 1,
      "role": "Client",
      "speaker_origin": "llm",
      "created_at": "2025-12-09T14:18:09.000000Z",
      "content": "Guten Tag... Ich bin mir nicht sicher, ob ich hier richtig bin.",
      "llm_model_id": "openai/gpt-oss-120b",
      "prompt_template_id": "T031",
      "oncoco_message_label": "CL-FB-*-*-*-*",
      "oncoco_spans": [{"start": 0, "end": 12, "label": "CL-FB-*-*-*-*"}, {"start": 13, "end": 63, "label": "CL-IF-ACP-*-DPD-*"}],
      "pii_placeholders": {"private_person": 1}
    }
  ]
}
```

* `conversation_id` is `<export>-<platform id>`; exports and courses are coded (`E01`…, `C01`…) because their names identify institutions and semesters. The course codes are the unit of the RQ3 analysis. The same codes stand in for export file names, course names and survey sheets in the analysis scripts and the aggregated CSVs (for example `data/raw/human_llm/chats/LLama3_3_70B/E04.json`, `survey_C09`). Exports that predate model logging and are not part of the release appear as `L01`…`L06`.
* `role` is `Client` or `Counselor`; `speaker_origin` is `human` or `llm`. In H-LLM the client is the LLM and the counselor the trainee.
* `oncoco_spans` are the spans of the analysed corpus (SaT-6l + LoRA segmentation, XLM-RoBERTa-large OnCoCo classifier) with character offsets **into the released, masked `content`**. `oncoco_message_label` is the message-level label. All 17,888 messages of the analysed corpus are present and labelled. `analysis/semantic/oncoco/label_text_map.json` maps label codes to readable names.
* `prompt_template_id` (LLM turns only) points to `prompts/templates/<id>.md`, the exact instruction the platform sent for that turn minus the conversation history. Mixtral 8x7B turns carry no template because the platform did not log prompts in that deployment.
* `pii_placeholders` appears on messages in which something was masked.

Quality filter and counts are those of the paper: 11 of 425 H-LLM conversations were dropped by the manual quality review (`analysis/quality_review/`), 414 remain; HH roleplay has 68.

### How personal data was masked

`scripts/release/redact_corpus.py` runs the open-weights token classifier **openai/privacy-filter** (Apache 2.0, https://huggingface.co/openai/privacy-filter) over every conversation and replaces detected spans by typed placeholders: `<PRIVATE_PERSON>`, `<PRIVATE_DATE>`, `<PRIVATE_ADDRESS>`, `<PRIVATE_EMAIL>`, `<PRIVATE_PHONE>`, `<PRIVATE_URL>`, `<ACCOUNT_NUMBER>`, `<SECRET>`. The OnCoCo span offsets are shifted through every replacement, so labels still align.

Detection runs twice and the spans are unioned: once over the whole conversation, so the model sees who introduced a name, and once over overlapping blocks of eight messages, a second look with local context that catches mentions the long-context pass misses. The model is trained mainly on English and over-detects German common nouns as names, so four rules are applied on top of its output:

* **Allowlist.** Tokens of the conversation's persona name, persona profile and prompt templates are never masked. The personas are fictional and published in Supplementary Table S1 and in `prompts/`, and the fictional relatives named in a persona story (“Max”, “Jan”) stay readable. Everything else the model labels as a person is masked, which includes trainees who introduce themselves by name.
* **Stoplist.** `scripts/release/redaction_stoplist.json` lists the strings that a manual review of every masked string of the first run found to be no personal data at all: role nouns, greetings, common nouns, welfare organisations.
* **Token level.** A person span is masked token by token, so “LG Jenny” becomes “LG <PRIVATE_PERSON>”. Spans found only by the block pass must look like a name (one to three capitalized tokens).
* **Consistency.** A string masked as a person, e-mail, phone, address, URL, account number or secret anywhere in a conversation is masked wherever else it occurs verbatim in that conversation.

Masking counts are in `data/conversations/REDACTION_REPORT.json`. Residual risk remains for names the model did not recognize, and readers who spot one are asked to report it. Place names and institutions are not among the model's categories. The 22 mentions that would reveal where the study ran were masked by hand in a final step, as `<PRIVATE_ADDRESS>` or `<PRIVATE_URL>` (`site_rule` in the report). Other place names were left in the text.

Prompts and persona profiles are fictional and were not masked.

## Pipeline overview

The main paper's Figure 2 describes the pipeline. The code follows the same steps.

1. **Corpus normalization.** `scripts/analysis/02_chat_to_common.py` reads the platform exports and writes one common chat table. It applies the manual quality filter whose decisions are documented in `analysis/quality_review/` (`apply_quality_filter.py` holds the single source of truth for the dropped conversations; the two `*drop_decisions.csv` files record the review rationale).
2. **Segmentation.** Messages are split into spans with SaT-6l (`segment-any-text/sat-6l`, wtpsplit) and the LoRA adapter in `analysis/models/segmentation/adapter/` (r=128, α=256 on the attention q/v matrices, `adapters` library). `analysis/models/segmentation/train_oncoco_aware.py` is the self-supervised training script that produced the adapter. The segmentation call lives in `classification_all.py` (`--segmentation sat6l`).
3. **Classification.** `analysis/semantic/classification/classification_all.py` labels each span with the XLM-RoBERTa-large classifier fine-tuned on the OnCoCo scheme (66 fine-grained categories). The study used its own checkpoint of this classifier, with 68 output units, two of them counselor codes outside the 66-category scheme that are never predicted in the study corpora. It is not identical to the checkpoint published on Hugging Face as `xlm-roberta-large-online-counseling-oncoco` (66 output units, different weights), so the paper's numbers reproduce only with the study checkpoint. `analysis/quality_review/repair_hh_real_roles.py` then repairs the speaker roles of the archived real-counseling exports at platform join events.
4. **Analyses.** Everything under `analysis/semantic/oncoco/` and `analysis/technology_acceptance/` reads the classified corpus (or the CSVs derived from it) and writes to `results/tables/` and `results/figures/oncoco/`.

The released JSONL is the classified corpus for the two roleplay conditions, restricted to the fields the analyses use; `data/README.md` shows how to rebuild the scripts' input format from it.

### Script to output map

| Script | Paper / supplement element | Main outputs |
|---|---|---|
| `oncoco_sentence_analysis.py` | label distributions, temporal deciles | `oncoco_label_distribution.csv`, `oncoco_temporal_distribution.csv` |
| `oncoco_rq_stats.py` | RQ1 distances and per-model realness | `oncoco_rq1_distances.csv`, `oncoco_rq1_label_effects.csv`, `oncoco_rq1b_model_realness.csv` |
| `oncoco_noise_band_analysis.py` | Figure 2, Table S12 (split-half noise band, size-matched null, Cramér's V, FDR on transitions) | `oncoco_noise_band.csv`, `oncoco_transition_fdr.csv` |
| `oncoco_rq1b_bootstrap.py` | Table S13 (conversation-level bootstrap per model) | `oncoco_rq1b_model_realness_bootstrap.{csv,tex}` |
| `oncoco_unit_ablation.py`, `oncoco_seg_ordering_bootstrap.py` | Table S20 and Table S9 (unit of analysis: spans, sentences, messages) | `oncoco_unit_ablation.{csv,tex}`, `oncoco_unit_noise_band.tex`, `oncoco_seg_ordering_bootstrap.csv` |
| `oncoco_rq1_segmentation_ablation.py`, `oncoco_rq2_segmentation_effects.py` | misspecified-segmentation row of Table S20 | `oncoco_rq1_segmentation_ablation.csv`, `oncoco_rq2_segmentation_*.csv` |
| `oncoco_sequence_analysis.py` | transition matrices, Figures S1–S3, Table S14 | `oncoco_transition_matrix.csv`, `oncoco_transition_top.csv`, `transition_*_chat.pdf` |
| `oncoco_hmm_model_selection.py`, `oncoco_hmm_final_fit.py` | Figure 3, Tables S5–S6 (best-of-20 restarts, k sweep) | `oncoco_hmm_model_selection.csv`, `oncoco_hmm_emissions*.csv`, `oncoco_hmm_transitions*.csv` |
| `oncoco_within_condition_analysis.py` | Tables S2, S7, Figures S6–S7 | `oncoco_within_condition_jsd*.csv`, `within_jsd_*_chat.pdf` |
| `oncoco_behavioral_signatures.py`, `oncoco_behavioral_evolution.py`, `oncoco_hllm_evolution.py`, `oncoco_signature_shares.py` | Section 5.2, Figure S8 | `oncoco_signature_shares.csv`, `oncoco_hllm_*.csv`, `behavioral_signatures_evolution.pdf` |
| `oncoco_confidence_sidecar.py`, `oncoco_confidence_diagnostics.py` | Section 5.5, Table S10, Figure S10 | `oncoco_confidence_*.csv`, `oncoco_rq1_highconf_sensitivity.{csv,tex}`, `confidence_ecdf.pdf` |
| `oncoco_category_sensitivity.py` | Section S4, Table S11, Figure S9 | `oncoco_category_sensitivity.{csv,tex}`, `oncoco_leave_one_out.csv`, `leave_one_out_delta.pdf` |
| `oncoco_tsne_embeddings.py`, `oncoco_tsne_from_classifier.py` | Figures S4–S5 | `oncoco_embedding_silhouette*.csv`, `tsne_*_chat.pdf` |
| `oncoco_mechanism_analysis.py` | counselor-side moderation contrasts (Section 5.2) | `oncoco_mechanism_*.csv` |
| `oncoco_corpus_provenance.py` | Table 1 counts and exclusions | `oncoco_provenance_*.csv` |
| `technology_acceptance/tam_convergence.py`, `tam_multiplicity.py` | RQ2, Table 3, Table S8 | `oncoco_tam_*.{csv,tex}` |
| `oncoco_prompt_replay.py`, `make_prompt_replay_paper_table.py` | RQ3, Table 4, Section S6, Table S21, Figure S11 | `oncoco_prompt_replay*.{csv,tex}`, `prompt_replay_deciles*.pdf` |
| `oncoco_hmm_phase_profile.py`, `oncoco_phase_transition_test.py` | Section S6, Table S22 (HH real phase model applied to every condition, turn-level phase test) | `oncoco_hmm_phase_profile.csv`, `oncoco_phase_transition_test.{csv,tex}`, `oncoco_phase_response_profile.csv` |
| `oncoco_human_annotation_agreement.py`, `oncoco_hh_real_annotation_agreement.py`, `oncoco_human_validation_table.py`, `oncoco_tolerant_agreement.py`, `oncoco_human_gold_condition_distance.py`, `oncoco_human_gold_signature_check.py`, `make_human_validation_supplement_tables.py` | Section 4.2, Section S7, Tables S23–S25 (expert validation on both human corpora; the expert annotations are not released) | `oncoco_human_validation*.{csv,tex}` |
| `oncoco_equal_weight_persona_delta.py` | Section S8, Table S26 | `oncoco_equal_weight_persona_delta.{csv,tex}` |
| `oncoco_student_clustered_bootstrap.py`, `oncoco_absolute_turn_profile.py` | Section S9, Tables S27–S28 (student-clustered intervals, absolute turn positions) | `oncoco_student_clustered_bootstrap.{csv,tex}`, `oncoco_absolute_turn_profile.{csv,tex}` |
| `oncoco_prompt_replay_heldout.py` | Section S10, Table S29 (held-out replay on the Llama 3.3 70B plain-template conversations) | `oncoco_prompt_replay_heldout.{csv,tex}` |
| `oncoco_blind_sample_build.py`, `oncoco_blind_sample_analysis.py` | blinded expert coding of client spans from all three conditions (the coded sheet holds real counseling text and is not released) | `oncoco_blind_sample_validation_*.{csv,tex}` |
| (hand-written) | Section S11, Table S30 (pipeline and generation parameters) | `oncoco_pipeline_parameters.tex` |
| `make_summary_tables.py`, `make_chat_only_tables.py`, `make_transition_fdr_sig_tables.py`, `make_persona_table.py` | LaTeX tables in the paper and supplement | `results/tables/*.tex` |
| `pub_figures.py`, `make_*_tikz.py` | all figures | `results/figures/oncoco/**` |

Several CSVs also carry rows for e-mail counseling conditions (`*_mail`) that an earlier version of the study analyzed. The submitted paper is restricted to the three chat conditions, and the `*_chat.tex` tables are the chat-only derivations produced by `make_chat_only_tables.py`.

## Prompt-replay experiment (RQ3)

`analysis/prompt_replay/` holds the request builder, the generation client, the phase decoder, and the SLURM job scripts of the replay experiment, and `prompts/replay/` holds the instruction blocks of every arm (`analysis/prompt_replay/README.md` documents the arms and the pipeline). The experiment regenerates the client turns of the GPT-OSS-120B conversations with the original history held fixed. The regenerated client turns themselves are not part of this release, because they were produced against unmasked counselor turns and have not passed the masking step described above. The aggregated results (`results/tables/oncoco_prompt_replay*.csv`, `oncoco_phase_*.csv`) are included.

## Data availability

The two roleplay corpora are released here in masked form (see above). The third condition, HH real, consists of archived conversations of a real online-counseling service with real help-seekers and cannot be released under the consent obtained, even in masked form. The acceptance surveys are anonymous course-level aggregates and are released only as the per-course numbers in `results/tables/oncoco_tam_*.csv`.

Analyses that use HH real as reference (every between-condition distance, the noise band, the HMM for HH real) therefore cannot be re-run from this repository alone; the aggregated CSVs carry their outputs. Analyses within or between the two roleplay conditions can.

## Environment

Python 3.11. The pinned versions used for the reported numbers are in `requirements.txt`. Scripts locate the project root relative to their own path, so run them from the repository root, for example:

```bash
python analysis/semantic/oncoco/oncoco_noise_band_analysis.py
```

Segmentation and classification (`classification_all.py`) additionally need `wtpsplit`, `adapters` (1.2.0), the OnCoCo classifier weights and a GPU or Apple MPS device. The PII masking script needs the `opf` package from https://github.com/openai/privacy-filter and downloads the checkpoint on first use.

## Building the supplement

The supplement compiles with pdflatex from the repository root (no bibliography):

```bash
pdflatex -interaction=nonstopmode oncoco_supplementary.tex
pdflatex -interaction=nonstopmode oncoco_supplementary.tex
```

The included `supplementary_material.pdf` is the output of exactly that command on the committed sources.

## Notes for reviewers

* The S-numbers cited in the main text (Table S1, Figure S4, Section S5, ...) refer to the numbering inside `supplementary_material.pdf`.
* Random seeds are fixed inside the scripts. Bootstrap and permutation counts are those stated in the paper (20,000 conversation-level resamples for the proximity gap, 1,000 split-half draws for the noise band, 2,000 for transition FDR).
* This repository is anonymized for double-blind review. Course, platform, and institution names in file names, scripts, and CSV cells have been replaced by codes where they could identify the authors. Paths and scheduler settings of the GPU cluster are placeholders (`<scratch>`, `<partition>`, `<qos>`).
