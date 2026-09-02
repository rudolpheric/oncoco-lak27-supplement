# Supplementary repository: Measuring Counseling Interaction Patterns Across Human–Human and Human–LLM Dialogues

Anonymized companion repository for a LAK 2027 Research Track submission (Journal of Learning Analytics format).
It contains the Supplementary Material referenced from the main text, the analysis code that produced every table and figure in the paper, and the aggregated, non-textual result tables the code writes.

It does **not** contain conversation transcripts, persona prompts, survey responses, or model weights. See [Data availability](#data-availability).

## Contents

| Path | What it is |
|---|---|
| `supplementary_material.pdf` | Supplementary Material (Tables S1–S16, Figures S1–S10) as cited in the main text |
| `oncoco_supplementary.tex`, `JLA_article.cls`, `header.png` | LaTeX source of the supplement; compiles from the repository root (see [Building the supplement](#building-the-supplement)) |
| `results/tables/*.tex` | Every table `\input` by the main paper or the supplement |
| `results/tables/*.csv` | Aggregated numeric outputs of the analysis scripts (label distributions, distances, bootstrap intervals, HMM parameters, transition statistics, confidence diagnostics, survey aggregates) |
| `results/figures/oncoco/*.pdf` | Every figure included by the main paper or the supplement |
| `results/figures/oncoco/tikz/` | TikZ sources of the four composite figures (pipeline, main-result, HMM phase, noise band) and the generator notes |
| `analysis/` | Analysis code (Python) |
| `scripts/analysis/` | Corpus normalization and inventory |

## Pipeline overview

The main paper's Figure 2 describes the pipeline. The code follows the same steps.

1. **Corpus normalization.** `scripts/analysis/02_chat_to_common.py` reads the platform exports and writes one common chat table. It applies the manual quality filter whose decisions are documented in `analysis/quality_review/` (`apply_quality_filter.py` holds the single source of truth for the dropped conversations; the two `*drop_decisions.csv` files record the review rationale).
2. **Segmentation.** Messages are split into spans with SaT-6l (wtpsplit) and a LoRA adapter. The adapter and the segmentation run belong to a separate segmentation project and are not part of this repository. The segmented, classified corpus is the starting point of all scripts below.
3. **Classification.** `analysis/semantic/edm/classification_all.py` labels each span with an XLM-RoBERTa-large classifier fine-tuned on the OnCoCo scheme (66 fine-grained categories). The classifier weights are not included. `analysis/quality_review/repair_hh_real_roles.py` then repairs the speaker roles of the archived real-counseling exports at platform join events.
4. **Analyses.** Everything under `analysis/semantic/oncoco/` and `analysis/technology_acceptance/` reads the classified corpus (or the CSVs derived from it) and writes to `results/tables/` and `results/figures/oncoco/`.

### Script to output map

| Script | Paper / supplement element | Main outputs |
|---|---|---|
| `oncoco_sentence_analysis.py` | label distributions, temporal deciles | `oncoco_label_distribution.csv`, `oncoco_temporal_distribution.csv` |
| `oncoco_rq_stats.py` | RQ1 distances and per-model realness | `oncoco_rq1_distances.csv`, `oncoco_rq1_label_effects.csv`, `oncoco_rq1b_model_realness.csv` |
| `oncoco_noise_band_analysis.py` | Figure 3, Table S12 (split-half noise band, size-matched null, Cramér's V, FDR on transitions) | `oncoco_noise_band.csv`, `oncoco_transition_fdr.csv` |
| `oncoco_rq1b_bootstrap.py` | Table S13 (conversation-level bootstrap per model) | `oncoco_rq1b_model_realness_bootstrap.{csv,tex}` |
| `oncoco_unit_ablation.py`, `oncoco_seg_ordering_bootstrap.py` | Table 3 and Table S9 (unit of analysis: spans, sentences, messages) | `oncoco_unit_ablation.{csv,tex}`, `oncoco_unit_noise_band.tex`, `oncoco_seg_ordering_bootstrap.csv` |
| `oncoco_rq1_segmentation_ablation.py`, `oncoco_rq2_segmentation_effects.py` | misspecified-segmentation row of Table 3 | `oncoco_rq1_segmentation_ablation.csv`, `oncoco_rq2_segmentation_*.csv` |
| `oncoco_sequence_analysis.py` | transition matrices, Figures S1–S3, Table S14 | `oncoco_transition_matrix.csv`, `oncoco_transition_top.csv`, `transition_*_chat.pdf` |
| `oncoco_hmm_model_selection.py`, `oncoco_hmm_final_fit.py` | Figure 4, Tables S5–S6 (best-of-20 restarts, k sweep) | `oncoco_hmm_model_selection.csv`, `oncoco_hmm_emissions*.csv`, `oncoco_hmm_transitions*.csv` |
| `oncoco_within_condition_analysis.py` | Tables S2, S7, Figures S6–S7 | `oncoco_within_condition_jsd*.csv`, `within_jsd_*_chat.pdf` |
| `oncoco_behavioral_signatures.py`, `oncoco_behavioral_evolution.py`, `oncoco_hllm_evolution.py`, `oncoco_signature_shares.py` | Section 5.2, Figure S8 | `oncoco_signature_shares.csv`, `oncoco_hllm_*.csv`, `behavioral_signatures_evolution.pdf` |
| `oncoco_confidence_sidecar.py`, `oncoco_confidence_diagnostics.py` | Section 5.6, Table S10, Figure S10 | `oncoco_confidence_*.csv`, `oncoco_rq1_highconf_sensitivity.{csv,tex}`, `confidence_ecdf.pdf` |
| `oncoco_category_sensitivity.py` | Section S4, Table S11, Figure S9 | `oncoco_category_sensitivity.{csv,tex}`, `oncoco_leave_one_out.csv`, `leave_one_out_delta.pdf` |
| `oncoco_tsne_embeddings.py`, `oncoco_tsne_from_classifier.py` | Figures S4–S5 | `oncoco_embedding_silhouette*.csv`, `tsne_*_chat.pdf` |
| `oncoco_mechanism_analysis.py` | counselor-side moderation contrasts (Section 5.2) | `oncoco_mechanism_*.csv` |
| `oncoco_corpus_provenance.py` | Table 1 counts and exclusions | `oncoco_provenance_*.csv` |
| `technology_acceptance/tam_convergence.py`, `tam_multiplicity.py` | RQ3, Table 5, Table S8 | `oncoco_tam_*.{csv,tex}` |
| `make_summary_tables.py`, `make_chat_only_tables.py`, `make_transition_fdr_sig_tables.py`, `make_persona_table.py` | LaTeX tables in the paper and supplement | `results/tables/*.tex` |
| `pub_figures.py`, `make_*_tikz.py` | all figures | `results/figures/oncoco/**` |

Several CSVs also carry rows for e-mail counseling conditions (`*_mail`) that an earlier version of the study analyzed. The submitted paper is restricted to the three chat conditions, and the `*_chat.tex` tables are the chat-only derivations produced by `make_chat_only_tables.py`.

## Data availability

The raw material consists of counseling conversations from training courses and from an archived real online-counseling service, together with anonymous course-level acceptance surveys. The conversations contain personal narratives and cannot be released, even in anonymized form, under the consent obtained. The persona prompts are withheld for the same reason and can be provided on request after review.

What the scripts expect as input is one JSON list (`data/processed/combined/normalized/oncoco_classification_all.json`) with one record per message:

```json
{
  "id": "<conversation id>",
  "source": "HH_roleplay_chat | HH_real_chat | H_LLM_roleplay_chat",
  "modality": "chat",
  "model": "<LLM name for H-LLM, empty otherwise>",
  "speaker_type": "Client | Counsellor",
  "msg_message_number": "<position in conversation>",
  "msg_content": "<message text>",
  "sentence_classification": [
    {"sentence_index": 0, "start_offset": 0, "end_offset": 6, "text": "<span>", "predicted_label": "CL-FB-*-*-*-*"}
  ]
}
```

Everything in `results/tables/*.csv` is an aggregate over that file with the text removed. The label codes follow the OnCoCo scheme; `analysis/semantic/oncoco/label_text_map.json` maps them to readable names.

## Environment

Python 3.11. The pinned versions used for the reported numbers are in `requirements.txt`. Scripts locate the project root relative to their own path, so run them from the repository root, for example:

```bash
python analysis/semantic/oncoco/oncoco_noise_band_analysis.py
```

Classification (`classification_all.py`) and the confidence sidecar additionally need the OnCoCo classifier weights and a GPU or Apple MPS device; the sidecar reproduces the exact batch composition of the original run so that probabilities are bit-identical.

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
* This repository is anonymized for double-blind review. Course, platform, and institution names in file names and CSV cells have been masked where they could identify the authors.
