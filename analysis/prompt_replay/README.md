# Prompt-replay experiment

Regenerates the simulated-client turns of the 90 GPT-OSS-120B conversations with the
conversation history held fixed (teacher-forced replay) under different prompts, then
measures them with the unchanged OnCoCo pipeline. Answers the objection that the realism
gap is "just prompting": without the measurement one does not know what to prompt.

Arms

| arm | prompt | role |
|---|---|---|
| O | original client text, re-classified with the current pipeline | baseline without pipeline drift |
| A | logged prompt, byte-identical | harness / sampling baseline |
| B | generic "write like a real client" cue | naive-prompting control |
| C | rules against the five measured signatures (`prompts/replay/`) | findings-based prompt |
| D | as C, rules 3 (questions) and 4 (assent) dosed | second iteration with the measurement as feedback |
| E | as D, positive feedback explicitly allowed | probe: short assent is labelled Consent, not feedback |
| F | phase-aware: history decoded with the HH-real HMM, rule block per phase (`phase_decoder.py`, `prompts/replay/phase_*_tail.md`) | entry into the help phase rate-limited, feedback where real clients give it |
| G | as F, no form examples in the help block, no implementation commitments, entry phrased conditionally (`phase_*_v2_tail.md`) | the phase-dependent arm reported in the paper |

Steps (repo root, local unless stated)

```bash
# 1. requests (arms A/B/C; anchors and persona/history invariants are asserted)
~/.pyenv/versions/3.11.11/bin/python analysis/prompt_replay/build_replay_prompts.py

# 2. generation on a SLURM GPU cluster (vLLM, openai/gpt-oss-120b, A100 or H200); see jobs/
rsync -az --no-perms analysis/prompt_replay/generate_replay.py analysis/prompt_replay/jobs \
    data/processed/prompt_replay/replay_requests.jsonl <user>@<cluster>:<scratch>/oncoco_replay/
ssh <user>@<cluster> 'cd <scratch>/oncoco_replay && sbatch --export=ALL,ARMS=A+B+C jobs/vllm_gptoss_replay.sh'
rsync -az <user>@<cluster>:<scratch>/oncoco_replay/outputs/replay_outputs.jsonl data/processed/prompt_replay/

# 3. chat CSV per arm and OnCoCo classification (SaT-6l + LoRA, pyenv base interpreter)
~/.pyenv/versions/3.11.11/bin/python analysis/prompt_replay/replay_to_chat_csv.py \
    --outputs data/processed/prompt_replay/replay_outputs.jsonl --out data/processed/prompt_replay/replay_chat.csv
~/.pyenv/versions/3.11.11/bin/python analysis/semantic/classification/classification_all.py \
    --chat_csv data/processed/prompt_replay/replay_chat.csv --no-include_mail --segmentation sat6l \
    --use_role_prefix --output data/processed/prompt_replay/oncoco_classification_replay.json

# 4. tables and figure
.venv/bin/python analysis/semantic/oncoco/oncoco_prompt_replay.py \
    --replay data/processed/prompt_replay/oncoco_classification_O.json data/processed/prompt_replay/oncoco_classification_replay.json \
    --turn-status data/processed/prompt_replay/replay_chat_O_turn_status.csv data/processed/prompt_replay/replay_chat_turn_status.csv
```

Phase diagnostics: `analysis/semantic/oncoco/oncoco_hmm_phase_profile.py` (client profile per
HH-real HMM phase, who leads phase changes) and `oncoco_phase_transition_test.py` (turn-level:
does the client's turn move the phase, response by phase x counselor's last act). Both decode every
condition with the one HH-real model, because states of separately fitted HMMs are not comparable.

Arm O is built from the requests file (`orig_outputs.jsonl`, content = original text) and
classified the same way. `verify_segmentation.py` compares the current segmentation with the
master corpus; it reproduces ~90 % of messages exactly (differences cluster on emoji and
special characters), which is why arm O and not the master row is the comparison base.

Cluster notes: the shared file systems lack the quota headroom, so the weights go to
the node-local tmpfs `/tmp` (counts against `--mem`) and are downloaded per job (~25 min).
Do not load the CUDA module in the job: it sets `CUDA_HOME` to a path the compute nodes lack and
FlashInfer then fails to JIT; the FlashInfer sampler is disabled instead.

Held-out replay (paper Section 5.7, Supplementary Section S10)

Prompts D and G were dosed on the 90 GPT-OSS conversations. The held-out run applies them unchanged
to the 117 Llama 3.3 70B conversations of the same plain template family, with GPT-OSS-120B as the
generator in every arm, so arm A (logged Llama prompt) is the baseline:

```bash
python analysis/prompt_replay/build_replay_prompts.py --model LLama3_3_70B --arms A,D,G \
    --raw-files data/raw/human_llm/chats/LLama3_3_70B/E04.json \
                data/raw/human_llm/chats/LLama3_3_70B/E09.json \
    --out data/processed/prompt_replay_heldout/replay_requests_heldout.jsonl \
    --orig-out data/processed/prompt_replay_heldout/orig_outputs_heldout.jsonl
# generation as in step 2, with REQUESTS=... OUTPUT=... ARMS=A+D+G passed to sbatch --export
python analysis/prompt_replay/replay_to_chat_csv.py --model LLama3_3_70B --group-suffix _llama \
    --raw-files <the two files above> --outputs <outputs> --arms A --out data/processed/prompt_replay_heldout/replay_chat_A.csv
# the same for O (from orig_outputs_heldout.jsonl), D and G, then classification_all.py per arm, then
python analysis/semantic/oncoco/oncoco_prompt_replay_heldout.py
```
