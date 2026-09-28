#!/usr/bin/env python3
"""Build the request file for the prompt-replay experiment (teacher-forced regeneration).

Purpose: test whether a client prompt written against the paper's measured signatures
(early disclosure, solution-space drift, missing reciprocal questions, missing disagreement,
verbosity) moves the OnCoCo label distribution of GPT-OSS-120B toward real clients, with the
conversation context held fixed.

The platform logged, for every LLM turn, the complete rendered prompt including the
conversation history up to that turn (``additions.prompt``). Each turn is therefore an
independent request whose context is the original history. Only the instruction blocks
vary between arms; persona block and history block are byte-identical across arms.

Arms
  A  replay:   the logged prompt, unchanged (harness / sampling baseline)
  B  generic:  instruction blocks replaced by a generic "write like a real client" cue
  C  adapted:  instruction blocks replaced by the findings-based rules in prompts/replay/
  D  adapted, dosed: as C, but rules 3 (questions) and 4 (assent) softened to target rates,
               because C overshot both (adapted_v2_later_tail.md); first message as in C
  E  as D, only rule 4 changed again: positive feedback allowed at about the rate of
               disagreement (adapted_v3_later_tail.md), because D still suppressed it
  F  phase-aware: the history is decoded with the HH-real HMM (phase_decoder.py) and the rule
               block depends on the phase: clarification (phase_klaerung_tail.md, entry into the
               help phase rate-limited) or help/closing (phase_hilfe_tail.md, feedback allowed,
               no re-opening, no planning); first message as in C
  G  F2: as F, but no examples in the help block, feedback without implementation commitments,
               entry into the help phase phrased conditionally instead of "once" (phase_*_v2_tail.md)

Input : data/raw/human_llm/chats/GPT_OSS_120B/*.json  (quality DROPS applied)
        prompts/replay/*.md
Output: data/processed/prompt_replay/replay_requests.jsonl
        data/processed/prompt_replay/replay_requests_skipped.csv
        data/processed/prompt_replay/example_prompts_<arm>.md  (one first + one later turn)
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
sys.path.insert(0, str(ROOT / "analysis/semantic/oncoco"))
sys.path.insert(0, str(ROOT / "analysis/prompt_replay"))
from apply_quality_filter import DROPS  # noqa: E402
from oncoco_prompt_meta import prompt_variant  # noqa: E402

RAW_DIR = ROOT / "data/raw/human_llm/chats/GPT_OSS_120B"
PROMPT_DIR = ROOT / "prompts/replay"
OUT_DIR = ROOT / "data/processed/prompt_replay"
MODEL = "GPT_OSS_120B"

# Anchors of the two plain-text template families used by the GPT-OSS deployment.
FIRST_PERSONA = "# Persona-Profil"
FIRST_TASK = "# Aufgabe"
FIRST_FORMAT_START = "2. **Antwortformatierung**"
FIRST_FORMAT_END = "3. **Inhaltliche Beschränkungen**"
LATER_PERSONA = "Hintergrund deiner Rolle:"
LATER_HISTORY = "Kontext der Beratungssitzung:"
LATER_TASK = "Deine Aufgabe:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default="A,B,C,D,E,F,G")
    p.add_argument("--out", default=str(OUT_DIR / "replay_requests.jsonl"))
    # held-out replay: other raw exports of the same plain template families (default: GPT-OSS)
    p.add_argument("--raw-files", nargs="+", default=None, help="raw export JSONs; default all of RAW_DIR")
    p.add_argument("--model", default=MODEL, help="deployment model key (quality DROPS, phase decoder)")
    p.add_argument("--orig-out", default=None,
                   help="also write arm O (original client text as output records) built from the arm-A requests")
    return p.parse_args()


def read(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def persona_name(prompt: str, variant: str) -> str:
    if variant == "first_message":
        m = re.match(r"# Rollenanweisung für erste Nachricht - (.+)", prompt)
    else:
        m = re.match(r"Du bist ([^.]+)\.", prompt)
    if not m:
        raise ValueError(f"no persona name in prompt starting {prompt[:60]!r}")
    return m.group(1).strip()


def _once(prompt: str, anchor: str, cid: str, num: int) -> int:
    n = prompt.count(anchor)
    if n != 1:
        raise ValueError(f"conv {cid} msg {num}: anchor {anchor!r} found {n} times, expected 1")
    return prompt.index(anchor)


def adapt_later(prompt: str, name: str, cid: str, num: int, tail_file: str, opening_file: str | None) -> str:
    i_persona = _once(prompt, LATER_PERSONA, cid, num)
    _once(prompt, LATER_HISTORY, cid, num)
    i_task = _once(prompt, LATER_TASK, cid, num)
    opening = read(opening_file).format(name=name) if opening_file else prompt[:i_persona]
    middle = prompt[i_persona:i_task]  # persona + history, verbatim
    tail = read(tail_file).format(name=name)
    return opening + middle + tail


def adapt_first(prompt: str, name: str, cid: str, num: int, task_file: str, format_file: str | None) -> str:
    i_persona = _once(prompt, FIRST_PERSONA, cid, num)
    i_task = _once(prompt, FIRST_TASK, cid, num)
    head = prompt[:i_persona]
    if format_file:
        i_f0 = _once(head, FIRST_FORMAT_START, cid, num)
        i_f1 = _once(head, FIRST_FORMAT_END, cid, num)
        head = head[:i_f0] + read(format_file) + head[i_f1:]
    persona = prompt[i_persona:i_task]  # verbatim
    task = read(task_file).format(name=name)
    return head + persona + task


PHASE_TAIL = {0: "phase_klaerung_tail.md", 1: "phase_hilfe_tail.md", 2: "phase_klaerung_tail.md"}
PHASE_TAIL_V2 = {0: "phase_klaerung_v2_tail.md", 1: "phase_hilfe_v2_tail.md", 2: "phase_klaerung_v2_tail.md"}


def build(prompt: str, arm: str, variant: str, name: str, cid: str, num: int, phase: int | None = None) -> str:
    if arm == "A":
        return prompt
    if variant == "later_turn":
        if arm == "F":
            if phase is None:
                raise ValueError(f"conv {cid} msg {num}: arm F needs a decoded phase")
            return adapt_later(prompt, name, cid, num, PHASE_TAIL[phase], "adapted_later_opening.md")
        if arm == "G":
            if phase is None:
                raise ValueError(f"conv {cid} msg {num}: arm G needs a decoded phase")
            return adapt_later(prompt, name, cid, num, PHASE_TAIL_V2[phase], "adapted_later_opening.md")
        if arm == "B":
            return adapt_later(prompt, name, cid, num, "generic_later_tail.md", None)
        if arm == "C":
            return adapt_later(prompt, name, cid, num, "adapted_later_tail.md", "adapted_later_opening.md")
        if arm == "D":
            return adapt_later(prompt, name, cid, num, "adapted_v2_later_tail.md", "adapted_later_opening.md")
        if arm == "E":
            return adapt_later(prompt, name, cid, num, "adapted_v3_later_tail.md", "adapted_later_opening.md")
    if variant == "first_message":
        if arm == "B":
            return adapt_first(prompt, name, cid, num, "generic_first_task.md", None)
        if arm in ("C", "D", "E", "F", "G"):
            return adapt_first(prompt, name, cid, num, "adapted_first_task.md", "adapted_first_format.md")
    raise ValueError(f"unsupported arm/variant {arm}/{variant} (conv {cid} msg {num})")


def check_invariants(orig: str, new: str, arm: str, variant: str, cid: str, num: int) -> None:
    """Persona and history must survive the surgery byte-identically."""
    if arm == "A":
        assert new == orig
        return
    if variant == "later_turn":
        seg = orig[orig.index(LATER_PERSONA):orig.index(LATER_TASK)]
    else:
        seg = orig[orig.index(FIRST_PERSONA):orig.index(FIRST_TASK)]
    if seg not in new:
        raise AssertionError(f"conv {cid} msg {num} arm {arm}: persona/history block altered")
    if new.count(LATER_TASK if variant == "later_turn" else FIRST_TASK) != 1:
        raise AssertionError(f"conv {cid} msg {num} arm {arm}: task anchor count != 1")


def main() -> None:
    args = parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from phase_decoder import PhaseDecoder
    dec = PhaseDecoder()
    dec.load_condition("H_LLM_roleplay_chat", args.model)
    print(f"HH-real HMM refit: loglik {dec.loglik:.1f}, occupancy {dec.occupancy.round(3)}")

    requests, skipped = [], []
    examples: dict[tuple[str, str], str] = {}
    n_conv = 0
    phase_count: dict[int, int] = {}
    files = [ROOT / f for f in args.raw_files] if args.raw_files else sorted(RAW_DIR.glob("*.json"))
    for fp in files:
        for conv in json.loads(fp.read_text(encoding="utf-8")):
            cid = str(conv["id"])
            if (args.model, cid) in DROPS:
                continue
            n_conv += 1
            persona = (conv.get("persona") or {}).get("name", "")
            msgs = sorted(conv["learn_counselling_messages"], key=lambda m: int(m["message_number"]))
            for m in msgs:
                if m.get("author") != "virtual_client":
                    continue
                num = int(m["message_number"])
                prompt = (m.get("additions") or {}).get("prompt")
                if not prompt:
                    skipped.append(dict(conv_id=cid, message_number=num, reason="no logged prompt"))
                    continue
                variant = prompt_variant(prompt)
                if variant not in ("first_message", "later_turn"):
                    skipped.append(dict(conv_id=cid, message_number=num, reason=f"variant {variant}"))
                    continue
                name = persona_name(prompt, variant)
                phase = dec.phase_before(f"{fp.relative_to(ROOT)}::{cid}", num)
                phase_count[phase] = phase_count.get(phase, 0) + 1
                for arm in arms:
                    new = build(prompt, arm, variant, name, cid, num, phase)
                    check_invariants(prompt, new, arm, variant, cid, num)
                    requests.append(dict(
                        request_id=f"{arm}:{cid}:{num}", arm=arm, conv_id=cid, message_number=num,
                        source_file=str(fp.relative_to(ROOT)), variant=variant, persona=persona or name,
                        persona_name=name, prompt=new, original_content=m.get("content") or "",
                        phase_before=phase,
                    ))
                    examples.setdefault((arm, variant if arm not in ("F", "G") else f"{variant}_phase{phase}"), new)

    with out_path.open("w", encoding="utf-8") as fh:
        for r in requests:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.orig_out:
        if "A" not in arms:
            raise SystemExit("--orig-out needs arm A in --arms")
        with Path(args.orig_out).open("w", encoding="utf-8") as fh:
            for r in requests:
                if r["arm"] == "A":
                    o = {k: v for k, v in r.items() if k != "prompt"}
                    o.update(request_id="O" + r["request_id"][1:], arm="O", content=r["original_content"],
                             finish_reason="original")
                    fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    with (out_path.parent / "replay_requests_skipped.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["conv_id", "message_number", "reason"])
        w.writeheader()
        w.writerows(skipped)
    for arm in arms:
        parts = []
        for variant in ("first_message", "later_turn", "later_turn_phase0", "later_turn_phase1", "later_turn_phase2"):
            if (arm, variant) in examples:
                parts.append(f"## {variant}\n\n```\n{examples[(arm, variant)]}\n```\n")
        (out_path.parent / f"example_prompts_{arm}.md").write_text("\n".join(parts), encoding="utf-8")

    per_arm = {a: sum(r["arm"] == a for r in requests) for a in arms}
    n_first = sum(r["variant"] == "first_message" for r in requests if r["arm"] == arms[0])
    print(f"phase before turn (0 clarification, 1 help, 2 opening): {dict(sorted(phase_count.items()))}")
    print(f"conversations: {n_conv}; turns per arm: {per_arm}; first-message turns: {n_first}; "
          f"skipped: {len(skipped)} -> {out_path}")


if __name__ == "__main__":
    main()
