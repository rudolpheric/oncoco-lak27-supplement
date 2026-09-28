#!/usr/bin/env python3
"""Check that the local SaT-6l + LoRA setup reproduces the master corpus segmentation.

Loads the splitter exactly as classification_all.py does, prints the active adapters, then
re-segments N client and counselor messages of the master corpus and counts messages whose
span offsets are identical. A correct setup gives N/N (the LoRA adapter changes boundaries;
an inactive adapter shows up as extra sentence splits).

    PYTHONPATH=<overlay> .venv/bin/python analysis/prompt_replay/verify_segmentation.py --n 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/semantic/classification"))


def patch_parallel_hook() -> None:
    """adapters 1.3.0 indexes input[1] positionally in its parallel-composition pre-hook, but
    wtpsplit 2.2.x passes attention_mask as a keyword -> IndexError. The hook is a no-op for a
    single LoRA, so guard it."""
    try:
        from adapters.models.bert import mixin_bert
    except ImportError:
        return

    def _set(self, layer):
        def hook(module, input):
            if len(input) > 1:
                mixin_bert.adjust_tensors_for_parallel_(input[0], input[1])
            return input

        layer.register_forward_pre_hook(hook)

    mixin_bert.BertModelAdaptersMixin._set_layer_hook_for_parallel = _set


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patch-hook", action="store_true")
    p.add_argument("--device", default=None, help="force the splitter onto this device (e.g. cpu)")
    p.add_argument("--master", default=str(ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"))
    args = p.parse_args()
    if args.patch_hook:
        patch_parallel_hook()

    import classification_all as ca
    splitter = ca.load_sat_with_lora("segment-any-text/sat-6l", str(ROOT / "analysis/models/segmentation/adapter"), 0.5)
    if args.device:
        splitter.model.to(args.device)
        splitter.device = next(splitter.model.parameters()).device
    print("active adapters:", getattr(splitter.model, "active_adapters", None))
    import wtpsplit, adapters, transformers, torch
    print("versions: wtpsplit", wtpsplit.__version__, "adapters", adapters.__version__,
          "transformers", transformers.__version__, "torch", torch.__version__, "device", splitter.device)

    rows = json.loads(Path(args.master).read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("model") == "GPT_OSS_120B" and (r.get("msg_content") or "").strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    same, diffs = 0, []
    for r in rows:
        spans = ca.split_sentences_with_sat(r["msg_content"], splitter, 1)
        got = [(s, e) for s, e, _ in spans]
        exp = [(x["start_offset"], x["end_offset"]) for x in r["sentence_classification"]]
        if got == exp:
            same += 1
        else:
            diffs.append((r["id"], r["msg_message_number"], r["speaker_type"], exp, got))
    print(f"identical: {same}/{len(rows)}")
    import re
    special = re.compile(r"[^\x00-\x7FäöüÄÖÜß€§°„“”‚‘–\-]")
    by_id = {(r["id"], r["msg_message_number"]): r["msg_content"] for r in rows}
    n_special_all = sum(bool(special.search(t)) for t in by_id.values())
    n_special_diff = sum(bool(special.search(by_id[(d[0], d[1])])) for d in diffs)
    print(f"messages with emoji/special chars: {n_special_all}/{len(rows)}; "
          f"among differing messages: {n_special_diff}/{len(diffs)}")
    for d in diffs[:5]:
        print("  diff", d[:3], "\n     master", d[3], "\n     now   ", d[4])


if __name__ == "__main__":
    main()
