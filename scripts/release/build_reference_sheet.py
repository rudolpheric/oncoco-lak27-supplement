#!/usr/bin/env python3
"""Build the HH real reference sheet: the label statistics of the real-counseling corpus, without its text.

The real-counseling conversations (HH real) are access-controlled and are not released. Every
distance in the paper is measured against them, so without a reference a third party cannot place
its own simulated client on the same scale. This script writes the minimal sufficient statistic
instead of the conversations: per conversation, the ordered sequence of (speaker, OnCoCo label)
pairs. No message text, no identifiers, no timestamps, no dates.

That is enough to reproduce, for any other corpus classified with the same scheme:
  - the HH real label distributions per speaker, and the JSD against them,
  - the split-half noise band and the size-matched null, so a distance gets a scale,
  - Cramer's V and conversation-level bootstrap intervals,
  - the role-aware transition matrix, the FDR cell tests, and the HMM phase structure.

It is not enough to re-segment the corpus, to apply a different coding scheme, or to read anything
a client wrote. Comparability requires running the OnCoCo classifier on the other corpus.

Run against the restricted corpus (authors only) and check it reproduces the published band:

    python scripts/release/build_reference_sheet.py --verify

Releasing the resulting file requires the data-governance sign-off for the HH real corpus. Until
that sign-off exists, write it outside the repository with --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
RESTRICTED = REPO.parent / "data/processed/combined/normalized/oncoco_classification_all.json"

SCHEMA = "oncoco-reference-sheet/1"
CONDITION = "HH_real_chat"
SPEAKERS = ["Client", "Counsellor"]


def conversation_key(rec: Dict) -> str:
    """Same key as the analyses: ids repeat across raw exports, so the source file must be part of it."""
    cid = rec.get("msg_learn_counselling_id") or rec.get("id")
    return f"{rec.get('source_file', '')}::{cid}"


def message_order(rec: Dict):
    num = rec.get("msg_message_number")
    try:
        return (int(num), rec.get("msg_created_at") or "")
    except (TypeError, ValueError):
        return (10 ** 9, rec.get("msg_created_at") or "")


def span_order(span: Dict):
    if "sentence_index" in span:
        return span["sentence_index"]
    return span.get("start_offset", 0)


def build(rows: List[Dict], condition: str = CONDITION) -> Dict:
    by_conv: Dict[str, List[Dict]] = {}
    for rec in rows:
        if rec.get("source") != condition:
            continue
        by_conv.setdefault(conversation_key(rec), []).append(rec)

    # The speaker is carried explicitly: in HH real it disagrees with the label's role prefix
    # for a small number of spans, and the analyses group by speaker_type, not by the prefix.
    conversations: List[List[List[int]]] = []
    labels: Dict[str, int] = {}
    for key in sorted(by_conv):
        spans: List[List[int]] = []
        for rec in sorted(by_conv[key], key=message_order):
            speaker = rec.get("speaker_type")
            if speaker not in SPEAKERS:
                continue
            for span in sorted(rec.get("sentence_classification") or [], key=span_order):
                label = span.get("predicted_label")
                if not label:
                    continue
                spans.append([SPEAKERS.index(speaker), labels.setdefault(label, len(labels))])
        if spans:
            conversations.append(spans)

    order = sorted(labels, key=labels.get)
    remap = {labels[lab]: i for i, lab in enumerate(sorted(order))}
    conversations = [[[s, remap[l]] for s, l in spans] for spans in conversations]

    return {
        "schema": SCHEMA,
        "condition": condition,
        "unit": "span",
        "speakers": SPEAKERS,
        "labels": sorted(order),
        "conversations": conversations,
        "provenance": {
            "n_conversations": len(conversations),
            "n_spans": sum(len(s) for s in conversations),
            "contains_text": False,
            "note": "Ordered (speaker, label) pairs per conversation. Derived from the "
                    "access-controlled HH real corpus. No text, identifiers, or timestamps.",
        },
    }


def count_matrices(sheet: Dict):
    """sheet -> {speaker: (matrix, label_index)}, the input the noise band and the JSD need."""
    import numpy as np

    labels = sheet["labels"]
    index = {lab: i for i, lab in enumerate(labels)}
    out = {}
    for si, speaker in enumerate(sheet["speakers"]):
        mat = np.zeros((len(sheet["conversations"]), len(labels)), dtype=float)
        for row, spans in enumerate(sheet["conversations"]):
            for s, l in spans:
                if s == si:
                    mat[row, l] += 1.0
        out[speaker] = (mat[mat.sum(axis=1) > 0], index)
    return out


def role_sequences(sheet: Dict) -> List[List[str]]:
    """Role-aware macro-category sequence per conversation, for the transition and phase analyses."""
    import sys

    sys.path.insert(0, str(REPO / "analysis/semantic/oncoco"))
    from oncoco_noise_band_analysis import role_category

    labels = sheet["labels"]
    seqs = [[role_category(labels[l]) for _, l in spans] for spans in sheet["conversations"]]
    return [s for s in seqs if len(s) >= 2]


def verify(sheet: Dict, band_csv: Path, n_splits: int = 1000, seed: int = 20260803) -> bool:
    """Recompute the split-half noise band from the sheet alone and compare with the published one."""
    import sys

    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(REPO / "analysis/semantic/oncoco"))
    from oncoco_noise_band_analysis import split_half_band

    published = pd.read_csv(band_csv)
    published = published[published["reference"] == sheet["condition"]]
    ok = True
    for speaker, (mat, _) in count_matrices(sheet).items():
        vals = split_half_band(mat, n_splits, np.random.default_rng(seed))
        got_mean, got_p95 = vals.mean(), float(np.percentile(vals, 95))
        # the published table uses the American spelling, the analyses the British one
        ref = published[published["speaker"].isin({speaker, speaker.replace("ll", "l")})]
        if ref.empty:
            print(f"  {speaker:11s} no published row to compare against")
            continue
        lo_m, hi_m = ref["band_mean"].min(), ref["band_mean"].max()
        lo_p, hi_p = ref["band_p95"].min(), ref["band_p95"].max()
        hit = lo_m - 0.002 <= got_mean <= hi_m + 0.002 and lo_p - 0.002 <= got_p95 <= hi_p + 0.002
        ok &= hit
        print(f"  {speaker:11s} mean {got_mean:.4f} (published {lo_m:.4f}-{hi_m:.4f})  "
              f"p95 {got_p95:.4f} (published {lo_p:.4f}-{hi_p:.4f})  {'ok' if hit else 'MISMATCH'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-json", default=str(RESTRICTED),
                        help="Classified spans of the restricted corpus (authors only).")
    parser.add_argument("--out", default=str(REPO / "data/hh_real_reference.json"),
                        help="Where to write the reference sheet. Keep it outside the repository "
                             "until the HH real data-governance sign-off exists.")
    parser.add_argument("--condition", default=CONDITION)
    parser.add_argument("--verify", action="store_true",
                        help="Recompute the noise band from the written sheet and compare with "
                             "results/tables/oncoco_noise_band.csv.")
    parser.add_argument("--band-csv", default=str(REPO / "results/tables/oncoco_noise_band.csv"))
    args = parser.parse_args()

    rows = json.loads(Path(args.data_json).read_text(encoding="utf-8"))
    sheet = build(rows, args.condition)

    blob = json.dumps(sheet, separators=(",", ":"), ensure_ascii=False)
    assert '"contains_text":false' in blob.replace(" ", ""), "text guard tripped"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(blob, encoding="utf-8")
    prov = sheet["provenance"]
    print(f"wrote {prov['n_conversations']} conversations, {prov['n_spans']} spans, "
          f"{len(sheet['labels'])} labels, {out.stat().st_size} bytes to {out}")

    if args.verify:
        print("verifying against the published noise band:")
        if not verify(sheet, Path(args.band_csv)):
            raise SystemExit("noise band does not reproduce from the reference sheet")
        print("reference sheet reproduces the published noise band")


if __name__ == "__main__":
    main()
