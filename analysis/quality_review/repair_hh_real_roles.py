#!/usr/bin/env python3
"""Repair the speaker-role assignment in the HH real counseling chat export.

WHY
---
`scripts/analysis/02_chat_to_common.py::_map_role` assigns roles from the author string
alone: for RealCounsellings.json, `user`/`vikl` -> client and `system` -> counselor. That
holds for 26 of the 54 conversations. In the other 28 the counseling platform put BOTH
participants in the `user` stream: the counselor enters the chat with a second "Beigetreten"
join event and every subsequent counselor turn is authored as `user`. Under the author rule
those counselor turns are labelled Client, which contaminates the client-side distribution of
the reference condition with counselor speech (72% of the HH real client stream sits in these
conversations). The `system` stream in those conversations carries only the platform's
automated queue notices.

WHAT THIS DOES
--------------
1. Drops platform boilerplate (queue/refusal notices and bare join markers) from every
   conversation. It is machine-written text, not a speaker turn, and it accounted for 54% of
   the reference condition's Moderation mass.
2. Reconstructs roles in the affected conversations. The `user` stream is segmented at join
   events; within a segment turn-taking alternates. The segment's STARTING role is not
   assumed -- both parities are scored against an independent, content-based role signal (the
   OnCoCo classifier run WITHOUT a role prefix still predicts a role-specific label) and the
   better-agreeing parity is taken. Structure supplies the alternation, content supplies the
   phase; the agreement rate of the winning parity is reported so the mapping is auditable.
3. Re-runs the classifier over every item whose role changed, with the corrected
   "Counselor: "/"Client: " prefix, reproducing the original inference recipe exactly.

Outputs a repaired copy of the classification JSON plus a per-conversation audit CSV.

Usage:
    python analysis/quality_review/repair_hh_real_roles.py                 # SaT corpus
    python analysis/quality_review/repair_hh_real_roles.py --data-json ..._regex.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
NORM = ROOT / "data/processed/combined/normalized"

# Platform automation. A message counts as boilerplate only if EVERY non-empty line matches,
# so a counselor turn that happens to follow a join marker in the same message is kept.
BOILERPLATE_PREFIXES = (
    "Demnächst wird Ihr Chat von einer/m Onlineberater",
    "Wir nehmen uns ausreichend Zeit für die Chats",
    "Derzeit sind leider keine Onlineberater",
    "Zur Zeit sind alle Onlineberater",
    "Beigetreten",
)
JOIN_MARKER = "Beigetreten"
CLIENT_AUTHORS = {"user", "vikl", "client", "ratsuchende"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-json", default=str(NORM / "oncoco_classification_all.json"))
    p.add_argument("--out-json", default="", help="Default: <data-json> written in place after a backup.")
    p.add_argument("--backup-suffix", default=".bak_pre_rolefix")
    p.add_argument("--audit-csv", default=str(ROOT / "results/tables/oncoco_hh_real_role_repair.csv"))
    p.add_argument("--model-path", default=str(ROOT / "analysis/models/oncoco/xlm-roberta-large-OnCoCo-DE-EN"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="Report the reconstruction without re-classifying.")
    return p.parse_args()


def is_boilerplate(text: str) -> bool:
    lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
    return bool(lines) and all(any(l.startswith(b) for b in BOILERPLATE_PREFIXES) for l in lines)


def has_join(text: str) -> bool:
    return any(l.strip().startswith(JOIN_MARKER) for l in (text or "").split("\n") if l.strip())


def apply_role_prefix(text: str, speaker_type: str) -> str:
    """Mirrors analysis/semantic/edm/classification_all.py::apply_role_prefix."""
    st = (speaker_type or "").strip().lower()
    if st.startswith("couns"):
        return f"Counselor: {text}"
    if st.startswith("client"):
        return f"Client: {text}"
    return text


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    id2label = model.config.id2label
    label_of = (lambda i: id2label[str(i)]) if (isinstance(id2label, dict) and "0" in id2label) \
        else (lambda i: id2label[i])
    return tokenizer, model, device, label_of


def classify(texts, tokenizer, model, device, label_of, batch_size):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        out.extend(label_of(int(j)) for j in logits.argmax(dim=-1).cpu().numpy())
    return out


def role_of_label(label: str) -> str:
    """Role implied by an OnCoCo label id (CL-* = client, CO-* = counselor)."""
    return "Client" if str(label).startswith("CL-") else "Counsellor"


def main():
    args = parse_args()
    data_json = Path(args.data_json)
    rows = json.loads(data_json.read_text(encoding="utf-8"))
    print(f"{len(rows)} records from {data_json.name}")

    real_idx = [i for i, r in enumerate(rows) if r.get("source") == "HH_real_chat"]
    by_conv = defaultdict(list)
    for i in real_idx:
        by_conv[rows[i]["msg_learn_counselling_id"]].append(i)
    for idxs in by_conv.values():
        idxs.sort(key=lambda i: int(rows[i]["msg_message_number"]))
    print(f"HH real: {len(real_idx)} messages across {len(by_conv)} conversations")

    # ---- 1. boilerplate ------------------------------------------------------------
    drop = {i for i in real_idx if is_boilerplate(rows[i].get("msg_content"))}
    print(f"platform boilerplate: {len(drop)} messages dropped")

    # ---- 2. which conversations have the counselor inside the client stream? -------
    broken = []
    for conv, idxs in by_conv.items():
        joins = [i for i in idxs
                 if str(rows[i].get("msg_author", "")).lower() in CLIENT_AUTHORS
                 and has_join(rows[i].get("msg_content"))]
        if len(joins) >= 2:
            broken.append(conv)
    print(f"role-broken conversations (counselor authored as a client id): {len(broken)} of {len(by_conv)}")

    # ---- 3. independent, content-based role signal ---------------------------------
    # Classify each surviving message of the broken conversations WITHOUT a role prefix. The
    # predicted label still carries a role, giving a signal independent of turn structure.
    cand = [i for conv in broken for i in by_conv[conv] if i not in drop
            and str(rows[i].get("msg_author", "")).lower() in CLIENT_AUTHORS]
    signal = {}
    if not args.dry_run and cand:
        tokenizer, model, device, label_of = load_model(args.model_path)
        print(f"model on {device}; scoring {len(cand)} unprefixed messages for the role signal")
        preds = classify([rows[i].get("msg_content") or "" for i in cand],
                         tokenizer, model, device, label_of, args.batch_size)
        signal = {i: role_of_label(p) for i, p in zip(cand, preds)}

    # ---- 4. reconstruct ------------------------------------------------------------
    audit = []
    changed = []
    for conv in broken:
        # Segment on the ORIGINAL stream, before boilerplate is removed: the counselor's entry
        # is often a bare "Beigetreten" marker, which is itself boilerplate. Dropping it first
        # would destroy the very boundary that marks where the second speaker appears.
        stream = [i for i in by_conv[conv]
                  if str(rows[i].get("msg_author", "")).lower() in CLIENT_AUTHORS]
        segments, cur = [], []
        for i in stream:
            if has_join(rows[i].get("msg_content")) and cur:
                segments.append(cur)
                cur = [i]
            else:
                cur.append(i)
        if cur:
            segments.append(cur)

        for seg_no, seg in enumerate(segments):
            if seg_no == 0:
                # before the counselor joined: the client is alone in the stream
                for i in seg:
                    rows[i]["_new_role"] = "Client"
                agree = np.nan
            else:
                # alternation within the segment; parity chosen by agreement with the signal
                best, best_agree = None, -1.0
                for start in ("Counsellor", "Client"):
                    roles = [start if k % 2 == 0 else ("Client" if start == "Counsellor" else "Counsellor")
                             for k in range(len(seg))]
                    scored = [(r, signal.get(i)) for i, r in zip(seg, roles) if signal.get(i)]
                    a = float(np.mean([r == s for r, s in scored])) if scored else 0.5
                    if a > best_agree:
                        best, best_agree = roles, a
                for i, r in zip(seg, best):
                    rows[i]["_new_role"] = r
                agree = best_agree
            audit.append(dict(conversation_id=conv, segment=seg_no, n_messages=len(seg),
                              start_role=rows[seg[0]]["_new_role"], signal_agreement=agree))

    for i in real_idx:
        new = rows[i].pop("_new_role", None)
        if new and new != rows[i].get("speaker_type"):
            rows[i]["speaker_type"] = new
            changed.append(i)
    print(f"messages whose role changed: {len(changed)}")

    audit_df = pd.DataFrame(audit)
    if len(audit_df):
        seg = audit_df[audit_df["segment"] > 0]
        print(f"parity agreement with the independent role signal: "
              f"median {seg['signal_agreement'].median():.3f}, "
              f"min {seg['signal_agreement'].min():.3f} over {len(seg)} segments")
        Path(args.audit_csv).parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(args.audit_csv, index=False)
        print(f"Wrote {args.audit_csv}")

    if args.dry_run:
        print("dry run: no re-classification, no output written")
        return

    # ---- 5. re-classify every item whose role changed ------------------------------
    if changed:
        if not signal:
            tokenizer, model, device, label_of = load_model(args.model_path)
        texts, targets = [], []
        for i in changed:
            st = rows[i]["speaker_type"]
            mc = rows[i].get("message_classification") or {}
            if mc.get("text") is not None:
                texts.append(apply_role_prefix(mc["text"], st))
                targets.append((i, "message", None))
            for s in rows[i].get("sentence_classification") or []:
                if s.get("text") is not None:
                    texts.append(apply_role_prefix(s["text"], st))
                    targets.append((i, "span", s["sentence_index"]))
        print(f"re-classifying {len(texts)} items with the corrected role prefix")
        preds = classify(texts, tokenizer, model, device, label_of, args.batch_size)
        for (i, unit, sidx), lab in zip(targets, preds):
            if unit == "message":
                rows[i]["message_classification"]["predicted_label"] = lab
            else:
                for s in rows[i]["sentence_classification"]:
                    if s["sentence_index"] == sidx:
                        s["predicted_label"] = lab
                        break

    kept = [r for j, r in enumerate(rows) if j not in drop]
    out_path = Path(args.out_json) if args.out_json else data_json
    if not args.out_json:
        backup = data_json.with_suffix(data_json.suffix + args.backup_suffix)
        if not backup.exists():
            backup.write_text(data_json.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"backed up to {backup.name}")
    out_path.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({len(kept)} records, {len(rows) - len(kept)} dropped)")


if __name__ == "__main__":
    main()
