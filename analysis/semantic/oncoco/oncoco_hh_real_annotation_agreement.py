#!/usr/bin/env python3
"""Compare the expert annotation of the HH-real chat counselings with the
classifier the paper's pipeline uses.

Human reference
    annotations/manual/human_human/Codierte Segmente-final.xlsx -- a MAXQDA
    export. One row per coded segment, code as a path through the German code
    tree, `Dokumentname` naming the source document and `Anfang`/`Ende` giving
    the first and last paragraph of the segment. Paragraph numbers count the
    non-empty paragraphs of the .docx, 1-based. Segments run over several chat
    messages, so the unit is coarser than in the roleplay annotation.
    Only the chat document groups are used; mail has no counterpart .docx.

Pipeline output
    data/processed/combined/normalized/oncoco_classification_all.json --
    SaT-6l spans with English OnCoCo codes, condition HH_real_chat.

The script reconstructs each segment from the .docx, links it to the messages
of the corresponding conversation in data/processed/chat/chat_all.csv, and then
compares units, roles and labels. With --reclassify the OnCoCo classifier is run
on the human segments themselves, so only the unit boundaries differ from the
pipeline.

Outputs (results/tables/):
    oncoco_hh_real_agreement_docmap.csv
    oncoco_hh_real_agreement_segmentation.csv
    oncoco_hh_real_agreement_labels.csv
    oncoco_hh_real_agreement_per_label.csv
    oncoco_hh_real_agreement_confusions.csv
    oncoco_hh_real_agreement_units.csv
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ANNOT = ROOT / "annotations/manual/human_human/Codierte Segmente-final.xlsx"
DOCX_DIR = ROOT / "data/raw/human_human/real_counsellings"
CHAT = ROOT / "data/processed/chat/chat_all.csv"
CLS = ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"
MAP = Path(__file__).resolve().parent / "oncoco_maxqda_label_map.json"
MODEL = ROOT / "analysis/models/oncoco/xlm-roberta-large-OnCoCo-DE-EN"
OUT = ROOT / "results/tables"

HEADER = re.compile(r"^.{1,80}?\s+per\s+(Chat|Mail),\s*\d{1,2}[:.]\d{2}\s*$")
DROP_ROOTS = {"Suizidalität", "ROT"}   # parallel scheme, no OnCoCo counterpart
# The name in the annotation file is truncated for one document.
NAME_FIX = {"OÖ_C16_": "OÖ_C16_KörpGes_w"}


def nfc(s) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()


def canon(s) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).replace("\xa0", " ")).strip()


def shingles(s, k=5) -> set:
    w = re.findall(r"\w+", str(s).lower())
    return {tuple(w[i:i + k]) for i in range(max(len(w) - k + 1, 1))}


def load_docs() -> dict[str, list[str]]:
    """Non-empty paragraphs per document, index 0 == MAXQDA paragraph 1."""
    import docx
    out = {}
    for p in sorted(DOCX_DIR.glob("*.docx")):
        out[nfc(p.stem)] = [nfc(q.text) for q in docx.Document(str(p)).paragraphs if nfc(q.text)]
    return out


def load_annotation() -> pd.DataFrame:
    df = pd.read_excel(ANNOT)
    df = df[df["Dokumentgruppe"].astype(str).str.startswith("Chat")].copy()
    df["doc"] = df["Dokumentname"].map(nfc).replace(NAME_FIX)
    parts = df["Code"].astype(str).str.split(" > ")
    df["root"] = parts.str[0].str.strip()
    df["leaf"] = parts.str[-1].str.strip()
    df = df[~df["root"].isin(DROP_ROOTS)].copy()
    df["key"] = df["root"] + "|" + df["leaf"]
    m = json.loads(MAP.read_text())
    df["gold"] = df["key"].map({k: v for k, v in m.items() if not k.startswith("_")})
    df["role"] = np.where(df["root"] == "Berater*in", "B", "K")
    return df


def map_documents(docs: dict[str, list[str]], chat: pd.DataFrame) -> pd.DataFrame:
    """Each .docx to its conversation in chat_all, by 5-gram overlap."""
    conv = {c: shingles(" ".join(map(str, g.content))) for c, g in chat.groupby("conversation_id")}
    rows = []
    for name, paras in docs.items():
        a = shingles(" ".join(paras))
        scored = sorted(((len(a & b) / max(len(a | b), 1), c) for c, b in conv.items()), reverse=True)
        rows.append(dict(doc=name, conversation_id=scored[0][1], jaccard=scored[0][0],
                         runner_up=scored[1][1], runner_up_jaccard=scored[1][0]))
    return pd.DataFrame(rows).sort_values("jaccard")


def align_paragraphs(paras: list[str], msgs: pd.DataFrame) -> dict[int, int]:
    """MAXQDA paragraph number -> message_id.

    The .docx keeps every sent line as its own paragraph and repeats the speaker
    header before each; the corpus export merges consecutive lines of one message.
    Both are therefore the same sequence of content lines, which difflib aligns.
    """
    doc_lines = [(i, canon(t)) for i, t in enumerate(paras, start=1) if not HEADER.match(t)]
    msg_lines = []
    for r in msgs.itertuples():
        for line in str(r.content).split("\n"):
            if canon(line):
                msg_lines.append((int(r.message_id), canon(line)))
    a = [t for _, t in doc_lines]
    b = [t for _, t in msg_lines]
    out = {}
    for i, j, n in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(n):
            out[doc_lines[i + k][0]] = msg_lines[j + k][0]
    return out


def build_units(annot: pd.DataFrame, docs: dict, docmap: pd.DataFrame,
                chat: pd.DataFrame, pipe: dict) -> pd.DataFrame:
    cid_of = dict(zip(docmap.doc, docmap.conversation_id))
    recs = []
    for doc, g in annot.groupby("doc"):
        paras = docs.get(doc)
        if paras is None:                      # 6 documents have no .docx export
            for r in g.itertuples():
                recs.append(dict(doc=doc, conversation_id=None, role=r.role, key=r.key,
                                 gold=r.gold, text=strip_headers(r.Segment),
                                 first_para=r.Anfang, last_para=r.Ende,
                                 n_messages=np.nan, n_sat_spans=np.nan,
                                 corpus_role=None, in_corpus=False))
            continue
        cid = int(cid_of[doc])
        msgs = chat[chat.conversation_id == cid].sort_values("message_id")
        p2m = align_paragraphs(paras, msgs)
        # roles come from the pipeline output, which carries the repaired HH-real
        # roles (repair_hh_real_roles.py); chat_all.csv still has the raw platform
        # attribution, where almost every counselor turn is filed under the client
        role_of = {m: {"Counsellor": "B", "Client": "K"}.get(
                       pipe[(cid, int(m))]["speaker_type"]) if (cid, int(m)) in pipe else None
                   for m in msgs.message_id}
        raw_role_of = dict(zip(msgs.message_id, msgs.role))
        for r in g.itertuples():
            mids = sorted({p2m[p] for p in range(int(r.Anfang), int(r.Ende) + 1) if p in p2m})
            roles = {role_of.get(m) for m in mids}
            raw_roles = {raw_role_of.get(m) for m in mids}
            n_sat = sum(len(pipe[(cid, m)]["sentence_classification"])
                        for m in mids if (cid, m) in pipe)
            recs.append(dict(
                doc=doc, conversation_id=cid, role=r.role, key=r.key, gold=r.gold,
                text=strip_headers(r.Segment), first_para=r.Anfang, last_para=r.Ende,
                n_messages=len(mids), n_sat_spans=n_sat,
                corpus_role=(next(iter(roles)) if len(roles) == 1 else
                             "mixed" if len(roles) > 1 else None),
                corpus_role_raw=("B" if raw_roles == {"Beratende"} else
                                 "K" if raw_roles == {"Ratsuchende"} else
                                 "mixed" if len(raw_roles) > 1 else None),
                in_corpus=True, message_ids=",".join(map(str, mids)),
            ))
    out = pd.DataFrame(recs)
    # (doc, first_para, last_para, code) is not unique: one paragraph can hold two
    # segments carrying the same code
    out.insert(0, "unit_id", np.arange(len(out)))
    return out


def strip_headers(segment: str) -> str:
    lines = [nfc(l) for l in str(segment).split("\n")]
    keep = [l for l in lines if l and not HEADER.match(l)]
    return canon(" ".join(keep))


def load_pipeline() -> dict:
    rows = json.loads(CLS.read_text(encoding="utf-8"))
    return {(int(r["msg_learn_counselling_id"]), int(r["msg_message_number"])): r
            for r in rows if r.get("source") == "HH_real_chat"}


def reclassify(units: pd.DataFrame, batch_size: int) -> list[str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    id2label = model.config.id2label
    prefix = {"B": "Counselor: ", "K": "Client: "}
    texts = [prefix.get(r.role, "") + r.text for r in units.itertuples()]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            pred = model(**enc).logits.argmax(-1).cpu().tolist()
        out += [id2label[str(p)] if isinstance(id2label, dict) and str(p) in id2label
                else id2label[p] for p in pred]
    return out


def kappa(a, b) -> float:
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = len(a)
    po = np.trace(m) / n
    pe = float((m.sum(0) / n) @ (m.sum(1) / n))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def tier(code, depth) -> str:
    return "-".join(str(code).split("-")[:depth])


def score_block(name: str, gold: pd.Series, pred: pd.Series, accept: dict) -> dict:
    hit = pd.Series([g == p or p in accept.get(g, ()) for g, p in zip(gold, pred)], index=gold.index)
    return dict(
        comparison=name, n=len(gold), accuracy=float(hit.mean()), kappa=kappa(gold, pred),
        majority_baseline=float(gold.value_counts(normalize=True).max()) if len(gold) else np.nan,
        acc_role=float((gold.map(lambda c: tier(c, 1)) == pred.map(lambda c: tier(c, 1))).mean()),
        acc_pillar=float((gold.map(lambda c: tier(c, 2)) == pred.map(lambda c: tier(c, 2))).mean()),
        acc_group=float((gold.map(lambda c: tier(c, 3)) == pred.map(lambda c: tier(c, 3))).mean()),
        tvd_distribution=float(0.5 * (gold.value_counts(normalize=True)
                                      .subtract(pred.value_counts(normalize=True), fill_value=0)
                                      .abs().sum())),
    )


def per_label_scores(gold: pd.Series, pred: pd.Series) -> pd.DataFrame:
    rows = []
    for lab in sorted(set(gold) | set(pred)):
        tp = int(((gold == lab) & (pred == lab)).sum())
        fp = int(((gold != lab) & (pred == lab)).sum())
        fn = int(((gold == lab) & (pred != lab)).sum())
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        rows.append(dict(label=lab, human_n=tp + fn, model_n=tp + fp,
                         human_share=(tp + fn) / len(gold), model_share=(tp + fp) / len(gold),
                         precision=p, recall=r, f1=2 * p * r / (p + r) if tp and p + r else 0.0))
    return pd.DataFrame(rows).sort_values("human_n", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reclassify", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    accept = json.loads(MAP.read_text()).get("_accept_also", {})

    annot = load_annotation()
    docs = load_docs()
    chat = pd.read_csv(CHAT)
    chat = chat[chat.source_file.astype(str).str.contains("RealCounsellings")]
    pipe = load_pipeline()

    docmap = map_documents(docs, chat)
    docmap.to_csv(OUT / "oncoco_hh_real_agreement_docmap.csv", index=False)
    print(f"documents mapped: {len(docmap)}  weakest jaccard {docmap.jaccard.min():.3f}  "
          f"strongest runner-up {docmap.runner_up_jaccard.max():.3f}  "
          f"duplicate targets {int(docmap.conversation_id.duplicated().sum())}")

    units = build_units(annot, docs, docmap, chat, pipe)
    print(f"coded segments: {len(units)}  with .docx: {int(units.in_corpus.sum())}  "
          f"mapped to an OnCoCo code: {int(units.gold.notna().sum())}")

    inc = units[units.in_corpus]
    seg = pd.Series(dict(
        conversations=int(inc.conversation_id.nunique()),
        human_segments=len(inc),
        corpus_messages=int(chat[chat.conversation_id.isin(inc.conversation_id)].shape[0]),
        sat_spans=int(sum(len(v["sentence_classification"]) for k, v in pipe.items()
                          if k[0] in set(inc.conversation_id))),
        messages_per_segment=float(inc.n_messages.mean()),
        sat_spans_per_segment=float(inc.n_sat_spans.mean()),
        segments_spanning_one_message=float((inc.n_messages == 1).mean()),
        segments_spanning_multiple=float((inc.n_messages > 1).mean()),
        segments_without_message=float((inc.n_messages == 0).mean()),
        chars_per_segment=float(inc.text.str.len().mean()),
        role_agreement_repaired=float((inc.role == inc.corpus_role).mean()),
        role_agreement_raw_export=float((inc.role == inc.corpus_role_raw).mean()),
        role_mixed=float((inc.corpus_role == "mixed").mean()),
        role_unknown=float(inc.corpus_role.isna().mean()),
    ))
    seg.to_csv(OUT / "oncoco_hh_real_agreement_segmentation.csv", header=False)
    print(seg.to_string())

    rows = []
    if args.reclassify:
        loc = units.dropna(subset=["gold"]).copy()
        loc = loc[loc.text.str.len() > 0]
        loc["pred"] = reclassify(loc, args.batch_size)
        units = units.merge(loc[["unit_id", "pred"]], on="unit_id", how="left")
        rows.append(score_block("human segments, reclassified", loc.gold, loc.pred, accept))
        for role, name in (("B", "counselor"), ("K", "client")):
            s = loc[loc.role == role]
            rows.append(score_block(f"human segments, reclassified, {name}", s.gold, s.pred, accept))
        one = loc[loc.n_messages == 1]
        rows.append(score_block("human segments within one message", one.gold, one.pred, accept))
        many = loc[loc.n_messages > 1]
        rows.append(score_block("human segments across messages", many.gold, many.pred, accept))
        per_label_scores(loc.gold, loc.pred).to_csv(
            OUT / "oncoco_hh_real_agreement_per_label.csv", index=False)
        conf = (loc[loc.gold != loc.pred].groupby(["gold", "pred"]).size()
                .sort_values(ascending=False).rename("n").reset_index())
        conf["share_of_errors"] = conf.n / conf.n.sum()
        conf.head(40).to_csv(OUT / "oncoco_hh_real_agreement_confusions.csv", index=False)

        lab = pd.DataFrame(rows)
        lab.to_csv(OUT / "oncoco_hh_real_agreement_labels.csv", index=False)
        print(lab.to_string(index=False))

    units.to_csv(OUT / "oncoco_hh_real_agreement_units.csv", index=False)


if __name__ == "__main__":
    main()
