#!/usr/bin/env python3
"""Compare the human OnCoCo annotation of the HH-roleplay chats with the
classifier the paper's pipeline uses.

Human reference
    data/processed/human_human/processed_dataset.xlsx -- one row per annotated
    segment, German B/K codes. Verlauf-ID 1-64 are conversations 1001-1064 of
    data/raw/human_human/chats/E01.json; Verlauf-ID 65-76 have no
    counterpart in the exported corpus (text only, no pipeline output).
    Provenance is recovered from the 2022 source workbook
    data/raw/human_human/chats/Gesprächsdaten.xlsx: units whose original
    category was blank or "Sonstiges" were filled in later and are reported
    separately.

Pipeline output
    data/processed/combined/normalized/oncoco_classification_all.json --
    SaT-6l spans with English OnCoCo codes.

Stage 1 answers "are the units the same?" (segmentation agreement).
Stage 2 answers "do the labels agree?", first on the pipeline's own spans and
then, with --reclassify, on the human spans: same model, same role prefix, only
the unit boundaries change.

Outputs (results/tables/):
    oncoco_human_agreement_segmentation.csv
    oncoco_human_agreement_labels.csv
    oncoco_human_agreement_per_label.csv
    oncoco_human_agreement_confusions.csv
    oncoco_human_agreement_units.csv
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
XLSX = ROOT / "data/processed/human_human/processed_dataset.xlsx"
RAW_XLSX = ROOT / "data/raw/human_human/chats/Gesprächsdaten.xlsx"
CLS = ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"
MAP = Path(__file__).resolve().parent / "oncoco_de_en_label_map.json"
MODEL = ROOT / "analysis/models/oncoco/xlm-roberta-large-OnCoCo-DE-EN"
OUT = ROOT / "results/tables"

VID_OFFSET = 1000   # Verlauf-ID v == E01 conversation 1000+v
MAX_VID = 64        # last Verlauf-ID with a counterpart in the corpus
FILLED_RAW = {"", "nan", "sonstiges"}


def canon(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).replace("\xa0", " ")


def find_span(hay: str, needle: str, start: int) -> tuple[int, int] | None:
    """Locate `needle` in `hay` from `start`, tolerant to whitespace runs."""
    needle = needle.strip()
    if not needle:
        return None
    pos = hay.find(needle, start)
    if pos >= 0:
        return pos, pos + len(needle)
    pattern = r"\s+".join(re.escape(tok) for tok in needle.split())
    for frm in (start, 0):
        m = re.compile(pattern).search(hay, frm)
        if m:
            return m.start(), m.end()
    return None


def load_provenance() -> dict[tuple[int, int, int], str]:
    """(Verlauf-ID, Ordnungsnummer, Sentence_Index) -> original 2022 category.

    Only messages whose category count in the 2022 workbook equals the number of
    exploded rows can be aligned unit by unit; the rest stay unknown.
    """
    raw = pd.read_excel(RAW_XLSX, sheet_name="Datensammlung", header=2)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.dropna(subset=["Verlauf-ID"])
    out: dict[tuple[int, int, int], str] = {}
    for _, r in raw.iterrows():
        cats = [c.strip() for c in str(r["Kategorie"]).split("\n")]
        out[(int(r["Verlauf-ID"]), int(r["Ordnungsnummer"]))] = cats
    return out


def load_human() -> pd.DataFrame:
    df = pd.read_excel(XLSX)
    df["code_de"] = df["Kategorie"].str.split(" | ", regex=False).str[0]
    df["role"] = df["B/K"]
    df["conversation_id"] = VID_OFFSET + df["Verlauf-ID"]
    raw_cats = load_provenance()
    prov, orig = [], []
    for (v, o), g in df.groupby(["Verlauf-ID", "Ordnungsnummer"]):
        cats = raw_cats.get((int(v), int(o)))
        for i, _ in enumerate(g.sort_values("Sentence_Index").index):
            if cats is None or len(cats) != len(g):
                prov.append("unaligned"); orig.append(None)
            else:
                c = cats[i]
                orig.append(c)
                prov.append("filled" if c.strip().lower() in FILLED_RAW else "annotated")
    order = df.sort_values(["Verlauf-ID", "Ordnungsnummer", "Sentence_Index"]).index
    df.loc[order, "provenance"] = prov
    df.loc[order, "category_2022"] = orig
    return df


def load_pipeline() -> dict:
    rows = json.loads(CLS.read_text(encoding="utf-8"))
    return {(int(r["msg_learn_counselling_id"]), int(r["msg_message_number"])): r
            for r in rows if "E01" in (r.get("source_file") or "")}


def build_units(human: pd.DataFrame, pipe: dict) -> pd.DataFrame:
    """One row per human-annotated segment, with its char span in the message."""
    de2en = {k: v for k, v in json.loads(MAP.read_text()).items() if not k.startswith("_")}
    recs = []
    for (vid, ordn), g in human.groupby(["Verlauf-ID", "Ordnungsnummer"]):
        cid = VID_OFFSET + int(vid)
        msg = pipe.get((cid, int(ordn))) if vid <= MAX_VID else None
        text = canon(msg["msg_content"]) if msg is not None else None
        cursor = 0
        for r in g.sort_values("Sentence_Index").itertuples():
            seg = canon(r.Nachricht).strip()
            span = find_span(text, seg, cursor) if text is not None else None
            if span is not None:
                cursor = span[1]
            recs.append(dict(
                verlauf_id=vid, conversation_id=cid, message_id=ordn,
                sentence_index=r.sentence_index if hasattr(r, "sentence_index") else r.Sentence_Index,
                in_corpus=msg is not None, role=r.role, provenance=r.provenance,
                category_2022=r.category_2022, code_de=r.code_de, gold=de2en.get(r.code_de),
                start=span[0] if span else -1, end=span[1] if span else -1,
                text=text[span[0]:span[1]] if span else seg,
            ))
    out = pd.DataFrame(recs)
    # a unique row id: (verlauf, message, sentence_index) is not unique, two
    # conversations number two different messages alike
    out.insert(0, "unit_id", np.arange(len(out)))
    return out


def segmentation_stats(units: pd.DataFrame, pipe: dict):
    """Compare human spans with the pipeline's SaT spans, message by message.

    Spans are compared in a whitespace-insensitive coordinate (non-whitespace
    characters before a position), so a cut at the same word counts as identical
    regardless of which side the blank ends up on.
    """
    per_msg, link = [], []
    located = units[(units.start >= 0) & units.in_corpus]
    for (cid, mid), g in located.groupby(["conversation_id", "message_id"]):
        msg = pipe[(int(cid), int(mid))]
        text = canon(msg["msg_content"])
        nws = np.cumsum([0] + [0 if c.isspace() else 1 for c in text])
        sat, cursor = [], 0
        for s in msg["sentence_classification"]:
            span = find_span(text, canon(s["text"]), cursor)
            if span is None:
                continue
            cursor = span[1]
            sat.append((int(nws[span[0]]), int(nws[span[1]]), s["predicted_label"]))
        if not sat:
            continue
        h = [(int(nws[a]), int(nws[b])) for a, b in zip(g.start, g.end)]
        sat_spans = {(b0, b1) for b0, b1, _ in sat}
        hb = {e for _, e in h[:-1]}
        sb = {e for _, e, _ in sat[:-1]}
        per_msg.append(dict(
            conversation_id=cid, message_id=mid, role=g.role.iloc[0], chars=int(nws[-1]),
            n_human=len(h), n_sat=len(sat), exact=sum(1 for a in h if a in sat_spans),
            b_tp=len(hb & sb), b_human=len(hb), b_sat=len(sb),
        ))
        for idx, (a0, a1) in enumerate(h):
            ov = [(min(a1, b1) - max(a0, b0), b0, b1, lab) for b0, b1, lab in sat]
            best = max(ov, key=lambda t: t[0])
            link.append(dict(
                conversation_id=cid, message_id=mid,
                sentence_index=g.sentence_index.iloc[idx],
                sat_label=best[3] if best[0] > 0 else None,
                exact_span=(a0, a1) == (best[1], best[2]),
                n_sat_overlapping=sum(1 for o in ov if o[0] > 0),
                human_chars=a1 - a0, sat_chars=best[2] - best[1],
            ))
    return pd.DataFrame(per_msg), pd.DataFrame(link)


def reclassify(units: pd.DataFrame, batch_size: int) -> list[str]:
    """Run the OnCoCo classifier on the human units, prefixed as in the pipeline."""
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


def tier(code: str, depth: int) -> str:
    return "-".join(str(code).split("-")[:depth])


def score_block(name: str, gold: pd.Series, pred: pd.Series) -> dict:
    maj = gold.value_counts(normalize=True).max() if len(gold) else float("nan")
    return dict(
        comparison=name, n=len(gold),
        accuracy=float((gold == pred).mean()),
        kappa=kappa(gold, pred),
        majority_baseline=float(maj),
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
                         precision=p, recall=r,
                         f1=2 * p * r / (p + r) if tp and p + r else 0.0))
    return pd.DataFrame(rows).sort_values("human_n", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reclassify", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    human = load_human()
    pipe = load_pipeline()
    units = build_units(human, pipe)
    print(f"human units: {len(units)}  in corpus: {int(units.in_corpus.sum())}  "
          f"span located: {int((units.start >= 0).sum())}")
    print(units.provenance.value_counts().to_string())

    per_msg, link = segmentation_stats(units, pipe)
    units = units.merge(link, on=["conversation_id", "message_id", "sentence_index"], how="left")

    seg = pd.Series(dict(
        messages=len(per_msg),
        human_segments=int(per_msg.n_human.sum()),
        sat_segments=int(per_msg.n_sat.sum()),
        segments_per_message_human=per_msg.n_human.mean(),
        segments_per_message_sat=per_msg.n_sat.mean(),
        chars_per_segment_human=per_msg.chars.sum() / per_msg.n_human.sum(),
        chars_per_segment_sat=per_msg.chars.sum() / per_msg.n_sat.sum(),
        messages_same_segment_count=float((per_msg.n_human == per_msg.n_sat).mean()),
        messages_sat_coarser=float((per_msg.n_sat < per_msg.n_human).mean()),
        messages_sat_finer=float((per_msg.n_sat > per_msg.n_human).mean()),
        exact_span_rate=per_msg.exact.sum() / per_msg.n_human.sum(),
        exact_span_rate_single_segment=(per_msg[per_msg.n_human == 1].exact.sum()
                                        / max((per_msg.n_human == 1).sum(), 1)),
        exact_span_rate_multi_segment=(per_msg[per_msg.n_human > 1].exact.sum()
                                       / max(per_msg[per_msg.n_human > 1].n_human.sum(), 1)),
        boundary_precision=per_msg.b_tp.sum() / max(per_msg.b_sat.sum(), 1),
        boundary_recall=per_msg.b_tp.sum() / max(per_msg.b_human.sum(), 1),
    ))
    seg["boundary_f1"] = (2 * seg.boundary_precision * seg.boundary_recall
                          / (seg.boundary_precision + seg.boundary_recall))
    seg.to_csv(OUT / "oncoco_human_agreement_segmentation.csv", header=False)
    print(seg.to_string())

    rows = []
    ok = units[units.in_corpus & (units.start >= 0)].dropna(subset=["gold", "sat_label"])
    ex = ok[ok.exact_span]
    rows.append(score_block("pipeline units, exact span match", ex.gold, ex.sat_label))
    rows.append(score_block("pipeline units, max-overlap match", ok.gold, ok.sat_label))

    if args.reclassify:
        loc = units.dropna(subset=["gold"]).copy()
        loc["pred"] = reclassify(loc, args.batch_size)
        units = units.merge(loc[["unit_id", "pred"]], on="unit_id", how="left")
        inc = loc[loc.in_corpus]
        rows.append(score_block("human units, reclassified", inc.gold, inc.pred))
        for role, name in (("B", "counselor"), ("K", "client")):
            s = inc[inc.role == role]
            rows.append(score_block(f"human units, reclassified, {name}", s.gold, s.pred))
        for prov in ("annotated", "filled"):
            s = inc[inc.provenance == prov]
            if len(s):
                rows.append(score_block(f"human units, reclassified, {prov} 2022", s.gold, s.pred))
        out_only = loc[~loc.in_corpus]
        if len(out_only):
            rows.append(score_block("human units, reclassified, Verlauf 65-76 (not in corpus)",
                                    out_only.gold, out_only.pred))
        per_label_scores(inc.gold, inc.pred).to_csv(
            OUT / "oncoco_human_agreement_per_label.csv", index=False)
        conf = (inc[inc.gold != inc.pred].groupby(["gold", "pred"]).size()
                .sort_values(ascending=False).rename("n").reset_index())
        conf["share_of_errors"] = conf.n / conf.n.sum()
        conf.head(40).to_csv(OUT / "oncoco_human_agreement_confusions.csv", index=False)

    # Lenient message-level check: does the pipeline's single message label appear
    # anywhere in the human annotation of that message?
    msg_label = {k: v["message_classification"]["predicted_label"] for k, v in pipe.items()}
    hit = tot = 0
    for (cid, mid), g in units[units.in_corpus].dropna(subset=["gold"]).groupby(
            ["conversation_id", "message_id"]):
        lbl = msg_label.get((int(cid), int(mid)))
        if lbl is None:
            continue
        tot += 1
        hit += lbl in set(g.gold)
    rows.append(dict(comparison="message unit, model label within human label set",
                     n=tot, accuracy=hit / tot if tot else float("nan")))

    lab = pd.DataFrame(rows)
    lab.to_csv(OUT / "oncoco_human_agreement_labels.csv", index=False)
    print(lab.to_string(index=False))
    units.to_csv(OUT / "oncoco_human_agreement_units.csv", index=False)


if __name__ == "__main__":
    main()
