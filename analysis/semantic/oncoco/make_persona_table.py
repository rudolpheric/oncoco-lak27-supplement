#!/usr/bin/env python3
"""Generate the persona table from the raw exports instead of maintaining it by hand.

The hand-maintained results/tables/persona_summary.tex had four defects that a reviewer would
notice: platform persona ids repeat across exports (6/7/8/9 each appeared twice), one row carried
another persona's description, two descriptions were empty, and several were cut mid-sentence.
Platform ids are per-export counters and are not usable as a key; personas are therefore
de-duplicated on (name, main concern).

Scope: the personas actually used by the ANALYSED conversations, i.e. HH roleplay chat and H--LLM
roleplay chat. HH real chat has no personas -- those are real help-seekers.

Outputs: results/tables/persona_summary.tex   (Supplementary Table S1)
         results/tables/oncoco_personas.csv
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SIMILARITY = 0.85


def brief_key(s: str) -> str:
    """Diacritic- and transliteration-insensitive key, so 'regelmäßig' and 'regelmaessfig' match."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"(ae|oe|ue|ss)", "", s)
    return re.sub(r"[^a-z0-9]", "", s)

ROOT = Path(__file__).resolve().parents[3]
NORM = ROOT / "data/processed/combined/normalized"
TABLES = ROOT / "results/tables"

PERSONA_CONDITIONS = ["HH_roleplay_chat", "H_LLM_roleplay_chat"]

LATEX = {"ä": r'\"a', "ö": r'\"o', "ü": r'\"u', "Ä": r'\"A', "Ö": r'\"O', "Ü": r'\"U',
         "ß": r"\ss{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
         "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
         "\\": r"\textbackslash{}", "„": "``", "“": "''", "”": "''", "–": "--", "—": "---",
         "’": "'", "‘": "'"}


def tex_escape(s: str) -> str:
    out = []
    for ch in s:
        if ch in LATEX:
            out.append(LATEX[ch])
        elif unicodedata.category(ch) == "Cf" or ord(ch) > 0x2100:
            continue  # drop emoji / formatting marks
        else:
            out.append(ch)
    return "".join(out)


def first_sentence(text: str, max_chars: int = 160) -> str:
    """First sentence, cut at a boundary -- never mid-word, never mid-sentence."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    m = re.search(r"(?<=[.!?])\s", text)
    s = text[: m.start()] if m else text
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "\\ldots"
    return s


def raw_path(source_file: str) -> Path | None:
    alt = source_file.replace("/chats/", "/chats_filtered/") if "human_llm" in source_file else source_file
    for cand in (alt, source_file):
        if (ROOT / cand).exists():
            return ROOT / cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-json", default=str(NORM / "oncoco_classification_all.json"))
    args = ap.parse_args()

    rows = json.loads(Path(args.data_json).read_text(encoding="utf-8"))
    analysed = defaultdict(set)  # source_file -> {conversation id}
    for r in rows:
        if r.get("source") in PERSONA_CONDITIONS:
            analysed[r["source_file"]].add(str(r["id"]))
    n_analysed = sum(len(v) for v in analysed.values())
    print(f"{n_analysed} analysed conversations across {len(analysed)} source files")

    personas: dict[tuple[str, str], dict] = {}
    conv_counts: Counter = Counter()
    missing = 0
    for sf, ids in analysed.items():
        p = raw_path(sf)
        if p is None:
            print(f"  WARNING: raw export not found for {sf}")
            continue
        for c in json.loads(p.read_text(encoding="utf-8")):
            if str(c.get("id")) not in ids:
                continue
            pers = c.get("persona")
            if not pers:
                missing += 1
                continue
            props = pers.get("properties") or {}
            concern = props.get("Hauptanliegen") or ""
            if isinstance(concern, list):
                concern = " ".join(concern)
            name = (pers.get("name") or "").strip()
            key = (name.lower(), concern.strip()[:120].lower())
            sk = props.get("Steckbrief") or {}
            personas.setdefault(key, dict(
                name=name, concern=concern.strip(),
                age=sk.get("Alter"), gender=sk.get("Geschlecht"), job=sk.get("Job"),
                platform_ids=set(), source_files=set()))
            personas[key]["platform_ids"].add(pers.get("id"))
            personas[key]["source_files"].add(sf)
            conv_counts[key] += 1

    print(f"{len(personas)} distinct personas (deduplicated on name + main concern); "
          f"{missing} analysed conversations carry no persona record")

    recs = []
    for key, p in personas.items():
        recs.append(dict(name=p["name"], age=p["age"], gender=p["gender"], job=p["job"],
                         n_conversations=conv_counts[key],
                         concern_first_sentence=first_sentence(p["concern"]),
                         concern_full=p["concern"],
                         platform_ids=";".join(str(x) for x in sorted(
                             v for v in p["platform_ids"] if v is not None)),
                         n_source_files=len(p["source_files"])))
    df = pd.DataFrame(recs)

    # The platform reuses a case brief under different persona names, sometimes without updating
    # the brief text, and some briefs exist in lightly edited (typo) variants. The identity of a
    # case study is therefore its BRIEF, not its name. Cluster on a diacritic- and
    # transliteration-insensitive normalisation of the brief; the count is stable for every
    # threshold in 0.70-0.95, so it does not hinge on where the cut is placed.
    df["_norm"] = df["concern_full"].map(brief_key)
    reps: list[str] = []
    case_of: list[int] = []
    for v in df["_norm"]:
        hit = next((i for i, r in enumerate(reps)
                    if difflib.SequenceMatcher(None, r, v).ratio() >= SIMILARITY), None)
        if hit is None:
            reps.append(v)
            hit = len(reps) - 1
        case_of.append(hit)
    df["case"] = case_of

    cases = (df.groupby("case")
               .agg(names=("name", lambda s: sorted(set(s))),
                    n_conversations=("n_conversations", "sum"),
                    brief=("concern_first_sentence", "first"))
               .sort_values("n_conversations", ascending=False).reset_index(drop=True))
    cases.insert(0, "case_id", range(1, len(cases) + 1))
    df = df.merge(cases[["case_id", "brief"]].assign(case=range(len(cases))), on="case", how="left")
    df.drop(columns=["_norm"]).to_csv(TABLES / "oncoco_personas.csv", index=False)
    cases.assign(names=cases["names"].map("; ".join)).to_csv(
        TABLES / "oncoco_persona_cases.csv", index=False)

    print(f"\n{len(df)} name/brief combinations reduce to {len(cases)} distinct case briefs "
          f"(similarity {SIMILARITY})")
    shared = cases[cases["names"].map(len) > 1]
    for _, r in shared.iterrows():
        print(f"  case {r['case_id']} is used under {len(r['names'])} names "
              f"({', '.join(r['names'])}) in {r['n_conversations']} conversations")
    def _mismatched(g: pd.DataFrame) -> int:
        # Conversations whose persona name differs from the name the brief text itself
        # refers to. Aggregate per NAME first (a name can appear on several brief
        # variants), and subtract the referent name's conversations -- not the modal
        # name's, which is only the same thing when the brief happens to name the
        # most frequently used persona.
        per_name = g.groupby("name")["n_conversations"].sum()
        brief = " ".join(str(x) for x in g["concern_full"].fillna("")).lower()
        referents = [n for n in per_name.index if str(n).lower() in brief]
        if referents:
            keep = max(per_name[n] for n in referents)
        else:
            keep = per_name.max()
        return int(per_name.sum() - keep)

    n_mismatch = int(sum(_mismatched(g) for _, g in df.groupby("case")
                         if g["name"].nunique() > 1))
    print(f"  -> {n_mismatch} of {n_analysed} conversations use a brief under a name other than "
          f"the one the brief itself refers to")

    empty = cases[cases["brief"].str.len() == 0]
    if len(empty):
        print(f"\n{len(empty)} cases have no brief recorded -- shown as an explicit note in the table.")

    lines = [r"\begin{tabular}{r p{0.24\linewidth} r p{0.52\linewidth}}", r"\toprule",
             r"\# & Persona name(s) & Conv. & Case brief (first sentence) \\", r"\midrule"]
    for _, r in cases.iterrows():
        desc = tex_escape(r["brief"]) or r"\emph{not recorded}"
        names = ", ".join(tex_escape(n) for n in r["names"])
        lines.append(f"{r['case_id']} & {names} & {r['n_conversations']} & {desc} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "persona_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nTotal conversations covered: {int(cases['n_conversations'].sum())} of {n_analysed}")
    print(f"Wrote {TABLES / 'persona_summary.tex'}, oncoco_personas.csv and oncoco_persona_cases.csv")
    print(f"\n>>> Section 4.3 should say: {len(cases)} distinct case studies "
          f"({len(df)} persona name/brief combinations) <<<")


if __name__ == "__main__":
    main()
