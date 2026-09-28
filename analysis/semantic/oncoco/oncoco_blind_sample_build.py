#!/usr/bin/env python3
"""Blinded expert-coding sample of client spans from all three chat conditions.

Answers whether the classifier compresses the distance from real counseling to simulated
clients as much as it compresses the distance to human roleplay. One coder labels pipeline
SaT spans from HH real, HH roleplay and H-LLM without seeing condition, model or classifier
label, in random order. Each unit then carries two labels, the coder's and the pipeline's,
so every distance and Delta_CL can be computed twice on identical units
(oncoco_blind_sample_analysis.py).

Design: simple random samples of client spans within strata, H-LLM 100 per model,
HH real 150, HH roleplay 150 (600 units). Wave 1 (rows 1-300) is itself a stratified half
sample, so the analysis also works if only wave 1 is coded.

Outputs (git-ignored, contain real counseling text, never publish):
  annotations/manual/human_llm/blind_sample_v1.xlsx       coding sheet, instructions, categories
  annotations/manual/human_llm/blind_sample_v1_key.csv    hidden key (condition, model, ids, pipeline label)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NORM = ROOT / "data/processed/combined/normalized/oncoco_classification_all.json"
OUTDIR = ROOT / "annotations/manual/human_llm"
STRATA = [  # (stratum, condition, model, n)
    ("HH_real", "HH_real_chat", None, 150),
    ("HH_roleplay", "HH_roleplay_chat", None, 150),
    ("H_LLM_GPT_OSS_120B", "H_LLM_roleplay_chat", "GPT_OSS_120B", 100),
    ("H_LLM_Llama3_3_70B", "H_LLM_roleplay_chat", "LLama3_3_70B", 100),
    ("H_LLM_Mixtral", "H_LLM_roleplay_chat", "Mixtral", 100),
]
UNCODABLE = "nicht kodierbar / keine Klientenkategorie | X"
# German category names of the client side, in the order of the scheme, with the definitions of the
# GeCCo codebook (data/GeCCo Kategoriensystem mit KurzBez.xlsx, sheet "Klient (2)"), from which OnCoCo
# grew, with obvious typos corrected. Names follow the MAXQDA codebook of the HH real annotation
# (oncoco_maxqda_label_map.json); final failure and final success follow the 2022 codebook.
# Two categories have no GeCCo definition and carry a marked hint instead.
CATEGORIES_DE = [
    ("K-FA-*-*-A-*", "Anrede / Formalitäten am Anfang", "Allgemeine Anrede."),
    ("K-FZ-*-*-F-*", "Formales zum Abschluss einer Beratung",
     "Im GeCCo-Codebuch ohne Definition. Im Codebuch 2022 als „Verabschiedung“ geführt."),
    ("K-FZ-*-*-NPR-*", "(weitere) Nutzung professioneller Ressourcen der zu beratenden Person",
     "Der Klient stimmt einem weiterführenden Beratungsangebot zu."),
    ("K-FA-*-E-ECM-*", "Mitgefühl mit anderen (EC)",
     "Der Klient drückt deutlich Mitgefühl mit einer anderen Person aus. Dabei kommt es darauf an, dass er explizit "
     "beschreibt, dass die Situation der Person bei ihm eine Emotion auslöst."),
    ("K-FA-*-E-ECS-*", "Sorge um andere Person (EC)",
     "Der Klient drückt deutlich Sorge um eine andere Person aus seinem Umfeld aus. Mit Sorge ist gemeint, dass er das "
     "Bedürfnis hat, die Person schützen zu müssen oder mehr für sie zu tun, als im Rahmen dieses Settings möglich ist. "
     "Die Verwendung des Wortes „Sorge“ kann hier ein Hinweis sein, muss es aber nicht."),
    ("K-FA-*-E-PT-*", "Empathie für Dritte bzw. bezogen auf die vorliegende Situation (PT)",
     "Situationsbetrachtung und Hineinversetzen in Dritte. Mitteilen von dem, was in Bezug auf Dritte oder die "
     "spezifische Situation emotional verstanden wurde. Entspricht „Perspective Taking“ in der Theorie."),
    ("K-WF-AKP-*-PDar-*", "Problemdarstellung",
     "Der Klient beschreibt das Problem, wegen dem er Rat sucht, allgemein. Hier kann die gesamte Problembeschreibung "
     "codiert werden."),
    ("K-WF-AKP-*-PDef-*", "Problemdefinition",
     "Der Klient definiert pointiert, was aus seiner Sicht das Problem ist, wegen dem er sich meldet. Das kann ein Fazit "
     "aus der Problembeschreibung sein oder auch der Ausschluss bestimmter Fragen."),
    ("K-WF-AKP-*-PPers-*", "Preisgeben persönlicher Daten",
     "Der Klient stellt sich und seine persönlichen Daten (Alter, Name etc.) vor."),
    ("K-WF-AKP-*-EG-*", "Eigene Gefühlsdarstellung", "Der Klient berichtet, wie es ihm aktuell geht."),
    ("K-WF-AKP-*-RBL-*", "Rückmeldung zu bisherigen Lösungsversuchen",
     "Der Klient erläutert Lösungsversuche, die er bisher vorgenommen hat, um mit dem Problem umgehen zu können."),
    ("K-WF-AKP-*-Nachf-*", "Allgemeine Nachfrage",
     "Der Klient stellt Nachfragen, um Handlungsempfehlungen, Deutungen/Bewertungen etc. besser zu verstehen."),
    ("K-WF-AKP-*-Zust-*", "Zustimmung", "Der Klient stimmt einer Entscheidungsfrage zu."),
    ("K-WF-AKP-*-Abl-*", "Ablehnung", "Der Klient lehnt eine Entscheidungsfrage ab."),
    ("K-WF-ABZ-*-Ziel-*", "Zielsetzung des Auftrags",
     "Der Klient formuliert die Zielsetzung des Beratungsprozesses klar und deutlich, auch in Abgrenzung zu "
     "Zielsetzungen, die in diesem Format nicht realisierbar sind. Eine Zustimmung zu einem Vorschlag zur Zielsetzung, "
     "der vom Berater kam, gilt hier ebenfalls."),
    ("K-WF-ABZ-*-Erw-*", "Erweiterung des Auftrags",
     "Der Klient fügt der vorher definierten Zielsetzung weitere Aspekte hinzu. Das kann sowohl in der initialen "
     "Auftragsklärung als auch zu einem späteren Zeitpunkt erfolgen."),
    ("K-WF-HP-*-PosR-*", "Allgemeine positive Rückmeldung",
     "Der Klient reagiert positiv auf das Gesagte/Geschriebene der/des Berater*in, mit Ausnahme der "
     "Handlungsempfehlungen."),
    ("K-WF-HP-*-PosRH-*", "Positive Rückmeldung zu spezifischer Handlungsempfehlung",
     "Der Klient nimmt positiv zu einer spezifischen Handlungsempfehlung Stellung. Das bedeutet, er zieht die Umsetzung "
     "dieser Empfehlung ernsthaft in Betracht."),
    ("K-WF-HP-*-NegRH-*", "Negative Rückmeldung zu spezifischer Handlungsempfehlung",
     "Der Klient nimmt negativ zu einer spezifischen Handlungsempfehlung Stellung. Das bedeutet, er verwirft den Rat "
     "und seine Umsetzung."),
    ("K-WF-HP-*-BerUH-*", "Bericht über Umsetzung von Handlungsempfehlung",
     "Der Klient stellt dar, wie er versucht hat, eine Handlungsempfehlung umzusetzen. Die Bewertung wird über die "
     "oberen Kategorien umgesetzt."),
    ("K-WF-HP-*-Erf-*", "Endgültiger Erfolg",
     "Der Klient definiert sein Problem als erledigt bzw. gelöst und beendet die Beratung positiv."),
    ("K-WF-HP-*-MErf-*", "Endgültiger Misserfolg",
     "Der Klient beendet die Beratung mit dem Hinweis, dass sein Problem nicht gelöst ist."),
    ("K-WF-Mot-*-VW-*", "Aussagen zu Veränderungswunsch (MI)", "Der Klient artikuliert einen Änderungswunsch."),
    ("K-WF-Mot-*-GV-*", "Artikulation von Gründen für eine Veränderung beim Klienten",
     "Der Klient nennt Gründe, warum er sich oder sein Verhalten ändern sollte."),
    ("K-WF-RA-*-RF-*", "In Betracht ziehen von Ressourcenaktivierung auf Freundes- und Familienebene",
     "Der Klient zieht in Betracht, sich bei Freunden Unterstützung zur Problembewältigung zu holen."),
    ("K-WF-RA-*-RP-*", "In Betracht ziehen von Ressourcenaktivierung auf professioneller Ebene",
     "Der Klient zieht in Betracht, sich bei professionellen oder semiprofessionellen Stellen Unterstützung zur "
     "Problembewältigung zu holen."),
    ("K-A-*-*-A-*", "Andere Aussagen / sonstige Antwort",
     "Im GeCCo-Codebuch ohne Definition. Restkategorie für Aussagen, die in keine andere Kategorie passen "
     "(Codebuch 2022: „Sonstiges“)."),
    ("K-A-*-*-UB-*", "Unpassende Bemerkung",
     "Aussage, die inhaltlich unangebracht ist. Dies kann dadurch begründet sein, dass sich der*die Klient*in nicht "
     "ernst genommen oder unprofessionell behandelt fühlt."),
]
UNCODABLE_DEF = ("Keine GeCCo-Kategorie. Nur für Abschnitte ohne kodierbare Klientenfunktion, z. B. ein einzelnes "
                 "Satzzeichen oder ein abgeschnittenes Wortfragment.")
INSTRUCTIONS = [
    "Kodierung nach OnCoCo, Klientenseite. Eine Kategorie pro Zeile.",
    "Kodiert wird nur der markierte Abschnitt in »…« (Spalte D wiederholt ihn).",
    "Die vorherige Beraternachricht und die ganze Klientennachricht dienen nur als Kontext.",
    "Die Zeilen stammen aus verschiedenen Gesprächsarten und sind zufällig gemischt. Bitte nicht versuchen, die Herkunft zu erraten.",
    "Wenn keine Klientenkategorie passt oder der Abschnitt keine Äußerung ist: " + UNCODABLE,
    "Spalte F (unsicher) mit x markieren, wenn die Entscheidung knapp war. Spalte G für Kommentare.",
    "Welle 1 (Zeilen 1-300) zuerst. Welle 1 allein ist schon auswertbar.",
    "Spalte I zeigt die Definition der gewählten Kategorie (GeCCo-Codebuch). Alle Definitionen stehen im Blatt „Kategorien“.",
]


def load_client_units(rows):
    """All client spans per condition, with the preceding counselor message as context."""
    convs = collections.defaultdict(list)
    for r in rows:
        if r.get("source") in {s[1] for s in STRATA}:
            convs[(r["source"], r.get("source_file"), str(r.get("id")))].append(r)
    units = collections.defaultdict(list)
    for (cond, sf, cid), ms in convs.items():
        ms.sort(key=lambda m: int(float(m.get("msg_message_number") or 0)))
        last_counselor = ""
        for m in ms:
            if m.get("speaker_type") == "Counsellor":
                last_counselor = m.get("msg_content") or ""
                continue
            if m.get("speaker_type") != "Client":
                continue
            text = m.get("msg_content") or ""
            for s in m.get("sentence_classification") or []:
                if not (s.get("text") or "").strip() or not s.get("predicted_label"):
                    continue
                marked = text[: s["start_offset"]] + "»" + s["text"] + "«" + text[s["end_offset"]:]
                units[(cond, m.get("model") or None)].append(dict(
                    condition=cond, model=m.get("model") or "", source_file=sf, conversation_id=cid,
                    msg_message_number=m.get("msg_message_number"), sentence_index=s.get("sentence_index"),
                    start_offset=s["start_offset"], end_offset=s["end_offset"], span=s["text"],
                    marked_message=marked, counselor_context=last_counselor,
                    pipeline_label=s["predicted_label"]))
    return units


def write_categories(cat) -> None:
    """Category sheet: dropdown text, German code, English OnCoCo code, GeCCo definition."""
    de_en = json.loads((HERE / "oncoco_de_en_label_map.json").read_text(encoding="utf-8"))
    cat.append(["Kategorie (Auswahl)", "Code (deutsch)", "OnCoCo-Code (englisch)", "Definition (GeCCo-Codebuch)"])
    for de, name, definition in CATEGORIES_DE:
        cat.append([f"{name} | {de}", de, de_en[de], definition])
    cat.append([UNCODABLE, "X", "", UNCODABLE_DEF])
    for c in cat[1]:
        c.font = Font(bold=True)
    for col, width in zip("ABCD", (75, 22, 24, 110)):
        cat.column_dimensions[col].width = width
    for row in cat.iter_rows(min_row=2):
        row[3].alignment = Alignment(wrap_text=True, vertical="top")


def add_dropdown_and_definition_column(ws, n_rows: int, n_options: int) -> None:
    """Dropdown on the category column F and, in column I, the definition of the chosen category."""
    dv = DataValidation(type="list", formula1=f"=Kategorien!$A$2:$A${n_options + 1}", allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"F2:F{n_rows + 1}")
    ws.cell(1, 9).value = "Definition der gewählten Kategorie"
    ws.cell(1, 9).font = Font(bold=True)
    ws.cell(1, 9).fill = PatternFill("solid", fgColor="DDDDDD")
    for r in range(2, n_rows + 2):
        c = ws.cell(r, 9)
        c.value = f'=IF(F{r}="","",IFERROR(VLOOKUP(F{r},Kategorien!$A$2:$D${n_options + 1},4,FALSE),""))'
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["I"].width = 70


def patch_definitions(path: Path) -> None:
    """Add definitions to an existing coding sheet in place. Entries in columns A-H stay untouched.

    Excel stores the cross-sheet dropdown in an extension openpyxl drops on load, so the dropdown is
    written again. The option texts are unchanged, so every existing entry stays a valid choice.
    """
    from openpyxl import load_workbook
    wb = load_workbook(path)
    ws = wb["Kodierung"]
    before = [(ws.cell(r, 1).value, ws.cell(r, 6).value, ws.cell(r, 7).value, ws.cell(r, 8).value)
              for r in range(2, ws.max_row + 1)]
    old_options = [ws2 for ws2 in (r[0] for r in wb["Kategorien"].iter_rows(min_row=2, values_only=True)) if ws2]
    del wb["Kategorien"]
    cat = wb.create_sheet("Kategorien")
    write_categories(cat)
    new_options = [r[0] for r in cat.iter_rows(min_row=2, values_only=True)]
    if old_options != new_options:
        raise SystemExit("dropdown texts would change, refusing to patch")
    ws.data_validations.dataValidation = []
    add_dropdown_and_definition_column(ws, len(before), len(new_options))
    after = [(ws.cell(r, 1).value, ws.cell(r, 6).value, ws.cell(r, 7).value, ws.cell(r, 8).value)
             for r in range(2, ws.max_row + 1)]
    assert before == after, "coding entries changed"
    wb.calculation.fullCalcOnLoad = True
    wb.save(path)
    print(f"patched {path}: {sum(1 for b in before if b[1])} coded rows kept, definitions added")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--patch-definitions", default=None,
                    help="only add the category definitions to this existing coding sheet, keeping its entries")
    ap.add_argument("--xlsx", default=None,
                    help="write the coding sheet here instead of blind_sample_<tag>.xlsx; the key must then already "
                         "exist and is checked, not rewritten")
    args = ap.parse_args()
    if args.patch_definitions:
        patch_definitions(Path(args.patch_definitions))
        return
    rng = random.Random(args.seed)
    rows = json.loads(NORM.read_text(encoding="utf-8"))
    units = load_client_units(rows)

    sample = []
    for stratum, cond, model, n in STRATA:
        pool = units[(cond, model)] if model else [u for (c, _), us in units.items() if c == cond for u in us]
        picked = rng.sample(pool, n)
        for i, u in enumerate(picked):
            u.update(stratum=stratum, stratum_population=len(pool), stratum_n=n, wave=1 if i < n // 2 else 2)
        sample.extend(picked)
        print(f"{stratum:20s} population {len(pool):6d} client spans, sampled {n}")
    ordered = []
    for wave in (1, 2):
        w = [u for u in sample if u["wave"] == wave]
        rng.shuffle(w)
        ordered.extend(w)
    for i, u in enumerate(ordered, 1):
        u["unit_no"] = i

    OUTDIR.mkdir(parents=True, exist_ok=True)
    key_path = OUTDIR / f"blind_sample_{args.tag}_key.csv"
    fields = ["unit_no", "wave", "stratum", "stratum_population", "stratum_n", "condition", "model", "source_file",
              "conversation_id", "msg_message_number", "sentence_index", "start_offset", "end_offset",
              "pipeline_label", "span"]
    new_key = OUTDIR / f".blind_sample_{args.tag}_key.tmp.csv"
    with new_key.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)
    if args.xlsx:
        # a second sheet for an existing sample: same seed, so the key must come out identical
        if new_key.read_bytes() != key_path.read_bytes():
            new_key.unlink()
            raise SystemExit(f"resampling does not reproduce {key_path}, refusing to write a mismatched sheet")
        new_key.unlink()
    else:
        new_key.replace(key_path)

    # category list: German name | German code, client side only, in the order of the scheme
    de_en = json.loads((HERE / "oncoco_de_en_label_map.json").read_text(encoding="utf-8"))
    client_codes = {de for de, en in de_en.items() if str(en).startswith("CL-") and de.startswith("K")}
    assert client_codes == {de for de, *_ in CATEGORIES_DE}, "German category list out of sync with the code map"
    cats = [(de, de_en[de]) for de, *_ in CATEGORIES_DE]
    options = [f"{name} | {de}" for de, name, _ in CATEGORIES_DE] + [UNCODABLE]

    wb = Workbook()
    ws = wb.active
    ws.title = "Kodierung"
    head = ["Nr", "Welle", "Vorherige Beraternachricht", "Klientennachricht (Abschnitt in »…«)",
            "Zu kodierender Abschnitt", "Kategorie", "unsicher (x)", "Kommentar"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    for u in ordered:
        ws.append([u["unit_no"], u["wave"], u["counselor_context"], u["marked_message"], u["span"], "", "", ""])
    for col, width in zip("ABCDEFGH", (6, 6, 60, 60, 40, 45, 10, 30)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for c in row[2:5]:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"

    cat = wb.create_sheet("Kategorien")
    write_categories(cat)
    add_dropdown_and_definition_column(ws, len(ordered), len(options))

    ins = wb.create_sheet("Anleitung", 0)
    for line in INSTRUCTIONS:
        ins.append([line])
    ins.column_dimensions["A"].width = 120
    wb.active = 1

    xlsx = Path(args.xlsx) if args.xlsx else OUTDIR / f"blind_sample_{args.tag}.xlsx"
    wb.calculation.fullCalcOnLoad = True
    wb.save(xlsx)
    print("wrote", xlsx, "with key", key_path)


if __name__ == "__main__":
    main()
