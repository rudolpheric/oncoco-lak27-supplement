#!/usr/bin/env python3
"""Redact personal data in the release corpus with openai/privacy-filter (stage 2 of the data release).

Input is the UNREDACTED intermediate written by build_release_corpus.py. Output is the released
JSONL with every detected span replaced by a typed placeholder (<PRIVATE_PERSON>, <PRIVATE_DATE>,
<PRIVATE_ADDRESS>, <PRIVATE_EMAIL>, <PRIVATE_PHONE>, <PRIVATE_URL>, <ACCOUNT_NUMBER>, <SECRET>) and the
OnCoCo span offsets shifted accordingly, so the labels still align with the released text.

Detection is run twice and the spans are unioned:
  pass 1  the whole conversation in one call, so that the model sees who introduced a name;
  pass 2  overlapping blocks of eight messages, a second look with local context that catches
          mentions the long-context pass missed.
The raw model spans are written to --spans-cache; a later run with the same cache re-applies the
rules below without re-running the model.

Rules applied on top of the model output (the model is trained mainly on English and over-detects
German common nouns as names):
  allowlist   tokens of the conversation's persona name, persona profile and prompt templates are
              never masked: the personas are fictional and published anyway. Mentions of fictional
              relatives named in the persona story ("Max", "Jan") stay readable.
  stoplist    a short list of strings that the manual review of every masked string found to be no
              personal data at all (role nouns, greetings, common nouns, welfare organisations),
              in scripts/release/redaction_stoplist.json.
  token level a person span is masked token by token: allowlisted or stoplisted tokens are kept,
              the rest is replaced. "LG Jenny" becomes "LG <PRIVATE_PERSON>".
  shape       spans found only by the block pass must look like a name (one to three
              capitalized tokens).
  consistency a string masked as a person, e-mail, phone, address, URL, account number or secret
              anywhere in a conversation is masked wherever else it occurs verbatim in it.

Requires the `opf` package (https://github.com/openai/privacy-filter) and its checkpoint.

    python scripts/release/redact_corpus.py --input <work>/all.unredacted.jsonl \
        --output <work>/all.redacted.jsonl --report <work>/report.json \
        --review-log <work>/review.jsonl --spans-cache <work>/model_spans.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEP = "\n\n"
BLOCK, STRIDE = 8, 6  # block size and stride (in messages) of the local-context pass
CONSISTENT_LABELS = {"private_person", "private_email", "private_phone", "private_address", "private_url",
                     "account_number", "secret"}
SALUTATIONS = {"frau", "herr", "herrn", "fr.", "hr.", "dr.", "von", "van", "de", "der", "zu", "ten"}
TOKEN_RE = re.compile(r"[\wäöüÄÖÜß'\-\.]+", re.U)
CAP_RE = re.compile(r"[A-ZÄÖÜ][\wäöüß\-]{2,}")


def placeholder_for(label: str) -> str:
    return "<" + label.upper() + ">"


def norm(tok: str) -> str:
    return tok.strip(" .,;:!?'\"()[]").lower()


def caps(text: str) -> set[str]:
    return {t.lower() for t in CAP_RE.findall(text)}


def merge(spans):
    """Union overlapping spans; the longer span decides the label."""
    out: list[list] = []
    for s, e, lab, src in sorted(spans):
        if out and s < out[-1][1]:
            prev = out[-1]
            if e - s > prev[1] - prev[0]:
                prev[2] = lab
            prev[1] = max(prev[1], e)
            prev[3] = prev[3] if prev[3] == src else "both"
        else:
            out.append([s, e, lab, src])
    return [tuple(x) for x in out]


def map_pos(p: int, repl, is_end: bool) -> int:
    delta = 0
    for s, e, ph in repl:
        if p <= s:
            break
        if p >= e:
            delta += len(ph) - (e - s)
            continue
        return s + delta + (len(ph) if is_end else 0)
    return p + delta


def apply(content: str, repl) -> str:
    pieces, last = [], 0
    for s, e, ph in repl:
        pieces.append(content[last:s])
        pieces.append(ph)
        last = e
    pieces.append(content[last:])
    return "".join(pieces)


# ---------------------------------------------------------------- detection
def detect(opf, contents: list[str]) -> list[list]:
    per_msg: list[list] = [[] for _ in contents]
    offsets, pos = [], 0
    for c in contents:
        offsets.append(pos)
        pos += len(c) + len(SEP)
    for sp in opf.redact(SEP.join(contents)).detected_spans:
        for i, off in enumerate(offsets):
            s, e = max(sp.start, off), min(sp.end, off + len(contents[i]))
            if s < e:
                per_msg[i].append((s - off, e - off, sp.label, "conv"))
    starts = list(range(0, max(1, len(contents) - BLOCK + 1), STRIDE))
    if starts[-1] + BLOCK < len(contents):
        starts.append(len(contents) - BLOCK)
    for lo in starts:
        block = contents[lo:lo + BLOCK]
        spans = opf.redact(SEP.join(block)).detected_spans
        off = 0
        for j, c in enumerate(block):
            for sp in spans:
                s, e = max(sp.start, off), min(sp.end, off + len(c))
                if s < e:
                    per_msg[lo + j].append((s - off, e - off, sp.label, "msg"))
            off += len(c) + len(SEP)
    return per_msg


# ---------------------------------------------------------------- rules
def name_like_strict(txt: str) -> bool:
    toks = [t for t in (norm(x) for x in txt.split()) if t]
    if not toks or len(toks) > 3:
        return False
    raw = [x.strip(" .,;:!?'\"()[]") for x in txt.split()]
    return all(t and (t[0].isupper() or t.lower() in SALUTATIONS) for t in raw)


def filter_spans(content: str, spans, allow: set[str], stop: dict[str, set[str]], stats) -> list:
    """Apply allowlist, stoplist, token-level splitting and the shape rule. Returns final spans."""
    out = []
    for s, e, lab, src in merge(spans):
        txt = content[s:e]
        if lab == "private_person":
            if src == "msg" and not name_like_strict(txt):
                stats["rejected_shape_msg_only"] += 1
                continue
            keep_any = False
            for m in TOKEN_RE.finditer(txt):
                tok = norm(m.group())
                if not tok or tok in SALUTATIONS or tok in allow or tok in stop["private_person"]:
                    stats["tokens_kept_readable"] += 1
                    continue
                if tok.isdigit():
                    continue
                out.append((s + m.start(), s + m.end(), lab, src))
                keep_any = True
            if not keep_any:
                stats["spans_fully_allowlisted"] += 1
        else:
            if txt.strip().lower() in stop.get(lab, set()):
                stats["stoplisted_" + lab] += 1
                continue
            out.append((s, e, lab, src))
    return merge(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--review-log", required=True, help="local-only: every masked string, for manual review")
    ap.add_argument("--spans-cache", required=True, help="raw model spans; written when the model runs, read otherwise")
    ap.add_argument("--templates-dir", default=str(REPO / "prompts/templates"))
    ap.add_argument("--stoplist", default=str(REPO / "scripts/release/redaction_stoplist.json"))
    ap.add_argument("--shard", default=None, help="k/n: process every n-th conversation starting at k (0-based)")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    stop = {k: {x.lower() for x in v} for k, v in json.loads(Path(args.stoplist).read_text(encoding="utf-8")).items()}
    tpl_text: dict[str, str] = collections.defaultdict(str)
    for p in Path(args.templates_dir).glob("T*.md"):
        t = p.read_text(encoding="utf-8")
        m = re.search(r"^persona: (.*)$", t, re.M)
        if m:
            tpl_text[m.group(1).strip()] += t

    convs = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        convs = convs[k::n]

    cache_path = Path(args.spans_cache)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for l in cache_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                cache[r["conversation_id"]] = r["spans"]
    opf = None
    if any(c["conversation_id"] not in cache for c in convs):
        import torch
        from opf import OPF
        torch.set_num_threads(args.threads)
        opf = OPF(device=args.device, output_mode="typed")
    cache_f = cache_path.open("a", encoding="utf-8")

    stats = collections.Counter()
    by_label = collections.Counter()
    t0 = time.time()
    out_f = Path(args.output).open("w", encoding="utf-8")
    log_f = Path(args.review_log).open("w", encoding="utf-8")

    for n, conv in enumerate(convs, 1):
        cid = conv["conversation_id"]
        msgs = conv["messages"]
        contents = [m["content"] for m in msgs]
        persona = conv.get("persona") or {}
        pname = persona.get("name") or ""
        allow = caps(pname) | caps(json.dumps(persona.get("profile") or {}, ensure_ascii=False)) | caps(tpl_text.get(pname, ""))
        allow |= {norm(t) for t in pname.split() if norm(t)}

        if cid in cache:
            per_msg = [[tuple(x) for x in cache[cid].get(str(i), [])] for i in range(len(msgs))]
        else:
            per_msg = detect(opf, contents)
            cache_f.write(json.dumps(dict(conversation_id=cid, spans={str(i): v for i, v in enumerate(per_msg) if v}),
                                     ensure_ascii=False) + "\n")
            cache_f.flush()

        kept = [filter_spans(contents[i], spans, allow, stop, stats) for i, spans in enumerate(per_msg)]

        # consistency: propagate masked strings within the conversation
        consistent: dict[str, str] = {}
        for i, spans in enumerate(kept):
            for s, e, lab, _ in spans:
                txt = contents[i][s:e].strip()
                if lab in CONSISTENT_LABELS and len(txt) >= 3 and not txt.isdigit():
                    consistent.setdefault(txt, lab)
        for i, c in enumerate(contents):
            for txt, lab in consistent.items():
                for m in re.finditer(r"(?<!\w)" + re.escape(txt) + r"(?!\w)", c):
                    if not any(s < m.end() and m.start() < e for s, e, _, _ in kept[i]):
                        kept[i].append((m.start(), m.end(), lab, "consistency"))
            kept[i] = merge(kept[i])

        for i, msg in enumerate(msgs):
            repl = [(s, e, placeholder_for(lab)) for s, e, lab, _ in kept[i]]
            if repl:
                for s, e, lab, src in kept[i]:
                    by_label[lab] += 1
                    stats["source_" + src] += 1
                    log_f.write(json.dumps(dict(conversation_id=cid, message_number=msg["message_number"], label=lab,
                                                text=contents[i][s:e], source=src), ensure_ascii=False) + "\n")
                msg["content"] = apply(contents[i], repl)
                msg["pii_placeholders"] = dict(collections.Counter(lab for _, _, lab, _ in kept[i]))
                if "oncoco_spans" in msg:
                    for sp in msg["oncoco_spans"]:
                        sp["start"] = map_pos(sp["start"], repl, False)
                        sp["end"] = map_pos(sp["end"], repl, True)
                stats["messages_redacted"] += 1
            stats["messages"] += 1
            for txt in consistent:
                assert not re.search(r"(?<!\w)" + re.escape(txt) + r"(?!\w)", msg["content"]), (cid, txt)
            if "oncoco_spans" in msg:
                L = len(msg["content"])
                assert all(0 <= sp["start"] <= sp["end"] <= L for sp in msg["oncoco_spans"]), cid
        stats["conversations"] += 1
        out_f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        out_f.flush()
        if n % 10 == 0 or n == len(convs):
            print(f"[{n}/{len(convs)}] {time.time() - t0:.0f}s  redacted msgs={stats['messages_redacted']}  {dict(by_label)}",
                  flush=True)

    out_f.close()
    log_f.close()
    cache_f.close()
    Path(args.report).write_text(json.dumps(dict(stats=dict(stats), by_label=dict(by_label),
                                                 seconds=round(time.time() - t0)), indent=1), encoding="utf-8")
    print("done", dict(stats), dict(by_label))


if __name__ == "__main__":
    main()
