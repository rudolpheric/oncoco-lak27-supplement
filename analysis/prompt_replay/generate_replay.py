#!/usr/bin/env python3
"""Regenerate simulated-client turns from replay_requests.jsonl through an OpenAI-compatible endpoint.

Purpose: produce the client turns of arms A/B/C under the platform's sampling settings
(temperature 1, max_tokens 3000, no penalties; logged in additions.llm_info) so that the
only difference between arms is the prompt text.

The prompt is sent as a single user message. The platform did not log its message layout;
arm A (verbatim replay) measures whatever harness difference that assumption introduces.

Resumable: every completed request is appended to the output JSONL immediately and skipped
on the next run, so a preempted job continues where it stopped.

    python generate_replay.py --base-url http://localhost:8000/v1 --model openai/gpt-oss-120b \
        --requests replay_requests.jsonl --output replay_outputs.jsonl --arms A,C
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--arms", default="A,C", help="comma list; empty = all arms in the request file")
    p.add_argument("--limit", type=int, default=0, help="max requests per arm (smoke test)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--seed", type=int, default=None, help="per-request seed offset; omit for none")
    p.add_argument("--role", choices=["user", "system"], default="user",
                   help="message role the prompt is sent under")
    p.add_argument("--reasoning-effort", default=None, choices=[None, "low", "medium", "high"])
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--retries", type=int, default=4)
    return p.parse_args()


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["request_id"])
    return done


async def one(client: AsyncOpenAI, sem: asyncio.Semaphore, req: dict, args: argparse.Namespace,
              fh, lock: asyncio.Lock, counter: dict) -> None:
    extra = {}
    if args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort
    seed = None if args.seed is None else args.seed + hash(req["request_id"]) % 100000
    async with sem:
        err = None
        for attempt in range(args.retries + 1):
            t0 = time.time()
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": args.role, "content": req["prompt"]}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    seed=seed,
                    timeout=args.timeout,
                    extra_body=extra or None,
                )
                choice = resp.choices[0]
                msg = choice.message
                rec = dict(
                    request_id=req["request_id"], arm=req["arm"], conv_id=req["conv_id"],
                    message_number=req["message_number"], variant=req["variant"],
                    persona_name=req["persona_name"], source_file=req["source_file"],
                    content=msg.content or "",
                    reasoning_content=getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
                    finish_reason=choice.finish_reason,
                    usage=resp.usage.model_dump() if resp.usage else None,
                    model=resp.model, role=args.role, seed=seed,
                    latency_s=round(time.time() - t0, 3), created=int(time.time()),
                )
                async with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    counter["done"] += 1
                    n = counter["done"]
                if n % 50 == 0:
                    print(f"  {n}/{counter['total']} done", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                err = e
                await asyncio.sleep(min(2 ** attempt, 30))
        async with lock:
            counter["failed"] += 1
        print(f"FAILED {req['request_id']}: {err!r}", file=sys.stderr, flush=True)


async def run(args: argparse.Namespace) -> None:
    arms = {a.strip() for a in re.split(r"[,;+]", args.arms) if a.strip()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    reqs, per_arm = [], {}
    with Path(args.requests).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if arms and r["arm"] not in arms:
                continue
            if args.limit and per_arm.get(r["arm"], 0) >= args.limit:
                continue
            per_arm[r["arm"]] = per_arm.get(r["arm"], 0) + 1
            if r["request_id"] in done:
                continue
            reqs.append(r)
    print(f"{len(done)} already done; {len(reqs)} to generate (arms {sorted(per_arm)}, "
          f"selected per arm {per_arm}); endpoint {args.base_url} model {args.model}", flush=True)
    if not reqs:
        return
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout, max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counter = dict(done=0, failed=0, total=len(reqs))
    with out.open("a", encoding="utf-8") as fh:
        await asyncio.gather(*(one(client, sem, r, args, fh, lock, counter) for r in reqs))
    print(f"finished: {counter['done']} written, {counter['failed']} failed -> {out}", flush=True)
    if counter["failed"]:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
