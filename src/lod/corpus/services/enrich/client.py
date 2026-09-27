"""The LLM calls: one chat completion per schema, through OpenRouter, many in flight.

Only this module touches the network. The answer is parsed here and checked by
`schemas.clean`; the caller owns the cache it lands in.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Callable

from lod.corpus.services.enrich.prompts import build_prompt
from lod.corpus.services.enrich.schemas import clean, parse_json

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
RETRY_STATUS = (429, 500, 502, 503, 520, 524)


def api_token() -> str:
    """OPENROUTER_API_KEY from the environment, or from a `.env` found from the cwd up."""
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass
    token = os.environ.get("OPENROUTER_API_KEY")
    if not token:
        raise RuntimeError("OPENROUTER_API_KEY is required to enrich a corpus")
    return token


async def call(client, key: str, prompt: str, model: str,
               retries: int) -> tuple[str, dict | None, dict]:
    """-> (schema key, parsed JSON answer or None, token usage)."""
    usage = {"in": 0, "out": 0}
    for attempt in range(retries + 1):
        try:
            r = await client.post(ENDPOINT, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                # the provider counts its own reasoning against this, and a many-option
                # schema's answer is long: at 4,000 most large schemas came back truncated
                "max_tokens": 16000,
            })
            if r.status_code in RETRY_STATUS:
                await asyncio.sleep(2 ** attempt + random.random())
                continue
            r.raise_for_status()
            body = r.json()
            u = body.get("usage") or {}
            usage = {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
            parsed = parse_json(body["choices"][0]["message"]["content"] or "")
            if parsed is not None:
                return key, parsed, usage
        except Exception:
            pass
        if attempt < retries:
            await asyncio.sleep(2 ** attempt + random.random())
    return key, None, usage


async def enrich_schemas(missing: list[tuple[str, dict]], cache: dict, *,
                         model: str = DEFAULT_MODEL, concurrency: int = 64,
                         retries: int = 2, token: str | None = None,
                         save: Callable[[dict], None] | None = None,
                         log: Callable[[str], None] = print) -> dict:
    """Ask for every (key, representative) in `missing`; accepted answers go into
    `cache` under their key. `save(cache)` is called every 25 answers and at the end, so
    an interrupted run keeps what it paid for. -> counts and token usage."""
    import httpx

    token = token or api_token()
    limits = httpx.Limits(max_connections=concurrency + 16,
                          max_keepalive_connections=concurrency + 16)
    sem = asyncio.Semaphore(concurrency)
    done = failed = tin = tout = 0
    started = time.perf_counter()
    reps_by_key = dict(missing)

    async with httpx.AsyncClient(
            timeout=httpx.Timeout(240.0), limits=limits,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"}) as client:
        async def one(k, rep):
            async with sem:
                prompt = build_prompt(rep["task"], rep["options"], rep["question"],
                                      rep["state"], rep["describe"])
                return await call(client, k, prompt, model, retries)

        pending = [asyncio.create_task(one(k, r)) for k, r in missing]
        for future in asyncio.as_completed(pending):
            k, parsed, usage = await future
            tin += usage["in"]
            tout += usage["out"]
            rep = reps_by_key[k]
            kept = clean(parsed, rep["options"], rep["describe"]) if parsed else None
            if kept:
                cache[k] = kept
                done += 1
            else:
                failed += 1
            if (done + failed) % 100 == 0 or done + failed == len(missing):
                el = time.perf_counter() - started
                log(f"  {done + failed:,}/{len(missing):,} ok={done:,} fail={failed:,} "
                    f"tok={tin + tout:,} {(tin + tout) / max(el, 1e-3):,.0f} tok/s {el:,.0f}s")
            if save is not None and (done + failed) % 25 == 0:
                save(cache)
    if save is not None:
        save(cache)
    return {"ok": done, "failed": failed, "tokens_in": tin, "tokens_out": tout}
