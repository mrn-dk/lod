"""Plain HTTP: GET a URL politely, or save it to a file atomically.

Every direct download in the fetch stage goes through here, so they all send the same
User-Agent (several hosts answer 403 without one), all back off on 429/503 the way
OAI-PMH and GitHub ask, and none can leave a half-written file that a later run would
mistake for a complete one.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "lod-corpus/1.0 (research dataset build; non-commercial)"}


def get(url: str, timeout: int = 180, retries: int = 5, pause: float = 5.0,
        headers: dict[str, str] | None = None) -> bytes:
    """GET `url`. Honours Retry-After on 429/503; retries transient network errors."""
    req = urllib.request.Request(url, headers=headers or UA)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                wait = int(e.headers.get("Retry-After") or 15)
                print(f"    {e.code} from {url[:60]}, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == retries - 1:
                raise
            print(f"    {type(e).__name__}, retry {attempt + 1}", flush=True)
            time.sleep(pause)
    raise RuntimeError(f"GET failed: {url}")


def save(body: bytes, dest: Path) -> Path:
    """Write `body` to `dest` through a temporary file, so the write is all or nothing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return dest


def download(url: str, dest: Path, refresh: bool = False, min_bytes: int = 1, **kw) -> Path:
    """`url` -> `dest`, skipped when `dest` already holds at least `min_bytes`."""
    if not refresh and dest.is_file() and dest.stat().st_size >= min_bytes:
        return dest
    body = get(url, **kw)
    if len(body) < min_bytes:
        raise OSError(f"{url}: {len(body)} bytes, expected at least {min_bytes}")
    return save(body, dest)
