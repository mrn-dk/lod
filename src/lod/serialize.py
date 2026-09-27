"""State serializers that more than one caller needs.

`flatten_paths` writes a nested JSON-like value as one line per leaf, each line carrying
its full path from the root:

    orders[3].weight_kg: 14
    orders[3].flags[0]: fragile
    policy.max_weight_kg: 20

It is the shape a log line, a config dump or a flattened API payload arrives in, and it
is the one state shape where no line depends on an earlier line for its meaning: the
record a value belongs to is in the line itself, not in an indentation level or a header
three rows up. Row 47 (`sources/synth/compose.py`) renders states this way; the API can
call the same function on a caller's JSON.

The encoding is lossless for JSON values, and `unflatten_paths` inverts it:

* a key that is not a plain identifier is written `["like this"]` (JSON-quoted);
* a string is written bare unless reading it back would give something else -- empty,
  padded, multi-line, or spelling a number, `true`/`false`/`null`, or a JSON container --
  in which case it is JSON-quoted;
* numbers, booleans and null are written as JSON; empty containers as `[]` and `{}`.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PLAIN_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")
_TOKEN = re.compile(r'\.([A-Za-z_][A-Za-z0-9_\-]*)|\[(\d+)\]|\[("(?:[^"\\]|\\.)*")\]')
_HEAD = re.compile(r"^([A-Za-z_][A-Za-z0-9_\-]*)")


def _join(path: str, key: Any) -> str:
    k = str(key)
    if _PLAIN_KEY.match(k):
        return f"{path}.{k}" if path else k
    return f"{path}[{json.dumps(k, ensure_ascii=False)}]"


def _needs_quotes(s: str) -> bool:
    if s == "" or s != s.strip() or "\n" in s or "\r" in s:
        return True
    if s[0] in "\"[{":
        return True
    try:
        json.loads(s)
    except ValueError:
        return False
    return True                 # a bare 12, true, null or 1e3 would read back as not-a-string


def _scalar(v: Any) -> str:
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False) if _needs_quotes(v) else v
    if v is None or isinstance(v, (bool, int, float)):
        return json.dumps(v)
    return json.dumps(str(v), ensure_ascii=False)


def flatten_paths(obj: Any) -> str:
    """One `path: value` line per leaf, full path on every line. See the module doc."""
    lines: list[str] = []

    def walk(x: Any, path: str) -> None:
        if isinstance(x, dict):
            if not x:
                lines.append(f"{path}: {{}}" if path else "{}")
            for k, v in x.items():
                walk(v, _join(path, k))
        elif isinstance(x, (list, tuple)):
            if not x:
                lines.append(f"{path}: []" if path else "[]")
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        else:
            # a bare root scalar is always JSON: `a: b` alone would read back as a dict
            lines.append(f"{path}: {_scalar(x)}" if path
                         else json.dumps(x, ensure_ascii=False))

    walk(obj, "")
    return "\n".join(lines)


def _parse_value(s: str) -> Any:
    if s == "[]":
        return []
    if s == "{}":
        return {}
    try:
        v = json.loads(s)
    except ValueError:
        return s
    return v


def _parse_path(line: str) -> tuple[list, str]:
    """-> (path tokens, the text after ': '). Tokens are str keys and int indices."""
    toks: list = []
    i = 0
    m = _HEAD.match(line)
    if m:
        toks.append(m.group(1))
        i = m.end()
    while i < len(line) and line[i] != ":":
        m = _TOKEN.match(line, i)
        if not m:
            raise ValueError(f"cannot parse path at column {i}: {line!r}")
        if m.group(1) is not None:
            toks.append(m.group(1))
        elif m.group(2) is not None:
            toks.append(int(m.group(2)))
        else:
            toks.append(json.loads(m.group(3)))
        i = m.end()
    if not line.startswith(": ", i):
        raise ValueError(f"no ': ' after the path: {line!r}")
    return toks, line[i + 2:]


def unflatten_paths(text: str) -> Any:
    """The inverse of `flatten_paths` for any JSON value it produced."""
    lines = [ln for ln in text.split("\n") if ln != ""]
    if len(lines) == 1:
        try:
            _parse_path(lines[0])
        except ValueError:
            return _parse_value(lines[0])        # a bare root scalar or empty container
    root: Any = None

    def slot(container, key, nxt):
        """The child at `key`, created as a list or dict depending on the next token."""
        fresh = [] if isinstance(nxt, int) else {}
        if isinstance(container, list):
            while len(container) <= key:
                container.append(None)
            if container[key] is None:
                container[key] = fresh
            return container[key]
        return container.setdefault(key, fresh)

    for ln in lines:
        toks, raw = _parse_path(ln)
        val = _parse_value(raw)
        if root is None:
            root = [] if isinstance(toks[0], int) else {}
        cur = root
        for a, b in zip(toks, toks[1:]):
            cur = slot(cur, a, b)
        last = toks[-1]
        if isinstance(cur, list):
            while len(cur) <= last:
                cur.append(None)
            cur[last] = val
        else:
            cur[last] = val
    return root


__all__ = ["flatten_paths", "unflatten_paths"]
