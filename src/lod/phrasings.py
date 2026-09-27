"""The phrasing bank: many wordings per question template, sampled at generation time.

Wording variety used to come from calling an LLM per record. That is O(records) in cost
and it bought very little, because what came back were near-paraphrases -- four gefs
wordings differing only in verb choice. A model can ignore a question that is always the
same question and key off the state's shape instead.

So phrasings are generated ONCE per template (offline, from `lod/assets/phrasing_specs/`),
checked, frozen into a JSON bank, and sampled here. Cost is O(templates). The sampling is
seeded by the record's own identity, so a corpus rebuild picks the same phrasing for the
same record and two records of the same task get different ones.

**A phrasing may never change what is being asked.** In an earlier corpus a gefs question
was reworded to "does at least one ensemble member forecast >= 1mm" while its target
stayed the *fraction* of members. Those two agree 23.1% of the time, so on 945 of 1,969
`probability_recovery` questions the model was marked against a question it had not been
asked, and that gate went to r = -0.046. The bank is built with a checker for exactly this
(quantity words that appear in a phrasing but not in its template), and the sampler
below refuses a phrasing whose placeholders do not match the template's.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from lod.paths import ASSETS

PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

DEFAULT_BANK = ASSETS / "phrasings.json"


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


class PhrasingBank:
    """Frozen wordings per template id. Missing id -> the template itself, unchanged."""

    def __init__(self, banks: dict[str, list[str]] | None = None,
                 templates: dict[str, str] | None = None):
        self._banks = banks or {}
        # the template each bank was written for; a generator whose template has since
        # changed must not draw phrasings of the OLD question
        self._templates = templates or {}

    @classmethod
    def load(cls, path: Path | str | None = None) -> "PhrasingBank":
        # LOD_PHRASING_BANK=none: every template verbatim. For tests that parse the
        # generated wording, and for an A/B build without the bank.
        env = os.environ.get("LOD_PHRASING_BANK")
        if path is None and env:
            if env.lower() == "none":
                return cls({})
            path = env
        p = Path(path) if path is not None else DEFAULT_BANK
        if not p.exists():
            return cls({})
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls({k: list(v["phrasings"]) if isinstance(v, dict) else list(v)
                    for k, v in raw.items()},
                   {k: v["template"] for k, v in raw.items()
                    if isinstance(v, dict) and "template" in v})

    def __contains__(self, template_id: str) -> bool:
        return template_id in self._banks

    def count(self, template_id: str) -> int:
        return len(self._banks.get(template_id, ()))

    def pick(self, template_id: str, template: str, key: str) -> str:
        """One phrasing for this record, chosen deterministically from `key`.

        Falls back to the template when the bank has nothing, and skips any stored
        phrasing whose placeholders differ from the template's -- a phrasing that has
        lost or gained a slot cannot be formatted and would otherwise raise at build
        time, or worse, silently format with a stale value.
        """
        stored = self._templates.get(template_id)
        if stored is not None and stored != template:
            return template                  # stale bank: reworded since it was built
        want = placeholders(template)
        options = [p for p in self._banks.get(template_id, ()) if placeholders(p) == want]
        if not options:
            return template
        h = hashlib.sha1(f"{template_id}\x00{key}".encode("utf-8")).digest()
        return options[int.from_bytes(h[:8], "big") % len(options)]
