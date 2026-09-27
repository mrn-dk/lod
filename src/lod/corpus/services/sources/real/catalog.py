"""Resolved locations for the source table, and what broke on the way.

`datasets` 5.x removed script-based loading ("Dataset scripts are no longer supported"),
which takes out a number of the canonical repos the spec names by their well-known paths:
`dynabench/dynasent`, `allenai/social_bias_frames`, `wenhu/tab_fact`,
`ibm-research/tab_fact`, `qanastek/MASSIVE`, `google-research-datasets/newsgroup`. Every
one of those is still reachable, but under a different repo, so the path the spec implies
and the path that works are not the same string.

This module is that mapping, kept separate from the loaders so that when a mirror rots --
and community mirrors do -- the fix is one line here and the failure is legible. Each
entry records where it came from and, where it matters, what shape the rows arrive in.

Verified 2026-09-18 with `datasets` 5.0.1 / `huggingface_hub` 1.32.0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HFSource:
    row: int
    path: str
    config: str | None = None
    licence: str = ""
    note: str = ""


# Resolved and load-checked. `config=None` means the default config.
HF_SOURCES: tuple[HFSource, ...] = (
    HFSource(2, "tasksource/bigbench", None, "Apache-2.0",
             "167 configs"),
    HFSource(2, "tasksource/mmlu", None, "MIT", "57 configs"),
    HFSource(3, "Muennighoff/natural-instructions", None, "Apache-2.0",
             "Super-NaturalInstructions; filter to tasks with <= 30 distinct outputs"),
    HFSource(16, "clinc/clinc_oos", "plus", "CC-BY-3.0", "151 intents incl. out-of-scope"),
    HFSource(16, "mteb/amazon_massive_intent", None, "CC-BY-4.0",
             "MASSIVE via mteb; qanastek/MASSIVE is script-based and no longer loads"),
    HFSource(16, "benayas/snips", None, "CC-BY-4.0", ""),
    HFSource(16, "Bhuvaneshwari/intent_classification", None, "unknown", "HWU64-like"),
    HFSource(17, "coastalcph/lex_glue", "ledgar", "CC-BY-4.0", "100 contract clause types"),
    HFSource(17, "coastalcph/lex_glue", "scotus", "CC-BY-4.0", "13 issue areas"),
    HFSource(17, "coastalcph/lex_glue", "eurlex", "CC-BY-4.0", "EuroVoc level 1, multilabel"),
    HFSource(17, "coastalcph/lex_glue", "case_hold", "CC-BY-4.0",
             "5 per-example holdings -- the per-example-option source A2 needs"),
    HFSource(17, "coastalcph/lex_glue", "unfair_tos", "CC-BY-4.0", "multilabel"),
    HFSource(18, "SetFit/20_newsgroups", None, "public domain", ""),
    HFSource(18, "yangwang825/reuters-21578", None, "public domain", ""),
    HFSource(19, "google/civil_comments", None, "CC0-1.0",
             "no ClassLabel columns at all; 7 float rater-fraction columns -> soft"),
    HFSource(20, "google-research-datasets/go_emotions", "raw", "Apache-2.0",
             "one row PER RATER; aggregate over `id` across raters to get the soft target"),
    HFSource(21, "ucberkeley-dlab/measuring-hate-speech", None, "CC-BY-4.0",
             "one row PER ANNOTATOR; aggregate over `comment_id`"),
    HFSource(22, "HelloWorld2307/dynasent", None, "Apache-2.0",
             "dynabench/dynasent is script-based and no longer loads"),
    HFSource(23, "momererkoc/social_bias_frames", None, "CC-BY-4.0",
             "allenai/social_bias_frames is script-based and no longer loads"),
    HFSource(24, "nyu-mll/multi_nli", None, "CC-BY-SA-3.0/OANC", "5 annotator labels"),
    HFSource(24, "stanfordnlp/snli", None, "CC-BY-SA-4.0", "5 annotator labels"),
    HFSource(31, "Raywithyou/TabFact", None, "CC-BY-4.0",
             "wenhu/tab_fact and ibm-research/tab_fact are both script-based"),
    HFSource(32, "google-research-datasets/mbpp", "full", "CC-BY-4.0",
             "row 32 needs the tests executed, not just loaded"),
)

# Rows with no working Hub mirror, which have to be fetched directly. `lod.corpus.services.sources.base
# .fetch` already does cached HTTP with a User-Agent, which several of these hosts require.
DIRECT_ONLY = {
    25: "ChaosNLI — no Hub mirror found; release archive from the ChaosNLI GitHub",
}

# Rows that were never on the Hub: bulk downloads, APIs and archives.
NON_HF_ROWS = {
    4: "Public Jira Dataset (Zenodo)",
    5: "GH Archive 2024 (hourly .json.gz)",
    6: "arXiv metadata snapshot",
    7: "PubMed baseline",
    8: "OpenAlex works",
    9: "NVD CVE feeds",
    10: "Congress.gov bills; Federal Register API",
    11: "ClinicalTrials.gov API; openFDA SPL",
    12: "SEC EDGAR 10-K Item 1",
    13: "Lichess puzzle DB (.csv.zst)",
    14: "Loghub, 16 systems",
    15: "SPDX license-list-data (GitHub)",
    26: "FiveThirtyEight forecast CSVs",
    27: "NWS gridpoint forecast + observations API",
    28: "Manifold Markets API",
    29: "GH Archive (PRs, workflow runs)",
    30: "OpenAlex, MusicBrainz, OpenLibrary, USDA FoodData Central, PokeAPI",
}


def rows_covered() -> set[int]:
    return {s.row for s in HF_SOURCES} | set(DIRECT_ONLY) | set(NON_HF_ROWS) | {1}
