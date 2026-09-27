"""LLM enrichment: a real question and option criteria for every schema of a corpus.

`schemas` decides what may be rewritten and checks and applies the answers, `prompts`
writes the request, `client` makes the calls. Nothing here reads or writes the corpus.
"""

from lod.corpus.services.enrich.client import DEFAULT_MODEL, api_token, enrich_schemas
from lod.corpus.services.enrich.prompts import PromptSpec, build_prompt, prompt_for_task
from lod.corpus.services.enrich.schemas import (
    PROTECTED_PREFIXES,
    apply,
    clean,
    collect_schemas,
    protected,
    schema_key,
)

__all__ = ["DEFAULT_MODEL", "PROTECTED_PREFIXES", "PromptSpec", "api_token", "apply",
           "build_prompt", "clean", "collect_schemas", "enrich_schemas", "prompt_for_task",
           "protected", "schema_key"]
