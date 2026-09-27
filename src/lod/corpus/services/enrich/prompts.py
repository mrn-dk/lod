"""The enrichment prompt: dataset-specific guidance plus one schema to rewrite."""

from __future__ import annotations

import json
from dataclasses import dataclass

from lod.corpus.services.enrich.schemas import MAX_DESCRIBED_OPTIONS


@dataclass(frozen=True)
class PromptSpec:
    name: str
    purpose: str
    constraints: str


PROMPTS = {
    "google_civil_comments": PromptSpec(
        name="Civil Comments",
        purpose="Measure whether a public comment has each published moderation attribute.",
        constraints="Use the dataset field meaning: toxicity, severe toxicity, obscene, threat, insult, identity attack, or sexual explicitness. Ask yes/no questions. A yes description must state the attribute is present; a no description must state it is absent.",
    ),
    "goemotions": PromptSpec(
        name="GoEmotions",
        purpose="Measure whether the comment expresses one named emotion.",
        constraints="Ask whether the comment expresses the named emotion. Do not treat the rater fraction as a hard fact; it remains the target distribution.",
    ),
    "hatespeech": PromptSpec(
        name="Measuring Hate Speech",
        purpose="Represent the published annotator survey dimensions.",
        constraints="Preserve the survey dimension and make ordinal level meanings explicit when the source adapter supplies levels.",
    ),
    "default": PromptSpec(
        name="Dataset-specific decision task",
        purpose="Make the supplied decision criterion precise and readable.",
        constraints="Use only semantics supported by the dataset name, field name, state, and supplied option keys. Never invent labels or evidence.",
    ),
    "momererkoc_social_bias_frames": PromptSpec(
        name="Social Bias Frames",
        purpose="Represent the published yes/no annotation fields for offensive intent and sexual content.",
        constraints="Use only the documented field named in the task. Do not discuss annotator demographics, worker IDs, or inferred stereotypes. Ask a precise yes/no question.",
    ),
    "nvd_": PromptSpec(
        name="NVD CVE",
        purpose="Describe published CVSS or CWE decisions for a vulnerability description.",
        constraints="Use only the NVD-defined metric or weakness class. Preserve CVSS terminology and do not infer a severity from prose beyond the supplied source label.",
    ),
    "loghub_": PromptSpec(
        name="Loghub",
        purpose="Describe the published severity or emitting-component decision for a log line.",
        constraints="For severity, preserve the ordered level meanings. For component, describe the emitting subsystem without inventing operational consequences.",
    ),
    "kalshi_": PromptSpec(
        name="Kalshi settled markets",
        purpose="Describe the yes/no settlement decision for a market family.",
        constraints="Use the market title and published rules. Do not describe the post-settlement price as an independent forecast or alter the settlement target.",
    ),
}


def prompt_for_task(task: str) -> PromptSpec:
    for prefix, spec in PROMPTS.items():
        if prefix != "default" and task.startswith(prefix):
            return spec
    return PROMPTS["default"]


def build_prompt(task: str, options: list[str], question: str, state: str,
                 describe: bool) -> str:
    """The prompt for one schema: its task, current question, options and one state."""
    spec = prompt_for_task(task)
    shown = options[:MAX_DESCRIBED_OPTIONS]
    body = {
        "task_name": task,
        "current_question": question,
        "options": shown,
        "example_state": state[:1200],
    }
    want = ('"question": "<the rewritten question>", '
            '"descriptions": ["<one per option, same order>", ...]') if describe else \
           '"question": "<the rewritten question>"'
    extra = "" if describe else (
        "\nThis task's option set changes from example to example, so do NOT return "
        "descriptions. Return the question only.")
    return f"""You are preparing one decision schema for a calibrated classifier that answers
by picking from a supplied option list. It never generates text.

Dataset: {spec.name}
Purpose: {spec.purpose}
Dataset-specific constraints: {spec.constraints}

Here is one schema from the corpus. `current_question` was auto-generated from a column
name and is usually poor.

{json.dumps(body, ensure_ascii=False, indent=2)}

Rewrite it:

1. `question` — a single self-contained question a careful annotator could answer from
   the state alone. Never name a column, a dataset or a field. Never mention the options.
2. `descriptions` — for each option **in the given order**, one clause saying what that
   option means, so that the meaning lives in the description rather than in the key.
   Write what the label means in this dataset, not what the example happens to be.
   Never restate the option key as its own definition. 4-25 words each.{extra}

Write in the language the states are written in; English unless they are clearly not.
If an option key is a bare number with no meaning you can justify from the task name or
the state, say so plainly rather than inventing a scale for it.

Do not invent labels, do not reorder, do not add or drop options.
Return ONLY a JSON object: {{{want}}}"""
