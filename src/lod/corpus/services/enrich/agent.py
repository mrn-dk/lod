"""Lazy PydanticAI/OpenRouter integration.

Importing this module and running dry-run mode never needs credentials or a
network connection. The token is read only when an agent is requested.
"""

from __future__ import annotations

import os

from lod.corpus.services.enrich.models import EnrichmentResult

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"


def make_agent(model: str = DEFAULT_MODEL):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError as exc:
        raise RuntimeError(
            "install the preprocessing extra first: uv sync --extra preprocess"
        ) from exc
    token = os.environ.get("OPENROUTER_API_KEY")
    if not token:
        raise RuntimeError("OPENROUTER_API_KEY is required for LLM preprocessing")
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:
        raise RuntimeError(
            "install the preprocessing extra first: uv sync --extra preprocess"
        ) from exc
    provider = OpenAIProvider(
        api_key=token,
        base_url="https://openrouter.ai/api/v1",
    )
    return Agent(
        OpenAIChatModel(model, provider=provider),
        output_type=EnrichmentResult,
        system_prompt="Return only the requested structured output. Accuracy and source fidelity matter more than eloquence.",
    )


# --- diversification agents ----------------------------------------------------------

_OUTPUT_TYPE = {
    "span": "SpanReplaced",
    "spancheck": "SpanCheck",
    "spot": "SpotInvented",
    "distractor_judge": "DistractorVerdicts",
    "distractors": "Distractors",
    "phrasings": "Phrasings",
}


def make_agent_for(stage: str, model: str = DEFAULT_MODEL):
    """One agent per diversification stage, typed by its own output model.

    Separate agents rather than one with a union return: the output type is what makes
    the boundary enforceable, and a union would let a stage return another stage's shape.
    """
    from lod.corpus.services.enrich import diversify as _d
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError as exc:
        raise RuntimeError("uv sync --extra preprocess") from exc
    token = os.environ.get("OPENROUTER_API_KEY")
    if not token:
        raise RuntimeError("OPENROUTER_API_KEY is required for LLM preprocessing")
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    name = _OUTPUT_TYPE.get(stage)
    if name is None:
        raise ValueError(f"no output type for stage {stage!r}")
    provider = OpenAIProvider(api_key=token, base_url="https://openrouter.ai/api/v1")
    return Agent(OpenAIChatModel(model, provider=provider),
                 output_type=getattr(_d, name), retries=3)
