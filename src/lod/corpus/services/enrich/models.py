"""Strict models for the LLM preprocessing boundary.

The model may improve wording and descriptions, but it may not change labels or
targets. The caller verifies those invariants before writing output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


QuestionType = Literal["noul", "choice", "score"]


class EnrichedQuestion(BaseModel):
    id: str
    type: QuestionType
    instructions: str = Field(min_length=1, max_length=500)
    descriptions: list[str] = Field(min_length=2, max_length=255)

    @field_validator("descriptions")
    @classmethod
    def descriptions_are_nonempty(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("option descriptions must be non-empty")
        return value


class EnrichmentResult(BaseModel):
    questions: list[EnrichedQuestion] = Field(min_length=1)


class InputQuestion(BaseModel):
    """Stable question context supplied to the LLM."""

    id: str
    type: QuestionType
    options: list[str] = Field(min_length=2, max_length=255)
    target: int | list[float] | None = None


class InputExample(BaseModel):
    task: str
    state: str
    questions: list[InputQuestion] = Field(min_length=1)
