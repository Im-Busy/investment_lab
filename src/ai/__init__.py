"""Instructor-powered structured LLM output for trading workflow.

Provides Pydantic-validated responses from LLMs, eliminating JSON parsing bugs.
Activated when the project first adds LLM-powered features.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TradingSignalAnalysis(BaseModel):
    """Structured analysis of a trading signal."""

    direction: str = Field(description="Trade direction: long, short, or neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0 to 1")
    rationale: str = Field(description="Brief explanation of the analysis")
    risk_flags: list[str] = Field(default_factory=list, description="Risk concerns identified")
    suggested_hold: int = Field(
        default=5, ge=1, le=252, description="Suggested hold period in bars"
    )


class PatternReviewOutput(BaseModel):
    """Structured review of a chart pattern detection result."""

    pattern_name: str
    is_valid: bool = Field(description="Whether the pattern is confirmed")
    quality_score: float = Field(ge=0.0, le=1.0, description="Pattern quality assessment")
    key_levels: list[float] = Field(description="Key support/resistance levels")
    notes: str = Field(default="", description="Reviewer notes")


class BacktestSummary(BaseModel):
    """Structured backtest performance summary."""

    return_pct: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: int
    verdict: str = Field(description="One paragraph verdict on the strategy")


class StructuredLLM:
    """Wrapper for Instructor that enforces Pydantic response models.

    Usage:
        from openai import OpenAI
        from src.ai.structured_llm import StructuredLLM, PatternReviewOutput

        client = OpenAI()
        llm = StructuredLLM(client)
        result: PatternReviewOutput = llm.ask("Analyze this double bottom...", PatternReviewOutput)
    """

    def __init__(self, client: Any) -> None:
        """Initialize with any OpenAI-compatible client."""
        try:
            import instructor

            self._client = instructor.patch(client)
        except ImportError:
            raise ImportError(
                "instructor not installed. Run: uv add instructor --group dev"
            ) from None

    def ask(self, prompt: str, response_model: type[BaseModel], **kwargs: Any) -> BaseModel:
        """Send a prompt and receive a validated Pydantic model.

        Args:
            prompt: The user message / instruction.
            response_model: A Pydantic model class defining the expected response shape.
            **kwargs: Additional arguments forwarded to the client (model, temperature, etc.).

        Returns:
            An instance of `response_model` populated with validated LLM output.
        """
        response = self._client.chat.completions.create(
            model=kwargs.pop("model", "gpt-4o"),
            response_model=response_model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response


def get_structured_llm(
    api_key: str | None = None,
    base_url: str | None = None,
) -> StructuredLLM:
    """Factory for a configured StructuredLLM instance.

    Uses OPENAI_API_KEY / OPENAI_BASE_URL env vars if not provided.
    """
    import os

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
    )
    return StructuredLLM(client)
