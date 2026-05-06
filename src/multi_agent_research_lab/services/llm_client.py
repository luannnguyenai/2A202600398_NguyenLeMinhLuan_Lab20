"""LLM client abstraction with OpenAI + Mock support.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("llm")

# Pricing for gpt-4o-mini (dollars per 1M tokens)
GPT4O_MINI_PRICING = {
    "input": 0.15,
    "output": 0.60,
}


@dataclass(frozen=True)
class LLMResponse:
    """Response from LLM with token and cost tracking."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLMClient:
    """OpenAI-based LLM client with retry and token tracking."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model

        # Import OpenAI SDK with fallback
        try:
            from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
            self.openai = OpenAI(api_key=self.api_key)
            self.api_errors = (RateLimitError, APITimeoutError, APIConnectionError)
        except ImportError:
            self.openai = None
            self.api_errors = ()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> LLMResponse:
        """Get completion from LLM with retry logic.

        Args:
            system_prompt: System/role prompt
            user_prompt: User message
            response_format: "json" for JSON mode, None for text
            timeout_seconds: Timeout per request

        Returns:
            LLMResponse with content and token counts
        """
        if not self.openai:
            raise RuntimeError(
                "OpenAI not installed. Install with: pip install openai"
            )

        started = perf_counter()

        # Build request
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "timeout": timeout_seconds,
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        # Call API
        response = self.openai.chat.completions.create(**kwargs)

        # Extract response
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Calculate cost
        cost_usd = (
            input_tokens * GPT4O_MINI_PRICING["input"]
            + output_tokens * GPT4O_MINI_PRICING["output"]
        ) / 1_000_000

        latency_ms = (perf_counter() - started) * 1000

        logger.info(
            json.dumps(
                {
                    "event": "llm_call",
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                }
            )
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )


class MockLLMClient(LLMClient):
    """Deterministic mock LLM client for testing without API calls."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> LLMResponse:
        """Return deterministic mock response based on prompt content.

        Heuristics:
        - If prompt contains "sub-queries" or "queries" → JSON with queries array
        - If prompt contains "key_claims" → JSON with claims array
        - If prompt contains "final answer" or "500 words" → markdown ~500 words
        - Default: echo first 80 chars of user prompt
        """

        # Determine response type
        combined = (system_prompt + user_prompt).lower()
        content = ""

        if "sub-queries" in combined or ("queries" in combined and "{" in combined):
            # Query expansion response
            base_query = user_prompt.split('"')[1] if '"' in user_prompt else "test"
            queries = [
                base_query,
                f"{base_query} overview",
                f"{base_query} best practices",
            ]
            content = json.dumps({"queries": queries})

        elif "key_claims" in combined or "claims" in combined:
            # Analyst response with JSON
            content = json.dumps(
                {
                    "key_claims": [
                        {
                            "claim": "Mock claim 1 about the topic",
                            "evidence_index": [1],
                            "confidence": "med",
                        },
                        {
                            "claim": "Mock claim 2 about the topic",
                            "evidence_index": [2],
                            "confidence": "high",
                        },
                    ],
                    "conflicts": [],
                    "gaps": ["limited sources", "temporal coverage"],
                }
            )

        elif "final answer" in combined or "500 words" in combined:
            # Writer response - markdown with citations
            content = (
                "# Research Summary\n\n"
                "## Overview\n"
                "This research explores the topic in depth. "
                "Based on available sources, several key findings emerge [#1].\n\n"
                "## Key Findings\n"
                "The topic has significant implications [#2]. "
                "Recent developments show promising directions [#1].\n\n"
                "## Discussion\n"
                "Multiple perspectives exist on this subject. "
                "Evidence suggests strong correlation [#3].\n\n"
                "### Sources\n"
                "[#1] Mock source 1 — https://example.com/1\n"
                "[#2] Mock source 2 — https://example.com/2\n"
                "[#3] Mock source 3 — https://example.com/3\n"
            )

        else:
            content = f"Mock response for: {user_prompt[:80]}"

        # Estimate tokens
        prompt_len = len(system_prompt) + len(user_prompt)
        input_tokens = math.ceil(prompt_len / 4)
        output_tokens = math.ceil(len(content) / 4)
        cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )


def get_llm_client(api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> LLMClient:
    """Factory to get real or mock LLM client based on API key availability."""
    if api_key or os.getenv("OPENAI_API_KEY"):
        return LLMClient(api_key=api_key, model=model)
    return MockLLMClient(model=model)
