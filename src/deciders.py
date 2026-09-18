"""Anthropic decider — jedini LLM klijent u app-u.

A "decider" is a callable that takes the fully rendered prompt (a plain string built
from the *public* investigator context only) and returns the model's raw text
response. Everything else — schema validation, provenance, demo/evaluation fallback
policy — stays in `LLMInvestigator`.
"""

from __future__ import annotations

import os


class ProviderConfigError(RuntimeError):
    """Raised for a missing key or unusable configuration. Surfaced to the API as a
    controlled 400, never a stack trace."""


DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"


class AnthropicDecider:
    provider = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderConfigError(
                "ANTHROPIC_API_KEY is not set. Export it, or use rule_based mode."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigError(
                "The 'anthropic' package is not installed (pip install anthropic)."
            ) from exc
        self._client = Anthropic(api_key=key)

    def __call__(self, prompt: str) -> str:  # pragma: no cover - needs network
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content
                       if getattr(b, "type", "") == "text")
