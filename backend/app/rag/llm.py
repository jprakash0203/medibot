"""LLM client wrapper.

Uses the OpenAI SDK to talk to either OpenAI directly or any
OpenAI-compatible endpoint (Groq, together.ai, Anyscale, etc.).
Which one is used depends on `.env` — the base_url switches providers.
"""
from openai import OpenAI

from app.config import get_settings


class LLMClient:
    """Chat completion wrapper. One instance per app lifetime."""

    def __init__(self):
        settings = get_settings()
        # Passing base_url=None makes it default to OpenAI's public endpoint.
        # For Groq, base_url is https://api.groq.com/openai/v1 (from .env).
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 800,
    ) -> str:
        """Run a single chat completion. Low temperature by default —
        for RAG we want deterministic, grounded answers, not creativity."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()