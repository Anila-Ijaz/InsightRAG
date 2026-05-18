"""LLM provider abstraction.

Single interface across OpenAI and Anthropic. Configurable via env. Streaming returns
async iterators of text deltas, which the API layer forwards as SSE.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from functools import lru_cache

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from insightrag.config import get_settings


class LLMClient(ABC):
    """Provider-agnostic chat client."""

    @abstractmethod
    async def complete(self, system: str, user: str, **kwargs) -> str: ...

    @abstractmethod
    async def stream(self, system: str, user: str, **kwargs) -> AsyncIterator[str]: ...


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def complete(self, system: str, user: str, **kwargs) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content or ""

    async def stream(self, system: str, user: str, **kwargs) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def complete(self, system: str, user: str, **kwargs) -> str:
        resp = await self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        # Anthropic returns a list of content blocks
        return "".join(block.text for block in resp.content if block.type == "text")

    async def stream(self, system: str, user: str, **kwargs) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        ) as stream:
            async for text in stream.text_stream:
                yield text


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    settings = get_settings()
    logger.info(f"Initializing LLM client: provider={settings.llm_provider} model={settings.llm_model}")

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when llm_provider=openai")
        return OpenAIClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when llm_provider=anthropic")
        return AnthropicClient(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    raise ValueError(f"Unknown llm_provider: {settings.llm_provider}")
