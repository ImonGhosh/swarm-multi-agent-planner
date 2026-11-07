"""A fake OpenAI-compatible client for tests: no network, deterministic answers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List

CANDIDATE = "Here is the proposed time: Monday, 9:30 - 10:00"


def default_responder(messages: List[Dict[str, str]]) -> str:
    if "Produce one final answer" in messages[0]["content"]:
        return CANDIDATE
    return "intermediate"


class FakeClient:
    """Mimics ``AsyncOpenAI``: ``await client.chat.completions.create(...)``."""

    def __init__(
        self,
        responder: Callable[[List[Dict[str, str]]], str] = default_responder,
        *,
        provider: str = "FakeProvider",
        cost: float = 0.0001,
        delay: float = 0.0,
    ) -> None:
        self.responder = responder
        self.provider = provider
        self.cost = cost
        self.delay = delay
        self.calls: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        import asyncio

        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        text = self.responder(kwargs["messages"])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, cost=self.cost),
            provider=self.provider,
        )
