from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from openagent.core.types import Message, ToolDef


class BaseProvider(ABC):
    def __init__(self, model: str, api_key: str | None = None, **kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> Message:
        """Send messages to the LLM and return the assistant response."""

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text chunks from the LLM response.

        Default implementation falls back to non-streaming chat.
        Subclasses can override for true streaming support.
        """
        response = await self.chat(messages, tools, system_prompt, **kwargs)
        yield response.text
