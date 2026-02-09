from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from openagent.core.types import Message, ToolDef


class MessageConverterMixin(ABC):
    @abstractmethod
    def convert_messages(
        self, messages: list[Message], system_prompt: str = ""
    ) -> dict[str, Any]:
        """Convert canonical messages to provider-specific format.

        Returns a dict that the provider's chat() method can unpack into API kwargs.
        E.g. {"messages": [...]} or {"messages": [...], "system": "..."}.
        """

    @abstractmethod
    def convert_response(self, response: Any) -> Message:
        """Convert provider API response to a canonical Message."""

    @abstractmethod
    def convert_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        """Convert canonical ToolDefs to provider-specific tool format."""
