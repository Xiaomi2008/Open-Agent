from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .types import (
    ContentBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    text_message,
    tool_result_message,
)


@dataclass
class Session:
    system_prompt: str = ""
    _messages: list[Message] = field(default_factory=list)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def add(self, role: Literal["user", "assistant", "system"], content: str) -> Message:
        msg = text_message(role, content)
        self._messages.append(msg)
        return msg

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def add_tool_results(self, results: list[ToolResultBlock]) -> Message:
        msg = tool_result_message(results)
        self._messages.append(msg)
        return msg

    def clear(self) -> None:
        self._messages.clear()

    def replace_messages(self, messages: list[Message]) -> None:
        """Replace the entire message list with a new list.

        Used by compaction to swap out old messages for a summarized version.
        """
        self._messages = list(messages)

    def __len__(self) -> int:
        return len(self._messages)

    def to_list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in self._messages:
            if isinstance(msg.content, str):
                out.append({"role": msg.role, "content": msg.content})
            else:
                blocks = []
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        blocks.append({"type": "text", "text": b.text})
                    elif isinstance(b, ToolUseBlock):
                        blocks.append({
                            "type": "tool_use",
                            "id": b.id,
                            "name": b.name,
                            "arguments": b.arguments,
                        })
                    elif isinstance(b, ToolResultBlock):
                        blocks.append({
                            "type": "tool_result",
                            "tool_use_id": b.tool_use_id,
                            "content": b.content,
                            "is_error": b.is_error,
                        })
                out.append({"role": msg.role, "content": blocks})
        return out

    def save(self, path: str | Path) -> None:
        """Save session to a JSON file.

        Args:
            path: Path to save the session to
        """
        data = {
            "system_prompt": self.system_prompt,
            "messages": self.to_list(),
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        """Load session from a JSON file.

        Args:
            path: Path to load the session from

        Returns:
            Loaded Session instance
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        session = cls(system_prompt=data.get("system_prompt", ""))

        for msg_data in data.get("messages", []):
            role = msg_data["role"]
            content = msg_data["content"]

            if isinstance(content, str):
                session._messages.append(Message(role=role, content=content))
            else:
                blocks: list[ContentBlock] = []
                for block_data in content:
                    block_type = block_data.get("type")
                    if block_type == "text":
                        blocks.append(TextBlock(text=block_data["text"]))
                    elif block_type == "tool_use":
                        blocks.append(ToolUseBlock(
                            id=block_data["id"],
                            name=block_data["name"],
                            arguments=block_data["arguments"],
                        ))
                    elif block_type == "tool_result":
                        blocks.append(ToolResultBlock(
                            tool_use_id=block_data["tool_use_id"],
                            content=block_data["content"],
                            is_error=block_data.get("is_error", False),
                        ))
                session._messages.append(Message(role=role, content=blocks))

        return session
