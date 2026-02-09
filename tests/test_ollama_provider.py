"""Tests for Ollama provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openagent.core.types import (
    Message,
    TextBlock,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
)
from openagent.provider.ollama import OllamaConverterMixin, OllamaProvider


# ---------------------------------------------------------------------------
# Converter mixin tests (pure logic, no network)
# ---------------------------------------------------------------------------


class TestOllamaConverterMixin:
    def setup_method(self):
        self.converter = OllamaConverterMixin()

    def test_convert_simple_messages(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        result = self.converter.convert_messages(messages)
        assert result["messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_convert_messages_with_system_prompt(self):
        messages = [Message(role="user", content="Hello")]
        result = self.converter.convert_messages(messages, system_prompt="Be helpful")
        assert result["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert result["messages"][1] == {"role": "user", "content": "Hello"}

    def test_convert_system_message(self):
        messages = [Message(role="system", content="You are helpful")]
        result = self.converter.convert_messages(messages)
        assert result["messages"] == [{"role": "system", "content": "You are helpful"}]

    def test_convert_assistant_with_tool_calls(self):
        messages = [
            Message(role="assistant", content=[
                TextBlock(text="Let me check"),
                ToolUseBlock(id="call_1", name="get_weather", arguments={"city": "Tokyo"}),
            ]),
        ]
        result = self.converter.convert_messages(messages)
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        # Ollama uses dict arguments, not JSON strings
        assert msg["tool_calls"][0]["function"]["arguments"] == {"city": "Tokyo"}

    def test_convert_assistant_tool_calls_only(self):
        """Assistant message with only tool calls and no text."""
        messages = [
            Message(role="assistant", content=[
                ToolUseBlock(id="call_1", name="search", arguments={"q": "test"}),
            ]),
        ]
        result = self.converter.convert_messages(messages)
        msg = result["messages"][0]
        assert msg["content"] == ""
        assert len(msg["tool_calls"]) == 1

    def test_convert_tool_results(self):
        messages = [
            Message(role="tool_result", content=[
                ToolResultBlock(tool_use_id="call_1", content="22C and sunny"),
            ]),
        ]
        result = self.converter.convert_messages(messages)
        # Ollama tool results have no tool_call_id
        assert result["messages"] == [{"role": "tool", "content": "22C and sunny"}]

    def test_convert_multiple_tool_results(self):
        messages = [
            Message(role="tool_result", content=[
                ToolResultBlock(tool_use_id="call_1", content="Result 1"),
                ToolResultBlock(tool_use_id="call_2", content="Result 2"),
            ]),
        ]
        result = self.converter.convert_messages(messages)
        assert len(result["messages"]) == 2
        assert result["messages"][0] == {"role": "tool", "content": "Result 1"}
        assert result["messages"][1] == {"role": "tool", "content": "Result 2"}

    # -- convert_response --

    def test_convert_response_text_only(self):
        mock_response = MagicMock()
        mock_response.message.content = "Hello!"
        mock_response.message.tool_calls = None
        result = self.converter.convert_response(mock_response)
        assert result.role == "assistant"
        assert result.content == "Hello!"

    def test_convert_response_with_tool_calls(self):
        mock_tc = MagicMock()
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = {"city": "Tokyo"}

        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = [mock_tc]

        result = self.converter.convert_response(mock_response)
        assert result.role == "assistant"
        assert isinstance(result.content, list)
        tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "get_weather"
        assert tool_blocks[0].arguments == {"city": "Tokyo"}

    def test_convert_response_with_text_and_tool_calls(self):
        mock_tc = MagicMock()
        mock_tc.function.name = "search"
        mock_tc.function.arguments = {"q": "test"}

        mock_response = MagicMock()
        mock_response.message.content = "Let me search"
        mock_response.message.tool_calls = [mock_tc]

        result = self.converter.convert_response(mock_response)
        assert result.role == "assistant"
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert isinstance(result.content[0], TextBlock)
        assert isinstance(result.content[1], ToolUseBlock)

    def test_convert_response_arguments_as_string(self):
        """Defensive: handle arguments as JSON string."""
        mock_tc = MagicMock()
        mock_tc.function.name = "calc"
        mock_tc.function.arguments = '{"expr": "1+1"}'

        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = [mock_tc]

        result = self.converter.convert_response(mock_response)
        tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks[0].arguments == {"expr": "1+1"}

    # -- convert_tools --

    def test_convert_tools(self):
        tools = [
            ToolDef(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            ),
        ]
        result = self.converter.convert_tools(tools)
        assert len(result) == 1
        assert result[0] == {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }

    def test_convert_tools_empty(self):
        result = self.converter.convert_tools([])
        assert result == []


# ---------------------------------------------------------------------------
# Provider tests (mocked SDK)
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    @staticmethod
    def _make_provider(mock_client: AsyncMock) -> OllamaProvider:
        """Create an OllamaProvider with a pre-set mock client."""
        provider = OllamaProvider.__new__(OllamaProvider)
        provider.model = "llama3"
        provider.api_key = None
        provider._max_retries = 0
        provider._client = mock_client
        return provider

    @staticmethod
    def _text_response(text: str) -> MagicMock:
        resp = MagicMock()
        resp.message.content = text
        resp.message.tool_calls = None
        return resp

    async def test_chat_simple(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=self._text_response("Hello!"))
        provider = self._make_provider(mock_client)

        result = await provider.chat([Message(role="user", content="Hi")])
        assert result.content == "Hello!"
        mock_client.chat.assert_called_once()

    async def test_chat_with_tools(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=self._text_response("Sure"))
        provider = self._make_provider(mock_client)

        tools = [ToolDef(name="test", description="A test", parameters={"type": "object", "properties": {}})]
        await provider.chat([Message(role="user", content="Hi")], tools=tools)

        call_kwargs = mock_client.chat.call_args
        assert "tools" in call_kwargs.kwargs

    async def test_chat_no_tools_omits_tools_key(self):
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=self._text_response("Hi"))
        provider = self._make_provider(mock_client)

        await provider.chat([Message(role="user", content="Hi")])
        call_kwargs = mock_client.chat.call_args
        assert "tools" not in call_kwargs.kwargs

    async def test_custom_host(self):
        with patch("ollama.AsyncClient") as MockClient:
            MockClient.return_value = AsyncMock()
            provider = OllamaProvider(model="llama3", host="http://remote:11434")
            MockClient.assert_called_once_with(host="http://remote:11434")

    async def test_default_host(self):
        with patch("ollama.AsyncClient") as MockClient:
            MockClient.return_value = AsyncMock()
            provider = OllamaProvider(model="llama3")
            # No host kwarg passed when host is None
            MockClient.assert_called_once_with()

    async def test_import_error(self):
        with patch.dict("sys.modules", {"ollama": None}):
            with pytest.raises(ImportError, match="Install ollama"):
                provider = OllamaProvider.__new__(OllamaProvider)
                OllamaProvider.__init__(provider, model="llama3")
