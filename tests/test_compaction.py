"""Tests for auto-compaction: token estimation and compaction logic."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from openagent.core.agent import Agent
from openagent.core.session import Session
from openagent.core.token_counter import (
    estimate_conversation_tokens,
    estimate_message_tokens,
    estimate_tokens,
    get_context_window,
)
from openagent.core.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from tests.conftest import MockProvider


# ---------------------------------------------------------------------------
# Token estimation tests
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_english_text(self):
        tokens = estimate_tokens("Hello, how are you today?")
        assert tokens > 0
        # Rough: ~6-8 tokens for this sentence
        assert tokens < 30

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = estimate_tokens(text)
        assert tokens > 50

    def test_cjk_text(self):
        tokens = estimate_tokens("你好世界，这是一个测试")
        assert tokens > 0

    def test_code_text(self):
        code = 'def hello():\n    print("world")\n    return True'
        tokens = estimate_tokens(code)
        assert tokens > 0


class TestEstimateMessageTokens:
    def test_simple_text_message(self):
        msg = Message(role="user", content="Hello world")
        tokens = estimate_message_tokens(msg)
        assert tokens > estimate_tokens("Hello world")  # includes overhead

    def test_message_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content=[
                TextBlock(text="Let me check."),
                ToolUseBlock(id="call_1", name="read_file", arguments={"path": "x.py"}),
            ],
        )
        tokens = estimate_message_tokens(msg)
        assert tokens > 10

    def test_message_with_tool_results(self):
        msg = Message(
            role="tool_result",
            content=[
                ToolResultBlock(tool_use_id="call_1", content="file contents here"),
            ],
        )
        tokens = estimate_message_tokens(msg)
        assert tokens > 0


class TestGetContextWindow:
    def test_gpt4(self):
        assert get_context_window("gpt-4o") == 128_000

    def test_claude(self):
        assert get_context_window("claude-sonnet-4-20250514") == 200_000

    def test_unknown_model(self):
        assert get_context_window("unknown-model-xyz") == 128_000

    def test_gemini(self):
        ctx = get_context_window("gemini-2.0-flash")
        assert ctx > 0


# ---------------------------------------------------------------------------
# Compaction logic tests
# ---------------------------------------------------------------------------

class TestShouldCompact:
    @pytest.mark.asyncio
    async def test_no_compaction_when_under_budget(self):
        provider = MockProvider([Message(role="assistant", content="OK")])
        agent = Agent(
            provider=provider,
            context_budget=10_000,
        )
        agent.session.add("user", "Hello")
        assert not await agent._should_compact()

    @pytest.mark.asyncio
    async def test_compaction_when_over_budget(self):
        provider = MockProvider([Message(role="assistant", content="OK")])
        agent = Agent(
            provider=provider,
            context_budget=50,  # very small budget
        )
        # Fill session with enough messages to exceed budget
        for i in range(10):
            agent.session.add("user", f"Message number {i} with some padding text")
            agent.session.add("assistant", f"Response number {i} with some padding text")

        assert await agent._should_compact()

    @pytest.mark.asyncio
    async def test_zero_budget_disables_compaction(self):
        provider = MockProvider([Message(role="assistant", content="OK")])
        agent = Agent(
            provider=provider,
            context_budget=0,
        )
        agent.session.add("user", "Hello" * 100)
        assert not await agent._should_compact()


class TestCompact:
    @pytest.mark.asyncio
    async def test_compact_reduces_message_count(self):
        responses = [
            Message(role="assistant", content="Response")  # for compaction summary
        ]
        provider = MockProvider(responses)
        agent = Agent(
            provider=provider,
            context_budget=50,
            keep_tail=2,
        )
        for i in range(10):
            agent.session.add("user", f"Message {i}")
            agent.session.add("assistant", f"Response {i}")

        initial_count = len(agent.session.messages)
        await agent._compact()
        assert len(agent.session.messages) < initial_count

    @pytest.mark.asyncio
    async def test_compact_preserves_tail(self):
        responses = [
            Message(role="assistant", content="Summary: earlier discussion.")
        ]
        provider = MockProvider(responses)
        agent = Agent(
            provider=provider,
            context_budget=50,
            keep_tail=2,
        )
        for i in range(5):
            agent.session.add("user", f"Msg {i}")
            agent.session.add("assistant", f"Resp {i}")

        # The last 4 messages (2 turns) should be preserved
        last_user = f"Msg {4}"
        last_assistant = f"Resp {4}"
        await agent._compact()

        # Check tail messages are still present
        messages = agent.session.messages
        found_tail = False
        for msg in messages:
            if msg.text == last_user or msg.text == last_assistant:
                found_tail = True
        assert found_tail

    @pytest.mark.asyncio
    async def test_multiple_compactions_merge_summary(self):
        responses = [
            Message(role="assistant", content="Summary A."),  # first compaction
            Message(role="assistant", content="Summary B."),  # second compaction
        ]
        provider = MockProvider(responses)
        agent = Agent(
            provider=provider,
            context_budget=30,
            keep_tail=1,
        )
        for i in range(10):
            agent.session.add("user", f"Msg {i}")
            agent.session.add("assistant", f"Resp {i}")

        await agent._compact()
        first_summary = agent._summary_text
        await agent._compact()

        # After second compaction, summary should have been updated
        assert agent._summary_text == "Summary B."


class TestCompactFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_provider_error(self):
        provider = MockProvider([])
        # Patch provider.chat to raise an error
        provider.chat = AsyncMock(side_effect=RuntimeError("API error"))
        agent = Agent(
            provider=provider,
            context_budget=50,
            keep_tail=2,
        )
        for i in range(10):
            agent.session.add("user", f"Message {i}")
            agent.session.add("assistant", f"Response {i}")

        initial_count = len(agent.session.messages)
        await agent._compact()  # should not raise
        # Fallback reduces messages
        assert len(agent.session.messages) < initial_count


# ---------------------------------------------------------------------------
# End-to-end: agent with compaction in long conversation
# ---------------------------------------------------------------------------

class TestAgentCompactionIntegration:
    @pytest.mark.asyncio
    async def test_agent_compacts_during_long_conversation(self):
        """Verify that compaction triggers during a long run without breaking."""
        # We simulate a long conversation by giving a small context budget
        # and making the provider return increasingly long messages.

        call_count = 0

        class ChattyMockProvider:
            model = "mock-model"

            async def chat(self, messages, tools=None, system_prompt="", **kwargs):
                nonlocal call_count
                call_count += 1
                # Return a long response to quickly fill context
                return Message(
                    role="assistant",
                    content="This is a rather long response to fill up the context window quickly. " * 20,
                )

        provider = ChattyMockProvider()
        agent = Agent(
            provider=provider,
            context_budget=200,  # very small budget to force compaction
            keep_tail=2,
            max_turns=5,
        )

        # This should complete without error even though context grows
        result = await agent.run("Hello, let's start a conversation.")
        assert isinstance(result, str)
        assert call_count > 0
