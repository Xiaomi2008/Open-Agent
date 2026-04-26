"""Tests for Session class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from openagent import Session
from openagent.core.types import Message, TextBlock, ToolResultBlock, ToolUseBlock


def test_session_init():
    """Test session initialization."""
    session = Session(system_prompt="Test prompt")

    assert session.system_prompt == "Test prompt"
    assert len(session) == 0
    assert session.messages == []


def test_session_add_message():
    """Test adding messages."""
    session = Session()

    msg = session.add("user", "Hello!")
    assert len(session) == 1
    assert msg.role == "user"
    assert msg.content == "Hello!"


def test_session_add_tool_results():
    """Test adding tool results."""
    session = Session()

    results = [
        ToolResultBlock(tool_use_id="123", content="Result"),
    ]
    msg = session.add_tool_results(results)

    assert len(session) == 1
    assert msg.role == "tool_result"


def test_session_clear():
    """Test clearing session."""
    session = Session()
    session.add("user", "Hello!")
    session.add("assistant", "Hi!")

    assert len(session) == 2
    session.clear()
    assert len(session) == 0


def test_session_to_list():
    """Test session serialization to list."""
    session = Session()
    session.add("user", "Hello!")
    session.add("assistant", "Hi!")

    data = session.to_list()

    assert len(data) == 2
    assert data[0]["role"] == "user"
    assert data[0]["content"] == "Hello!"


def test_session_to_list_complex():
    """Test serialization with complex content blocks."""
    session = Session()
    session.add("user", "Help me")

    # Add a message with multiple content blocks
    msg = Message(
        role="assistant",
        content=[
            TextBlock(text="Sure!"),
            ToolUseBlock(id="123", name="search", arguments={"q": "test"}),
        ],
    )
    session.add_message(msg)

    data = session.to_list()

    assert len(data) == 2
    assert len(data[1]["content"]) == 2
    assert data[1]["content"][0]["type"] == "text"
    assert data[1]["content"][1]["type"] == "tool_use"


def test_session_save_load():
    """Test session persistence."""
    session = Session(system_prompt="Test system")
    session.add("user", "Hello!")
    session.add("assistant", "Hi there!")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        session.save(path)

        # Verify file was created
        assert Path(path).exists()

        # Load and verify
        loaded = Session.load(path)
        assert loaded.system_prompt == "Test system"
        assert len(loaded) == 2
        assert loaded.messages[0].content == "Hello!"
        assert loaded.messages[1].content == "Hi there!"
    finally:
        Path(path).unlink()


def test_session_save_load_complex():
    """Test persistence with complex content."""
    session = Session(system_prompt="Complex test")

    # Add message with tool use
    msg = Message(
        role="assistant",
        content=[
            TextBlock(text="Running tool"),
            ToolUseBlock(id="abc", name="search", arguments={"query": "test"}),
        ],
    )
    session.add_message(msg)

    # Add tool result
    session.add_tool_results([
        ToolResultBlock(tool_use_id="abc", content="Found it", is_error=False),
    ])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        session.save(path)
        loaded = Session.load(path)

        assert len(loaded) == 2

        # Check tool use was preserved
        first_msg = loaded.messages[0]
        assert len(first_msg.content) == 2
        assert isinstance(first_msg.content[0], TextBlock)
        assert isinstance(first_msg.content[1], ToolUseBlock)
        assert first_msg.content[1].name == "search"

        # Check tool result was preserved
        second_msg = loaded.messages[1]
        assert isinstance(second_msg.content[0], ToolResultBlock)
        assert second_msg.content[0].content == "Found it"
    finally:
        Path(path).unlink()
