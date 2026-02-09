"""Tests for tool decorator and registry."""

from __future__ import annotations

import pytest

from openagent import tool
from openagent.tool import ToolRegistry, _build_parameters_schema
from openagent.types import ToolUseBlock


def test_tool_decorator_basic():
    """Test basic tool decorator."""

    @tool
    def simple_tool(x: str) -> str:
        """A simple tool."""
        return x

    assert hasattr(simple_tool, "_tool_name")
    assert simple_tool._tool_name == "simple_tool"
    assert simple_tool._tool_description == "A simple tool."


def test_tool_decorator_custom_name():
    """Test tool decorator with custom name."""

    @tool(name="custom_name", description="Custom description")
    def my_tool(x: str) -> str:
        return x

    assert my_tool._tool_name == "custom_name"
    assert my_tool._tool_description == "Custom description"


def test_build_parameters_schema():
    """Test parameter schema generation."""

    def func(name: str, count: int, active: bool = True) -> str:
        return ""

    schema = _build_parameters_schema(func)

    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "count" in schema["properties"]
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["required"] == ["name", "count"]


def test_registry_register():
    """Test tool registry registration."""
    registry = ToolRegistry()

    @tool
    def my_tool(x: str) -> str:
        return x

    registry.register(my_tool)

    assert len(registry) == 1
    assert registry.get("my_tool") is not None


def test_registry_definitions():
    """Test tool registry definitions export."""
    registry = ToolRegistry()

    @tool
    def get_data(query: str) -> str:
        """Fetch data."""
        return query

    registry.register(get_data)
    defs = registry.definitions

    assert len(defs) == 1
    assert defs[0].name == "get_data"
    assert defs[0].description == "Fetch data."


async def test_registry_execute():
    """Test tool execution."""
    registry = ToolRegistry()

    @tool
    def add(a: int, b: int) -> int:
        return a + b

    registry.register(add)

    call = ToolUseBlock(id="123", name="add", arguments={"a": 2, "b": 3})
    result = await registry.execute(call)

    assert result.content == "5"
    assert not result.is_error


async def test_registry_execute_not_found():
    """Test execution of missing tool."""
    registry = ToolRegistry()

    call = ToolUseBlock(id="123", name="missing", arguments={})
    result = await registry.execute(call)

    assert result.is_error
    assert "not found" in result.content


async def test_registry_execute_error():
    """Test tool that raises exception."""
    registry = ToolRegistry()

    @tool
    def fail() -> str:
        raise ValueError("Intentional failure")

    registry.register(fail)

    call = ToolUseBlock(id="123", name="fail", arguments={})
    result = await registry.execute(call)

    assert result.is_error
    assert "Intentional failure" in result.content


async def test_async_tool():
    """Test async tool execution."""
    registry = ToolRegistry()

    @tool
    async def async_fetch(url: str) -> str:
        return f"fetched {url}"

    registry.register(async_fetch)

    call = ToolUseBlock(id="123", name="async_fetch", arguments={"url": "test.com"})
    result = await registry.execute(call)

    assert result.content == "fetched test.com"
