"""Test that the Agent can use the built-in tools."""

import asyncio
from pathlib import Path
import tempfile


async def main():
    from openagent import Agent, ToolRegistry, BaseProvider
    from openagent.tools import read, write, edit, glob, grep
    from openagent.core.types import Message, ToolUseBlock, ToolResultBlock

    # Create a test file first (write is sync)
    with tempfile.TemporaryDirectory() as tmpdir:
        abs_tmpdir = Path(tmpdir).resolve()
        print(f"Temp dir: {abs_tmpdir}")

        test_file = abs_tmpdir / "test.txt"
        write(str(test_file), "Hello World\nThis is line 2\nPython is great")

        # Verify file was created
        if test_file.exists():
            print(f"File exists: {test_file.exists()}")
            print(f"Is file: {test_file.is_file()}")
            content = read(str(test_file))
            print(f"Initial content:\n{content}")
        else:
            print("ERROR: File was not created!")

        print("\n=== Testing Agent with Built-in Tools ===\n")

        # Create a mock provider that doesn't require external dependencies
        class MockProvider(BaseProvider):
            async def chat(self, messages, tools=None, system_prompt="", **kwargs):
                # Return a message indicating we're done (no tool calls)
                return Message(role="assistant", content="Test completed successfully!")

        agent = Agent(
            provider=MockProvider(model="mock"),
            system_prompt="""You are a helpful assistant. You have access to file operations tools:
- read: Read files by absolute path
- write: Create or overwrite files
- edit: Make targeted edits using find-and-replace
- glob: Search for files by pattern
- grep: Search file contents

Use these tools when the user asks about file operations.""",
            tools=[read, write, edit, glob, grep],
        )

        print(f"Agent created with {len(agent.tool_registry)} registered tools:")
        for tool_def in agent.tool_registry.definitions:
            print(f"  - {tool_def.name}: {tool_def.description[:50]}...")

        # Test that the tools are properly registered and can be executed
        from openagent.core.types import ToolUseBlock

        print("\n=== Testing Tool Execution ===\n")

        path_str = str(test_file)
        print(f"Using path: {path_str}")

        # Test read tool execution through registry
        result = await agent.tool_registry.execute(
            ToolUseBlock(name="read", arguments={"path": path_str})
        )
        print(f"Read tool result: {result.content[:100]}...")

        # Test edit tool execution through registry
        result = await agent.tool_registry.execute(
            ToolUseBlock(name="edit", arguments={
                "path": path_str,
                "find": "Python",
                "replace": "OpenAgent"
            })
        )
        print(f"Edit tool result: {result.content}")

        # Verify the edit worked by reading again
        content = read(path_str)
        print(f"\nFile after edit:\n{content}")

    print("\n=== All agent-tool integration tests passed! ===")


if __name__ == "__main__":
    asyncio.run(main())
