#!/usr/bin/env python3
"""Example usage of the Coder Agent.

This script demonstrates how to use the CoderAgent for code editing tasks.
Run with: uv run examples/coder_example.py
"""

import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from openagent.coder import CoderAgent, create_coder
from openagent.provider.openai import OpenAIProvider


async def example_basic_usage():
    """Basic usage of CoderAgent."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    coder = await create_coder(model="gpt-4o", max_turns=10)

    # Example request - create a simple Python file
    result = await coder.run(
        "Create a new Python file called 'hello.py' with a function "
        "called 'hello_world' that prints 'Hello, World!'"
    )

    print(f"\nResult: {result}")


async def example_read_file():
    """Read and analyze an existing file."""
    print("\n" + "=" * 60)
    print("Example 2: Read File")
    print("=" * 60)

    coder = await create_coder(model="gpt-4o", max_turns=10)

    # Example request - read a file
    result = await coder.run(
        "Read the README.md file and tell me what features it describes"
    )

    print(f"\nResult: {result}")


async def example_search_codebase():
    """Search the codebase for patterns."""
    print("\n" + "=" * 60)
    print("Example 3: Search Codebase")
    print("=" * 60)

    coder = await create_coder(model="gpt-4o", max_turns=10)

    # Example request - search for patterns
    result = await coder.run(
        "Search for all files that contain 'class Agent' and list them"
    )

    print(f"\nResult: {result}")


async def example_edit_file():
    """Edit an existing file."""
    print("\n" + "=" * 60)
    print("Example 4: Edit File")
    print("=" * 60)

    coder = await create_coder(model="gpt-4o", max_turns=15)

    # Example request - edit a file
    result = await coder.run(
        "Edit the cli_coder.py file to add a --version flag that prints "
        "the OpenAgent version"
    )

    print(f"\nResult: {result}")


async def example_shell_commands():
    """Execute shell commands."""
    print("\n" + "=" * 60)
    print("Example 5: Shell Commands")
    print("=" * 60)

    coder = await create_coder(model="gpt-4o", max_turns=10)

    # Example request - run shell commands
    result = await coder.run(
        "List all Python files in the openagent directory and count them"
    )

    print(f"\nResult: {result}")


async def example_complex_task():
    """Handle a complex multi-step task."""
    print("\n" + "=" * 60)
    print("Example 6: Complex Task")
    print("=" * 60)

    coder = await create_coder(model="gpt-4o", max_turns=20)

    # Example request - complex task with multiple steps
    result = await coder.run(
        """I want to add a new utility function to openagent/utils.py:

1. First, check if the file exists
2. If it doesn't exist, create it with basic structure
3. Add a function called 'format_duration' that takes seconds (float)
   and returns a human-readable string like "2h 30m 15s"
4. Include docstring and type hints

Let me know what you find!"""
    )

    print(f"\nResult: {result}")


async def example_with_working_dir():
    """Use CoderAgent with a specific working directory."""
    print("\n" + "=" * 60)
    print("Example 7: Working Directory")
    print("=" * 60)

    # Set working directory to the examples folder
    examples_dir = str(project_root / "examples")

    coder = await create_coder(
        model="gpt-4o",
        max_turns=10,
        working_dir=examples_dir
    )

    result = await coder.run(
        f"List files in the current directory ({examples_dir}) and tell me "
        "what Python example files are available"
    )

    print(f"\nResult: {result}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CODER AGENT EXAMPLES")
    print("=" * 60)
    print("\nThis script demonstrates various ways to use the CoderAgent.")
    print("Each example shows a different capability.\n")

    # Run examples sequentially
    try:
        await example_basic_usage()
        input("\nPress Enter to continue...")

        await example_read_file()
        input("\nPress Enter to continue...")

        await example_search_codebase()
        input("\nPress Enter to continue...")

        await example_edit_file()
        input("\nPress Enter to continue...")

        await example_shell_commands()
        input("\nPress Enter to continue...")

        await example_complex_task()
        input("\nPress Enter to continue...")

        await example_with_working_dir()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())
