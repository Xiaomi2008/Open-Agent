# OpenAgent

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/openagent.svg)](https://pypi.org/project/openagent/)

A lightweight, async-first agent framework with pluggable LLM providers and comprehensive built-in tools.

## Features

- **Multi-provider support**: OpenAI (GPT), Anthropic (Claude), Google (Gemini), Ollama (local), and LMStudio
- **Async-first**: Built for modern Python 3.11+ async/await patterns
- **Tool system**: Easy function-to-tool conversion with `@tool` decorator
- **Coder Agent**: Specialized agent for code editing tasks with interactive CLI
- **MCP support**: Integrate Model Context Protocol servers for additional tools
- **Built-in tools**: File operations, shell execution, task management, web search, and more
- **Streaming**: Real-time token streaming from all providers
- **Retry logic**: Automatic exponential backoff for transient failures
- **Session persistence**: Save and load conversations to JSON
- **Task tracking**: Built-in TODO manager for multi-step workflow coordination
- **Background execution**: Persistent bash sessions with output retrieval

## Installation

### Using uv (recommended)

```bash
# Install the package
uv pip install openagent

# Install with optional dependencies
uv pip install "openagent[openai]"      # OpenAI support
uv pip install "openagent[anthropic]"   # Anthropic/Claude support
uv pip install "openagent[google]"      # Google/Gemini support
uv pip install "openagent[ollama]"      # Ollama/local models support
uv pip install "openagent[web]"         # Web search capabilities
uv pip install "openagent[all]"         # All providers and tools

# Or use uv to manage a project with openagent
uv init my-project
cd my-project
uv add openagent
```

### Using pip

```bash
pip install openagent

# Install provider dependencies
pip install "openagent[all]"  # All providers and tools
```

## Quick Start

### Basic Agent

```python
import asyncio
from openagent import Agent, OpenAIProvider, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 22°C."

async def main():
    agent = Agent(
        provider=OpenAIProvider(model="gpt-4o"),
        system_prompt="You are a helpful assistant.",
        tools=[get_weather],
    )

    answer = await agent.run("What's the weather in Tokyo?")
    print(answer)

asyncio.run(main())
```

### Coder Agent - Chat-based Coding Assistant

For code editing and development tasks, use the specialized `CoderAgent`:

```python
import asyncio
from openagent import create_coder

# Quick start with defaults
coder = await create_coder(model="gpt-4o")
result = await coder.run("Create a new Python file with a hello world function")

# Or use the full constructor for more control
from openagent import CoderAgent, OpenAIProvider

provider = OpenAIProvider(model="gpt-4-turbo", api_key="sk-...")
coder = CoderAgent(
    provider=provider,
    system_prompt="You are an expert Python developer.",
    max_turns=20,
    working_dir="/path/to/project",
)
result = await coder.run("Add a new endpoint to the API that handles user registration")
```

**Key Features:**
- File operations: Read, write, and edit files seamlessly
- Code search: Find patterns across your project with grep
- Shell commands: Execute bash for testing and building
- Task tracking: Automatically breaks down complex tasks into steps
- Working directory: Set context for file operations

**Quick Commands (in CLI):**
- `@list` - List files in current directory
- `@todo` - Show task list
- `@clear` - Clear conversation history

## Providers

### OpenAI

```python
from openagent import OpenAIProvider

provider = OpenAIProvider(
    model="gpt-4o",           # or "gpt-4-turbo", "gpt-3.5-turbo", etc.
    api_key="sk-...",         # optional, uses OPENAI_API_KEY env var
    max_retries=3,            # retry on transient failures
)
```

### Anthropic (Claude)

```python
from openagent import AnthropicProvider

provider = AnthropicProvider(
    model="claude-sonnet-4-20250514",
    api_key="sk-ant-...",     # optional, uses ANTHROPIC_API_KEY env var
    max_tokens=4096,          # configurable max output tokens
    max_retries=3,
)
```

### Google (Gemini)

```python
from openagent import GoogleProvider

provider = GoogleProvider(
    model="gemini-2.0-flash",
    api_key="...",            # optional, uses GOOGLE_API_KEY env var
    max_retries=3,
)
```

### Ollama (Local Models)

```python
from openagent import OllamaProvider

provider = OllamaProvider(
    model="llama2",           # or any model available in your Ollama instance
    base_url="http://localhost:11434",
    max_retries=3,
)
```

## Tools

Define tools using the `@tool` decorator:

```python
from openagent import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@tool(name="search", description="Search the web for information")
def web_search(query: str, max_results: int = 5) -> str:
    # Your search implementation
    return "Search results..."
```

### Built-in Tools

OpenAgent includes a comprehensive set of built-in tools that can be used directly:

#### File Operations

```python
from openagent.tools import read, write, edit, glob, grep, notebook_edit

# Read file contents (with optional line range)
content = read("path/to/file.txt", line_start=1, line_end=10)

# Write/overwrite files
write("output.txt", "Hello World!", create_parents=True)

# Edit files with string replacement
edit("file.py", old_text="old_value", new_text="new_value")

# Search for files by pattern
files = glob("*.py", path="/project", max_results=50)

# Grep/search file contents
results = grep("function_name", path="/project", regex=True, context_lines=2)

# Edit Jupyter notebook cells
notebook_edit("notebook.ipynb", cell_index=0, new_source="print('hello')")
```

#### Shell & Process Management

```python
from openagent.tools import bash, bash_background, bash_output, kill_shell

# Execute a single command (synchronous)
result = bash("ls -la")

# Start background session and execute commands
session_msg = bash_background("echo Hello; ls -la", working_dir="/project")
# Returns: "Started bash session 'abc123' in '/project'. Use bash_output to retrieve output."

# Retrieve output from a background session
output = bash_output("abc123", tail_lines=50)  # Last 50 lines only

# Terminate a background session
kill_shell("abc123")  # Returns: "Session 'abc123' terminated successfully."
```

#### Task/TODO Manager

Track multi-step workflows with the integrated task manager:

```python
from openagent.tools import todo_write, todo_update, todo_list

# Create multiple tasks at once
tasks = [
    {"subject": "Implement feature A", "description": "Add new functionality", "activeForm": "Implementing feature A"},
    {"subject": "Write tests", "description": "Create unit tests", "activeForm": "Writing tests"},
]
todo_write(tasks)

# Update task status and details
todo_update("task_abc123", status="in_progress")  # pending, in_progress, completed, deleted
todo_update("task_abc123", subject="[DONE] Feature A")

# View all tasks with statuses
print(todo_list())
```

#### Web & Search (requires dependencies)

```bash
pip install duckduckgo-search httpx
```

```python
from openagent.tools import web_search, web_fetch

# Search the web
results = web_search("Python programming", num_results=5)

# Fetch and parse a webpage
content = web_fetch("https://example.com")
```

#### Planning & Workflow

```python
from openagent.tools import enter_plan_mode, exit_plan_mode

# Enter planning mode before complex tasks
enter_plan_mode("Need to design database schema first")

# Exit plan mode when ready to implement
exit_plan_mode("Plan approved: Use PostgreSQL with SQLAlchemy")
```

#### User Interaction

```python
from openagent.tools import ask_user_question

# Ask user for input with options
result = ask_user_question(
    "What approach should we take?",
    options=["Option A", "Option B", "Option C"],
    multi_select=False,
)
```

#### Extensibility

```python
from openagent.tools import skill, slash_command, task

# Launch specialized sub-agents
task("explore", "Find all Python files in the project", None)

# Use custom skills (requires setup)
skill("simplify", None)

# Execute slash commands (requires setup)
slash_command("/review", ["--changed-files"])
```

### Using Built-in Tools with Agent

```python
from openagent import Agent, OpenAIProvider
from openagent.tools import read, write, edit, glob, grep, bash, todo_write

provider = OpenAIProvider(model="gpt-4o")

agent = Agent(
    provider=provider,
    system_prompt="You are a helpful coding assistant with file and shell access.",
    tools=[read, write, edit, glob, grep, bash, todo_write],
)

await agent.run("Create a new Python file with a hello world function")
```

## Streaming

All providers support streaming responses:

```python
async def stream_example():
    provider = OpenAIProvider(model="gpt-4o")
    messages = [Message(role="user", content="Tell me a story")]

    async for chunk in provider.stream(messages):
        print(chunk, end="", flush=True)
```

## Session Persistence

Save and restore conversations:

```python
from openagent import Session

# Create and use session
session = Session(system_prompt="You are helpful.")
session.add("user", "Hello!")
session.add("assistant", "Hi there!")

# Save to file
session.save("conversation.json")

# Load from file
restored = Session.load("conversation.json")
```

## Logging

Enable logging for debugging:

```python
import logging
from openagent import configure_logging

# Enable debug logging
configure_logging(level=logging.DEBUG)

# Or use the logger directly
from openagent import logger
logger.setLevel(logging.INFO)
```

## MCP Integration

Use Model Context Protocol servers to extend the Agent with additional tools:

```python
from openagent import Agent, OpenAIProvider
from openagent.mcp import McpClient

# Using stdio transport (local MCP server)
async def main():
    async with McpClient("npx", ["@modelcontextprotocol/server-filesystem", "/path/to/dir"]) as mcp_client:
        agent = Agent(
            provider=OpenAIProvider(model="gpt-4o"),
            mcp_client=mcp_client,  # Automatically discover and register MCP tools
        )
        result = await agent.run("List files in /path/to/dir")

# Using SSE transport (remote MCP server)
async with McpClient("http://localhost:8000/sse") as mcp_client:
    agent = Agent(
        provider=OpenAIProvider(model="gpt-4o"),
        mcp_client=mcp_client,
    )
```

## API Reference

### Agent

```python
Agent(
    provider: BaseProvider,      # LLM provider instance
    system_prompt: str = "",     # System prompt for the agent
    tools: list[Callable] = [],  # List of tool functions
    max_turns: int = 10,         # Max conversation turns
    agent_id: str = None,        # Optional ID for logging
)

# Methods
await agent.run(user_input: str) -> str
agent.messages -> list[Message]
```

### McpClient

```python
McpClient(command: str, args: list[str] | None = None, env: dict[str, str] | None = None)

# Methods
async with client as mcp_client:
    tools = await mcp_client.get_tools()  # Discover available MCP tools
```

Supports both stdio transport (local subprocesses like `npx @modelcontextprotocol/server-filesystem`) and SSE transport (remote servers at HTTP/SSE endpoints).

### Session

```python
Session(system_prompt: str = "")

# Methods
session.add(role, content) -> Message
session.add_message(message) -> None
session.add_tool_results(results) -> Message
session.clear() -> None
session.save(path) -> None
Session.load(path) -> Session
```

### Task Manager

```python
from openagent.core.task_manager import get_task_manager, TodoTask, TaskStatus

manager = get_task_manager()

# Create tasks
task_id = manager.create_task(
    subject="Implement feature",
    description="Add new functionality",
    active_form="Implementing feature"
)

# Update task status
manager.update_task(task_id, status=TaskStatus.IN_PROGRESS)

# List all tasks
tasks = manager.list_tasks(status_filter=TaskStatus.PENDING)

# Get formatted summary
summary = manager.get_summary()
```

## Project Structure

```
openagent/
├── __init__.py                  # Public API exports
├── core/
│   ├── agent.py                 # Agent class — main orchestrator
│   ├── types.py                 # Canonical types: Message, ToolUseBlock, etc.
│   ├── tool.py                  # @tool decorator and ToolRegistry
│   ├── session.py               # Session management and persistence
│   ├── logging.py               # Logging configuration
│   └── retry.py                 # Retry logic with exponential backoff
├── provider/
│   ├── base.py                  # BaseProvider ABC
│   ├── anthropic.py             # Anthropic/Claude support
│   ├── openai.py                # OpenAI/GPT support
│   ├── google.py                # Google/Gemini support
│   └── ollama.py                # Ollama/local models support
├── tools/                       # Built-in tool implementations
└── mcp.py                       # MCP client integration
```

## License

MIT
