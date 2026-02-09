# OpenAgent

A lightweight, async-first agent framework with pluggable LLM providers.

## Features

- **Multi-provider support**: OpenAI, Anthropic (Claude), and Google (Gemini)
- **Async-first**: Built for modern Python async/await patterns
- **Tool support**: Easy function-to-tool conversion with `@tool` decorator
- **Streaming**: Real-time token streaming from all providers
- **Retry logic**: Automatic exponential backoff for transient failures
- **Session persistence**: Save and load conversations
- **Logging**: Built-in observability for debugging

## Installation

```bash
pip install openagent

# Install provider dependencies
pip install openagent[openai]      # OpenAI
pip install openagent[anthropic]   # Anthropic
pip install openagent[google]      # Google
pip install openagent[all]         # All providers
```

## Quick Start

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

### Anthropic

```python
from openagent import AnthropicProvider

provider = AnthropicProvider(
    model="claude-sonnet-4-20250514",
    api_key="sk-ant-...",     # optional, uses ANTHROPIC_API_KEY env var
    max_tokens=4096,          # configurable max output tokens
    max_retries=3,
)
```

### Google

```python
from openagent import GoogleProvider

provider = GoogleProvider(
    model="gemini-2.0-flash",
    api_key="...",            # optional, uses GOOGLE_API_KEY env var
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

## License

MIT
