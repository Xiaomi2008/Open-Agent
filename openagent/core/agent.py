from __future__ import annotations

import asyncio
from typing import Any, Callable

# Changed imports for core modules
from .bash_manager import BashManager, get_bash_manager
from .display import (
    bold, dim, cyan, green, red, white,
    display_tool_call_claude_style, display_tool_result_claude_style,
    format_diff_output, display_code_block, display_diff_claude_style,
    truncate_text
)


def display_write_result(file_path: str, result_content: str) -> None:
    """Display write operation result with file info."""
    print(f"  ● {bold('write')}({cyan(f'\"{file_path}\"')})")

    # Parse the success message to show bytes written
    if 'Successfully wrote' in result_content:
        parts = result_content.split()
        for i, part in enumerate(parts):
            if part == 'wrote':
                bytes_written = parts[i + 1]
                print(f"    ⎿ {green(bytes_written)} to {cyan(file_path)}")
                break

    # Extract and show the actual content that was written (skip the success message)
    lines = result_content.split('\n')
    content_start = False
    for line in lines:
        if 'Successfully wrote' in line:
            continue
        if line.strip():  # Non-empty line
            if not content_start:
                print()
                content_start = True
            print(f"    {dim(line)}")

    if content_start:
        print()


def display_edit_result(file_path: str, result_content: str) -> None:
    """Display edit operation result with changed lines highlighted."""
    print(f"  ● {bold('edit')}({cyan(f'\"{file_path}\"')})")

    # Check if this is a unified diff format
    has_diff_format = '@@' in result_content and any(
        line.startswith('+') or line.startswith('-')
        for line in result_content.split('\n')[1:]
    )

    if has_diff_format:
        # Parse the success message first
        if 'Successfully made' in result_content:
            parts = result_content.split()
            for i, part in enumerate(parts):
                if part == 'made':
                    count = parts[i + 1]
                    print(f"    ⎿ {green(count)} replacement(s) made")
                    break

        # Show the unified diff with color coding
        lines = result_content.split('\n')
        for line in lines:
            if not line or 'Successfully' in line:
                continue
            if line.startswith('@@'):
                print(f"  {bold(cyan(line))}")
            elif line.startswith('+') and not line.startswith('+++'):
                print(f"    {green(line[1:])}")
            elif line.startswith('-') and not line.startswith('---'):
                print(f"    {red(line[1:])}")
            else:
                print(f"    {dim(line)}")

        return

    # Fallback for non-diff format results - show full content without truncation
    if 'Successfully made' in result_content:
        parts = result_content.split()
        for i, part in enumerate(parts):
            if part == 'made':
                count = parts[i + 1]
                print(f"    ⎿ {green(count)} replacement(s) made")
                break

    # Show full content without truncation
    lines = result_content.split('\n')
    for line in lines:
        if line.strip():
            print(f"    {dim(line)}")


from .logging import AgentLogger
from .session import Session
from .skill_manager import (
    SkillManager,
    SlashCommandRegistry,
    get_command_registry,
    get_skill_manager,
)
from .task_manager import TaskManager, get_task_manager
from .tool import ToolRegistry, tool
from .token_counter import (
    estimate_conversation_tokens,
    estimate_message_tokens,
    get_context_window,
)
from .types import Message, TextBlock, ToolUseBlock, ToolResultBlock

# BaseProvider is likely in parent package or sibling 'provider' package
# Since we are in core/, provider/ is '../provider/'
# But 'openagent.provider' is absolute import, which is fine and clearer.
from openagent.provider.base import BaseProvider


class Agent:
    def __init__(
        self,
        provider: BaseProvider,
        system_prompt: str = "",
        tools: list[Callable[..., Any]] | None = None,
        max_turns: int = 10,
        agent_id: str | None = None,
        bash_manager: BashManager | None = None,
        task_manager: TaskManager | None = None,
        skill_manager: SkillManager | None = None,
        mcp_client: Any | None = None,  # MCP client for tool discovery
        # Compaction parameters
        context_budget: int | None = None,  # max tokens per turn; None disables compaction
        keep_tail: int = 5,                 # recent turns to preserve unchanged
        compaction_prompt: str | None = None,
    ) -> None:
        self.provider = provider
        self.session = Session(system_prompt=system_prompt)
        self.max_turns = max_turns
        self.tool_registry = ToolRegistry()
        self._logger = AgentLogger(agent_id)

        # Initialize managers (use provided or create new instances)
        self.bash_manager = bash_manager or get_bash_manager()
        self.task_manager = task_manager or get_task_manager()
        self.skill_manager = skill_manager or get_skill_manager()
        self.command_registry = get_command_registry()

        # Compaction configuration
        if context_budget is None:
            context_budget = int(get_context_window(provider.model) * 0.75)
        self.context_budget = context_budget
        self.keep_tail = keep_tail
        self.compaction_prompt = compaction_prompt or (
            "Summarize the following conversation excerpt concisely, preserving "
            "all important decisions, facts, and action items. "
            "Focus on information that would be needed to continue the discussion."
        )
        self._summary_text: str = ""  # persists across multiple compactions

        if tools:
            for fn in tools:
                if not hasattr(fn, "_tool_name"):
                    fn = tool(fn)
                self.tool_registry.register(fn)

        # Integrate MCP client if provided - discover and register MCP tools
        self._mcp_client = mcp_client
        if mcp_client is not None:
            asyncio.run(self._integrate_mcp_tools())

    async def _integrate_mcp_tools(self) -> None:
        """Discover and integrate MCP tools from the client."""
        try:
            # Ensure the client is connected
            if hasattr(self._mcp_client, '__aenter__'):
                await self._mcp_client.__aenter__()

            # Get MCP tools
            mcp_tools = await self._mcp_client.get_tools()

            # Register each MCP tool
            for tool_fn in mcp_tools:
                if hasattr(tool_fn, "_tool_name"):
                    self.tool_registry.register(tool_fn)
                    self._logger.info(f"Registered MCP tool: {tool_fn._tool_name}")

        except Exception as e:
            self._logger.error(f"Failed to integrate MCP tools: {e}")

    @property
    def messages(self) -> list[Message]:
        return self.session.messages

    async def _should_compact(self) -> bool:
        """Check if conversation exceeds the context budget."""
        if self.context_budget <= 0:
            return False
        total = estimate_conversation_tokens(self.session.messages)
        return total > self.context_budget

    async def _compact(self) -> None:
        """Compress older messages into a summary to reduce context usage."""
        messages = self.session.messages
        if not messages:
            return

        # Determine how many turns constitute a "turn" (user+assistant pairs)
        # We keep the last keep_tail turns untouched
        tail_count = min(self.keep_tail * 2, len(messages))
        head = messages[:-tail_count] if tail_count < len(messages) else messages
        tail = messages[-tail_count:] if tail_count < len(messages) else []

        if not head:
            return

        # Build text from head messages for summarization
        parts: list[str] = []
        for msg in head:
            if isinstance(msg.content, str):
                parts.append(f"{msg.role}: {msg.content}")
            else:
                block_text = ""
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        block_text += block.text
                    elif isinstance(block, ToolUseBlock):
                        block_text += f"[tool call: {block.name}]"
                    elif isinstance(block, ToolResultBlock):
                        # Truncate long tool results for summarization
                        content = block.content
                        if len(content) > 500:
                            content = content[:500] + "..."
                        block_text += f"[tool result: {content}]"
                parts.append(f"{msg.role}: {block_text}")

        head_text = "\n".join(parts)

        # Build summarization prompt
        summary_context = ""
        if self._summary_text:
            summary_context = (
                f"\n\nPrevious conversation summary:\n{self._summary_text}\n"
            )

        compaction_prompt = (
            f"{self.compaction_prompt}\n"
            f"{summary_context}"
            f"\nConversation excerpt to summarize:\n---\n{head_text}\n---"
        )

        try:
            summary_response = await self.provider.chat(
                messages=[Message(role="user", content=compaction_prompt)],
                system_prompt="You are a concise summarizer.",
            )
            self._summary_text = summary_response.text
        except Exception:
            # Fallback: truncate oldest messages instead of summarizing
            if len(messages) > tail_count + 2:
                new_messages = [messages[0]] + tail  # keep system + tail
                self.session.replace_messages(new_messages)
            return

        # Replace head with summary + tail
        summary_msg = Message(
            role="user",
            content=f"Earlier conversation summary: {self._summary_text}",
        )
        self.session.replace_messages([summary_msg] + tail)

    async def run(self, user_input: str, **kwargs: Any) -> str:
        self._logger.run_start(user_input)
        self.session.add("user", user_input)
        result = await self._loop(**kwargs)
        return result

    async def _loop(self, **kwargs: Any) -> str:
        tool_defs = self.tool_registry.definitions if len(self.tool_registry) > 0 else None
        response: Message | None = None

        for turn in range(self.max_turns):
            self._logger.turn_start(turn + 1, self.max_turns)

            # Auto-compaction: compress context if over budget
            if self.context_budget and await self._should_compact():
                await self._compact()

            response = await self.provider.chat(
                messages=self.session.messages,
                tools=tool_defs,
                system_prompt=self.session.system_prompt,
                **kwargs,
            )
            self.session.add_message(response)

            has_tools = response.has_tool_calls
            self._logger.turn_end(turn + 1, has_tools)

            if not has_tools:
                self._logger.run_end(turn + 1)
                return response.text

            # Log and execute tool calls in Claude Code style
            for tc in response.tool_calls:
                display_tool_call_claude_style(tc.name, tc.arguments)

            tool_tasks = [
                self.tool_registry.execute(tc) for tc in response.tool_calls
            ]
            results = await asyncio.gather(*tool_tasks)

            # Log results in Claude Code style with diff highlighting for code changes
            for result, tc in zip(results, response.tool_calls):
                if result.is_error:
                    display_tool_result_claude_style(result.is_error, result.content)
                else:
                    content = result.content

                    # Check if this is a write/edit operation that should show diff formatting
                    is_write_operation = tc.name in ('write', 'edit')

                    if is_write_operation:
                        # Get file path from tool arguments and make it relative
                        full_path = str(tc.arguments.get('file', tc.arguments.get('path', 'unknown')))

                        try:
                            import os
                            cwd = os.getcwd()
                            if full_path.startswith(cwd):
                                rel_path = os.path.relpath(full_path, cwd)
                            else:
                                rel_path = full_path
                        except Exception:
                            rel_path = full_path

                        # For write operations, show success message with file info
                        if tc.name == 'write':
                            display_write_result(rel_path, content)
                        # For edit operations, try to extract and show the changed lines
                        elif tc.name == 'edit':
                            display_edit_result(rel_path, content)

                    else:
                        display_tool_result_claude_style(result.is_error, content)

            self.session.add_tool_results(list(results))

        self._logger.max_turns_reached()
        if response is None:
            raise RuntimeError("Agent loop completed without receiving any response")
        return response.text
