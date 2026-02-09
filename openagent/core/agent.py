from __future__ import annotations

import asyncio
from typing import Any, Callable

# Changed imports for core modules
from .logging import AgentLogger
from .session import Session
from .tool import ToolRegistry, tool
from .types import Message
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
    ) -> None:
        self.provider = provider
        self.session = Session(system_prompt=system_prompt)
        self.max_turns = max_turns
        self.tool_registry = ToolRegistry()
        self._logger = AgentLogger(agent_id)

        if tools:
            for fn in tools:
                if not hasattr(fn, "_tool_name"):
                    fn = tool(fn)
                self.tool_registry.register(fn)

    @property
    def messages(self) -> list[Message]:
        return self.session.messages

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

            # Log and execute tool calls in parallel
            for tc in response.tool_calls:
                print(f"  🔧 Tool Call: {tc.name}({tc.arguments})")
            
            tool_tasks = [
                self.tool_registry.execute(tc) for tc in response.tool_calls
            ]
            results = await asyncio.gather(*tool_tasks)
            
            # Log results
            for result in results:
                status = "❌" if result.is_error else "✅"
                preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
                print(f"     {status} Result: {preview}")
            
            self.session.add_tool_results(list(results))

        self._logger.max_turns_reached()
        if response is None:
            raise RuntimeError("Agent loop completed without receiving any response")
        return response.text
