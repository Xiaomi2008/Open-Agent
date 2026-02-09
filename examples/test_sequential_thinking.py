"""
Sequential Thinking MCP Server Test.

Tests the agent's ability to use the sequential-thinking MCP server
for structured problem-solving and step-by-step reasoning.

Usage:
    $env:OLLAMA_HOST="http://localhost:11439"; python test_sequential_thinking.py ollama
    python test_sequential_thinking.py anthropic
"""

import asyncio
import argparse
import os
import time
from dataclasses import dataclass

from openagent import Agent, AnthropicProvider, GoogleProvider, OllamaProvider, OpenAIProvider
from openagent.mcp import McpClient


# ---------- Test Cases for Sequential Thinking ----------

SEQUENTIAL_THINKING_TESTS = [
    {
        "name": "Algorithm Design",
        "description": "Design a sorting algorithm step by step",
        "prompt": """
You have access to a sequential thinking tool. Use it to work through this problem step by step.

Problem: Design an algorithm to find the two numbers in an array that add up to a target sum.

For each step:
1. Use the process_thought tool to record your thinking
2. Go through stages: Problem Definition → Research → Analysis → Synthesis → Conclusion
3. Number your thoughts sequentially
4. At the end, use generate_summary to summarize your thinking process

Work through:
- Understanding the problem constraints
- Considering brute force approach and its complexity
- Designing an optimized solution using a hash map
- Analyzing time and space complexity
- Writing pseudocode for the final solution

Use at least 5-7 thoughts to work through this problem thoroughly.
""",
    },
    {
        "name": "Business Decision Analysis",
        "description": "Analyze a business decision with structured thinking",
        "prompt": """
Use the sequential thinking tools to analyze this business decision:

Scenario: A software company is deciding whether to:
A) Build a new product in-house (6 months, $500k cost)
B) Acquire a smaller competitor that has the product ($2M acquisition)
C) Partner with another company for licensing ($100k/year ongoing)

Work through this decision using process_thought for each step:

1. Problem Definition: Define the key decision criteria
2. Research: Consider factors like time-to-market, total cost of ownership, risk
3. Analysis: Evaluate each option against the criteria
4. Synthesis: Compare the options and identify trade-offs  
5. Conclusion: Make a recommendation with justification

Number your thoughts (thought 1 of N, thought 2 of N, etc.) and set appropriate stages.

After completing your analysis, use generate_summary to provide an overview.
""",
    },
    {
        "name": "Debugging a System",
        "description": "Debug a hypothetical system issue step by step",
        "prompt": """
Use sequential thinking to debug this issue:

Bug Report: Users are experiencing slow page loads (10+ seconds) on the dashboard.
System: Web application with React frontend, Node.js backend, PostgreSQL database.
Recent Changes: Deployed new analytics feature that queries user activity logs.

Use process_thought to work through debugging:

Stage 1 - Problem Definition:
- What exactly is the symptom?
- When did it start?
- Who is affected?

Stage 2 - Research/Hypothesis:
- What could cause slow page loads?
- What are the likely suspects given recent changes?

Stage 3 - Analysis:
- How would you investigate each hypothesis?
- What metrics/logs would you check?

Stage 4 - Synthesis:
- Based on analysis, what's the most likely cause?
- What's the root cause vs symptoms?

Stage 5 - Conclusion:
- What's your recommended fix?
- How would you prevent this in the future?

Use at least 6 thoughts, then generate_summary at the end.
""",
    },
    {
        "name": "Mathematical Proof",
        "description": "Work through a mathematical proof step by step",
        "prompt": """
Use sequential thinking to prove this mathematical statement:

Statement: For any positive integer n, the sum 1 + 2 + 3 + ... + n = n(n+1)/2

Work through a proof by mathematical induction using process_thought:

Thought 1 (Problem Definition): State what we need to prove
Thought 2 (Research): Recall how mathematical induction works
Thought 3 (Analysis - Base Case): Prove it works for n=1
Thought 4 (Analysis - Inductive Hypothesis): State the assumption for n=k  
Thought 5 (Analysis - Inductive Step): Prove it works for n=k+1
Thought 6 (Synthesis): Connect all the pieces
Thought 7 (Conclusion): State the completed proof

After all thoughts, use generate_summary to summarize the proof structure.
""",
    },
    {
        "name": "Complex Planning",
        "description": "Plan a complex project with dependencies",
        "prompt": """
Use sequential thinking to plan launching a mobile app:

Project: Launch a food delivery mobile app in 3 months

Use process_thought to create a detailed plan:

Stage: Problem Definition
- What are the key deliverables?
- What are the hard constraints?

Stage: Research  
- What teams/resources are needed?
- What are the dependencies between tasks?

Stage: Analysis
- Break down into phases
- Identify critical path
- Estimate durations

Stage: Synthesis
- Create a timeline
- Identify risks and mitigations

Stage: Conclusion
- Final project plan summary
- Key milestones and go/no-go criteria

Use 8-10 thoughts to be thorough, then generate_summary for the executive overview.
""",
    },
]


# ---------- Helpers ----------

def get_env_clean(key: str) -> str | None:
    val = os.environ.get(key)
    if val:
        return val.strip()
    for k, v in os.environ.items():
        if k.strip() == key:
            return v.strip()
    return None


def make_provider(name: str):
    providers = {
        "openai": lambda: OpenAIProvider(
            model="gpt-4o",
            api_key=get_env_clean("OPENAI_API_KEY")
        ),
        "anthropic": lambda: AnthropicProvider(
            model="claude-sonnet-4-20250514",
            api_key=get_env_clean("ANTHROPIC_API_KEY")
        ),
        "google": lambda: GoogleProvider(
            model="gemini-2.0-flash",
            api_key=get_env_clean("GOOGLE_API_KEY")
        ),
        "ollama": lambda: OllamaProvider(
            model=os.environ.get("OLLAMA_MODEL", "glm-4.7-flash"),
            host=get_env_clean("OLLAMA_HOST") or "http://localhost:11439",
        ),
    }
    factory = providers.get(name)
    if not factory:
        raise ValueError(f"Unknown provider: {name}")
    return factory()


@dataclass  
class TestResult:
    name: str
    success: bool
    duration_seconds: float
    response_length: int
    tool_calls_seen: int
    error: str | None = None


# ---------- Main ----------

async def run_test(agent: Agent, test_case: dict) -> TestResult:
    """Run a single test case and measure performance."""
    name = test_case["name"]
    prompt = test_case["prompt"]
    
    print(f"\n{'='*70}")
    print(f"🧠 Running Test: {name}")
    print(f"   {test_case['description']}")
    print(f"{'='*70}\n")
    
    # Clear agent session for fresh test
    agent.session.messages.clear()
    
    start_time = time.time()
    
    try:
        response = await agent.run(prompt)
        duration = time.time() - start_time
        
        # Count tool calls from agent's session
        tool_calls_count = sum(
            len(msg.tool_calls) 
            for msg in agent.session.messages 
            if hasattr(msg, 'tool_calls')
        )
        
        print(f"\n📝 Agent Response:\n{'-'*50}")
        print(response)
        print(f"{'-'*50}")
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"🔧 Tool calls made: {tool_calls_count}")
        print(f"📊 Response length: {len(response)} chars")
        
        return TestResult(
            name=name,
            success=True,
            duration_seconds=duration,
            response_length=len(response),
            tool_calls_seen=tool_calls_count,
        )
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        tool_calls_count = sum(
            len(msg.tool_calls) 
            for msg in agent.session.messages 
            if hasattr(msg, 'tool_calls')
        )
        
        return TestResult(
            name=name,
            success=False,
            duration_seconds=duration,
            response_length=0,
            tool_calls_seen=tool_calls_count,
            error=str(e),
        )


async def main():
    parser = argparse.ArgumentParser(description="Sequential Thinking MCP Test")
    parser.add_argument("provider", nargs="?", default="ollama",
                        help="LLM Provider (openai, anthropic, google, ollama)")
    parser.add_argument("--test", type=int, default=None,
                        help="Run specific test (0-4) or all if not specified")
    
    args = parser.parse_args()
    
    provider = make_provider(args.provider)
    
    mcp_cmd = "npx"
    mcp_args = ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    
    print(f"🔌 Connecting to Sequential Thinking MCP server...")
    print(f"   Command: {mcp_cmd} {' '.join(mcp_args)}")
    
    async with McpClient(mcp_cmd, mcp_args) as mcp_client:
        mcp_tools = await mcp_client.get_tools()
        print(f"✅ Found {len(mcp_tools)} MCP tools: {[t._tool_name for t in mcp_tools]}")
        

        
        agent = Agent(
            provider=provider,
            system_prompt="""You are a methodical problem-solver that uses structured thinking.

When given a problem, use the sequential thinking tools to work through it step by step:
- Use process_thought to record each step of your reasoning
- Number your thoughts sequentially (thought 1 of N, thought 2 of N, etc.)
- Use appropriate stages: Problem Definition, Research, Analysis, Synthesis, Conclusion
- At the end, use generate_summary to summarize your thinking process

Be thorough and explicit about your reasoning. Show your work.""",
            tools=mcp_tools,
            max_turns=20,  # Allow more turns for complex reasoning
        )
        
        # Select tests to run
        if args.test is not None:
            tests = [SEQUENTIAL_THINKING_TESTS[args.test]]
        else:
            tests = SEQUENTIAL_THINKING_TESTS
        
        results: list[TestResult] = []
        
        for test_case in tests:
            # Clear history before each test
            try:
                for tool in mcp_tools:
                    if tool._tool_name == "clear_history":
                        await tool()
                        print("🧹 Cleared thought history")
                        break
            except Exception:
                pass  # Ignore if clear fails
            
            result = await run_test(agent, test_case)
            results.append(result)
        
        # Print summary
        print(f"\n{'='*70}")
        print("📊 TEST SUMMARY - Sequential Thinking")
        print(f"{'='*70}")
        
        total_duration = sum(r.duration_seconds for r in results)
        total_tool_calls = sum(r.tool_calls_seen for r in results)
        successful = sum(1 for r in results if r.success)
        
        for r in results:
            status = "✅ PASS" if r.success else "❌ FAIL"
            print(f"  {status} {r.name}")
            print(f"       Duration: {r.duration_seconds:.2f}s | Tool calls: {r.tool_calls_seen} | Response: {r.response_length} chars")
            if r.error:
                print(f"       Error: {r.error}")
        
        print(f"\n  📈 Results: {successful}/{len(results)} passed")
        print(f"  ⏱️  Total duration: {total_duration:.2f}s")
        print(f"  🔧 Total tool calls: {total_tool_calls}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
