"""
Long Task Performance Test for OpenAgent with MCP Servers.

This script tests the agent's ability to handle complex, multi-step tasks
using MCP tools. It measures performance, tool call counts, and task completion.

Usage:
    python test_long_task.py ollama --mcp "npx -y @modelcontextprotocol/server-memory"
    python test_long_task.py anthropic --mcp "npx -y @modelcontextprotocol/server-memory"
"""

import asyncio
import argparse
import shlex
import time
import os
from dataclasses import dataclass

from openagent import Agent, AnthropicProvider, GoogleProvider, OllamaProvider, OpenAIProvider


# ---------- Test Cases ----------

LONG_TASK_PROMPTS = [
    {
        "name": "Knowledge Graph Construction",
        "description": "Build a knowledge graph about a software team",
        "prompt": """
You have access to a knowledge graph memory system. Please complete the following complex task:

1. Create entities for a software development team with these members:
   - Alice (role: Tech Lead, expertise: Python, Go)
   - Bob (role: Backend Developer, expertise: Java, databases)
   - Carol (role: Frontend Developer, expertise: React, TypeScript)
   - David (role: DevOps Engineer, expertise: Kubernetes, AWS)
   
2. Create relationships between team members:
   - Alice manages Bob, Carol, and David
   - Bob collaborates_with Carol on the API project
   - David supports all other team members
   
3. Add observations about their current work:
   - Alice is reviewing the Q1 architecture proposal
   - Bob is optimizing database queries for the user service
   - Carol is implementing the new dashboard design
   - David is setting up the CI/CD pipeline for microservices

4. After creating everything, read the full graph to verify the structure.

5. Search for all entities related to "Python" expertise.

Please execute each step using the available tools and report what you've created.
""",
        "expected_tool_calls": 10,  # Minimum expected tool calls
    },
    {
        "name": "Research and Organize",
        "description": "Research a topic and organize findings in memory",
        "prompt": """
You have access to a knowledge graph memory. Please complete this research and organization task:

1. Create entities for key concepts in machine learning:
   - Neural Networks (type: concept, category: deep_learning)
   - Transformers (type: concept, category: attention_models)
   - Reinforcement Learning (type: concept, category: learning_paradigm)
   - Computer Vision (type: concept, category: application_domain)
   - Natural Language Processing (type: concept, category: application_domain)

2. Create relationships showing how these concepts relate:
   - Transformers are used_in NLP
   - Neural Networks are foundation_of Transformers
   - Reinforcement Learning uses Neural Networks
   - Computer Vision uses Neural Networks
   - Transformers improved Computer Vision (through Vision Transformers)

3. Add detailed observations to each concept:
   - For Neural Networks: "Invented in 1943, popularized with backpropagation in 1986"
   - For Transformers: "Introduced in 'Attention is All You Need' paper, 2017"
   - For RL: "Used in AlphaGo to beat world champion in 2016"
   - For CV: "Achieving superhuman performance in medical imaging"
   - For NLP: "GPT and BERT revolutionized the field in 2018-2019"

4. Search for all concepts related to "Neural Networks"

5. Read the complete graph and summarize the knowledge structure you've built.

Execute each step and provide a final summary of the knowledge graph.
""",
        "expected_tool_calls": 15,
    },
    {
        "name": "Project Management Simulation",
        "description": "Simulate managing a project with dependencies",
        "prompt": """
You have a knowledge graph memory. Simulate setting up a project management structure:

1. Create these project milestone entities:
   - "Requirements Gathering" (status: completed, duration: 2_weeks)
   - "System Design" (status: in_progress, duration: 3_weeks)
   - "Backend Development" (status: pending, duration: 6_weeks)
   - "Frontend Development" (status: pending, duration: 5_weeks)
   - "Integration Testing" (status: pending, duration: 2_weeks)
   - "Deployment" (status: pending, duration: 1_week)

2. Create dependency relationships:
   - "System Design" depends_on "Requirements Gathering"
   - "Backend Development" depends_on "System Design"
   - "Frontend Development" depends_on "System Design"
   - "Integration Testing" depends_on both "Backend Development" AND "Frontend Development"
   - "Deployment" depends_on "Integration Testing"

3. Add risk observations:
   - "Backend Development": "High complexity - may need additional resources"
   - "Integration Testing": "Critical path - buffer time recommended"
   - "Deployment": "Requires production environment setup"

4. Search for all pending milestones

5. Read the full project graph

6. Based on the graph, identify which milestone is on the critical path

Provide a project status report based on the knowledge graph.
""",
        "expected_tool_calls": 18,
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
    error: str | None = None


# ---------- Main ----------

async def run_test(agent: Agent, test_case: dict) -> TestResult:
    """Run a single test case and measure performance."""
    name = test_case["name"]
    prompt = test_case["prompt"]
    
    print(f"\n{'='*60}")
    print(f"🧪 Running Test: {name}")
    print(f"   {test_case['description']}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        response = await agent.run(prompt)
        duration = time.time() - start_time
        
        print(f"\n📝 Agent Response:\n{response}\n")
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"📊 Response length: {len(response)} chars")
        
        return TestResult(
            name=name,
            success=True,
            duration_seconds=duration,
            response_length=len(response),
        )
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n❌ Test failed with error: {e}")
        return TestResult(
            name=name,
            success=False,
            duration_seconds=duration,
            response_length=0,
            error=str(e),
        )


async def main():
    parser = argparse.ArgumentParser(description="Long Task Performance Test")
    parser.add_argument("provider", nargs="?", default="ollama", 
                        help="LLM Provider (openai, anthropic, google, ollama)")
    parser.add_argument("--mcp", required=True,
                        help="Command to run MCP server")
    parser.add_argument("--test", type=int, default=None,
                        help="Run specific test (0, 1, 2) or all if not specified")
    
    args = parser.parse_args()
    
    provider = make_provider(args.provider)
    
    # Setup MCP
    from openagent.mcp import McpClient
    
    cmd_parts = shlex.split(args.mcp)
    command = cmd_parts[0]
    arguments = cmd_parts[1:]
    
    print(f"🔌 Connecting to MCP server: {command} {arguments}")
    
    async with McpClient(command, arguments) as mcp_client:
        mcp_tools = await mcp_client.get_tools()
        print(f"✅ Found {len(mcp_tools)} MCP tools: {[t._tool_name for t in mcp_tools]}")
        
        agent = Agent(
            provider=provider,
            system_prompt="""You are a helpful assistant with access to a knowledge graph memory system.
Use the available tools to complete tasks step by step. 
Always verify your work by reading the graph after making changes.
Be thorough and complete all steps requested.""",
            tools=mcp_tools,
        )
        
        # Select tests to run
        if args.test is not None:
            tests = [LONG_TASK_PROMPTS[args.test]]
        else:
            tests = LONG_TASK_PROMPTS
        
        results: list[TestResult] = []
        
        for test_case in tests:
            result = await run_test(agent, test_case)
            results.append(result)
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")
        
        total_duration = sum(r.duration_seconds for r in results)
        successful = sum(1 for r in results if r.success)
        
        for r in results:
            status = "✅ PASS" if r.success else "❌ FAIL"
            print(f"  {status} {r.name}: {r.duration_seconds:.2f}s, {r.response_length} chars")
            if r.error:
                print(f"       Error: {r.error}")
        
        print(f"\n  Total: {successful}/{len(results)} passed")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
