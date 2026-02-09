from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn

mcp = FastMCP("demo")

@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    return a + b

# FastMCP doesn't directly expose an easy way to run SSE programmatically without CLI args sometimes.
# Let's use the underlying Starlette app if possible, or just instruct the user.
# Actually, the user's issue was "ConnectError", meaning nothing was listening.
# We will create a script that DEFINITELY runs an SSE server.

if __name__ == "__main__":
    print("Starting SSE server on http://localhost:8000/sse")
    mcp.run(transport="sse")
