"""Token estimation utilities for context window management."""

from __future__ import annotations

import re
from typing import Final

from .types import ContentBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock

# --- Provider-agnostic token estimates ---

_TOKEN_COST_BY_CHAR_CLASS: Final = 0.25
_CJK_COST: Final = 0.15
_CODE_COST: Final = 0.20

# Rough per-message overhead (role prefix, structure metadata)
_MSG_OVERHEAD: Final = 4

# Per-tool-call overhead
_TOOL_CALL_OVERHEAD: Final = 12

# Per-tool-result overhead
_TOOL_RESULT_OVERHEAD: Final = 8

# Unicode ranges for CJK detection
_CJK_PATTERN = re.compile(
    r'[一-鿿぀-ゟ゠-ヿ'
    r'가-힯ࠀ-࠿ༀ-࿿]'
)

# Simple heuristic to detect code-heavy text
_CODE_PATTERN = re.compile(
    r'(def |class |import |async |await |return |function |const |let |var |'
    r'if |else |for |while |try |catch |throw |new |this |self\.|'
    r'[{}();=<>|\-&*+/!%])',
)


def _is_code(text: str) -> bool:
    """Rough heuristic: if the text has many code keywords/symbols, treat it as code."""
    if not text:
        return False
    hits = len(_CODE_PATTERN.findall(text))
    return hits > max(2, len(text) / 200)


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a string.

    Uses a character-based heuristic with different rates for
    Latin/English text, CJK text, and code.
    """
    if not text:
        return 0

    # If tiktoken is available and text looks like English/Latin, use it
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        pass

    cjk_chars = len(_CJK_PATTERN.findall(text))
    non_cjk = len(text) - cjk_chars

    if _is_code(text):
        code_tokens = non_cjk * _CODE_COST
    else:
        code_tokens = non_cjk * _TOKEN_COST_BY_CHAR_CLASS

    return max(1, int(code_tokens + cjk_chars * _CJK_COST))


def get_context_window(model: str) -> int:
    """Return the approximate context window for a known model name.

    For unrecognized models returns 128_000 as a safe default.
    """
    m = model.lower()

    # OpenAI
    if any(x in m for x in ("gpt-3.5", "gpt3")):
        return 16_000
    if any(x in m for x in ("gpt-4-", "gpt-4o", "gpt-4-turbo")):
        return 128_000
    if "gpt-4" in m:
        return 8_000

    # Anthropic
    if any(x in m for x in ("claude-3", "claude-4", "sonnet-4", "opus-4")):
        return 200_000
    if "claude-3" in m:
        return 200_000
    if "claude" in m:
        return 100_000

    # Google
    if "gemini" in m:
        return 2_000_000 if "flash" in m else 1_000_000

    # Ollama / generic long-context
    if any(x in m for x in ("llama3", "llama-3")):
        return 8_000 if "instruct" not in m else 128_000

    return 128_000


def estimate_message_tokens(msg: Message) -> int:
    """Estimate the token cost of a single message including structural overhead."""
    if isinstance(msg.content, str):
        return _MSG_OVERHEAD + estimate_tokens(msg.content)

    total = _MSG_OVERHEAD
    for block in msg.content:
        if isinstance(block, TextBlock):
            total += estimate_tokens(block.text)
        elif isinstance(block, ToolUseBlock):
            total += _TOOL_CALL_OVERHEAD + estimate_tokens(block.name)
            # arguments are usually JSON; count them at code rate
            total += estimate_tokens(str(block.arguments))
        elif isinstance(block, ToolResultBlock):
            total += _TOOL_RESULT_OVERHEAD + estimate_tokens(block.content)

    return total


def estimate_conversation_tokens(messages: list[Message]) -> int:
    """Estimate the total tokens for a list of messages."""
    return sum(estimate_message_tokens(m) for m in messages)
