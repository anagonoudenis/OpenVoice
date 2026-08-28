"""Shared types for wiring LLM tool-calls into real actions."""

from collections.abc import Awaitable, Callable

from openvoice.llm.base import ToolCall

# (result content, is_error) -- fed back to the model as a TOOL-role message.
ToolExecutionResult = tuple[str, bool]

# Executes one tool call and returns its result. Must never raise: any
# failure (validation, provider error, unknown tool) is reported as
# `(message, is_error=True)` so the model can see what went wrong and
# decide how to recover, rather than the whole turn crashing.
ToolExecutor = Callable[[ToolCall], Awaitable[ToolExecutionResult]]
