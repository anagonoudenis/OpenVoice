"""Conversational agent core: intent detection, context, and fallback behavior.

Deliberately decoupled from LiveKit: everything here is testable against
mocked `BaseLLMProvider` instances. `openvoice.telephony` wires this into
the real-time call pipeline.
"""
