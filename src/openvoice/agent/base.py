"""Shared types for the conversational agent core."""

from enum import StrEnum

from pydantic import BaseModel


class Intent(StrEnum):
    """What the caller's utterance is trying to accomplish."""

    BOOKING = "booking"
    SUPPORT_QUESTION = "support_question"
    URGENT = "urgent"
    HUMAN_TRANSFER = "human_transfer"
    GENERAL = "general"


class AgentReply(BaseModel):
    """The agent's response to one caller utterance."""

    text: str
    intent: Intent
    transfer_to_human: bool = False
