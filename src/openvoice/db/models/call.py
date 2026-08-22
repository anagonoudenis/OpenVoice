"""Call model: one phone call handled by the agent."""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from openvoice.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from openvoice.db.models.appointment import Appointment
    from openvoice.db.models.client import Client
    from openvoice.db.models.transcript import CallTranscript


class CallDirection(StrEnum):
    """Whether the agent received or placed the call."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CallStatus(StrEnum):
    """Lifecycle state of a call."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TRANSFERRED_TO_HUMAN = "transferred_to_human"
    FAILED = "failed"
    NO_ANSWER = "no_answer"


class Call(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single phone call. `client_id` is nullable: it's set once the caller is resolved."""

    __tablename__ = "calls"

    client_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), index=True
    )
    livekit_room_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    direction: Mapped[CallDirection] = mapped_column(
        Enum(CallDirection, native_enum=False, length=16), nullable=False
    )
    status: Mapped[CallStatus] = mapped_column(
        Enum(CallStatus, native_enum=False, length=32),
        nullable=False,
        default=CallStatus.IN_PROGRESS,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[str | None] = mapped_column(Text)

    client: Mapped["Client | None"] = relationship(back_populates="calls")
    transcripts: Mapped[list["CallTranscript"]] = relationship(
        back_populates="call",
        cascade="all, delete-orphan",
        order_by="CallTranscript.sequence",
    )
    appointments: Mapped[list["Appointment"]] = relationship(back_populates="call")

    def __repr__(self) -> str:
        return f"Call(id={self.id!r}, status={self.status!r})"
