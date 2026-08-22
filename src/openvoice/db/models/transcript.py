"""CallTranscript model: one utterance in a call, in speaking order."""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from openvoice.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from openvoice.db.models.call import Call


class SpeakerRole(StrEnum):
    """Who produced a transcript line."""

    CALLER = "caller"
    AGENT = "agent"
    SYSTEM = "system"


class CallTranscript(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single turn of a call's transcript, ordered by `sequence` within the call."""

    __tablename__ = "call_transcripts"
    __table_args__ = (UniqueConstraint("call_id", "sequence", name="uq_call_transcript_sequence"),)

    call_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("calls.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[SpeakerRole] = mapped_column(
        Enum(SpeakerRole, native_enum=False, length=16), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    spoken_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    call: Mapped["Call"] = relationship(back_populates="transcripts")

    def __repr__(self) -> str:
        return (
            f"CallTranscript(call_id={self.call_id!r}, "
            f"sequence={self.sequence!r}, role={self.role!r})"
        )
