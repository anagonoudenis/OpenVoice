"""Client (caller) model."""

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from openvoice.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from openvoice.db.models.appointment import Appointment
    from openvoice.db.models.call import Call


class Client(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A caller, identified by phone number. Created automatically on first call."""

    __tablename__ = "clients"

    phone_number: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255))
    notes: Mapped[str | None] = mapped_column(Text)

    calls: Mapped[list["Call"]] = relationship(
        back_populates="client", cascade="all, delete-orphan"
    )
    appointments: Mapped[list["Appointment"]] = relationship(
        back_populates="client", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Client(id={self.id!r}, phone_number={self.phone_number!r})"
