"""calls.livekit_room_name is no longer unique

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-23

`Call.id` is the real identifier for a call; a LiveKit room name being
reused is legitimate (every `console`-mode local test session reuses the
fixed room name "console"), so the unique constraint added in 0001
rejected the second-ever local test call with an IntegrityError. See
CHANGELOG and `openvoice.db.models.call`.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("calls_livekit_room_name_key", "calls", type_="unique")
    op.create_index("ix_calls_livekit_room_name", "calls", ["livekit_room_name"])


def downgrade() -> None:
    op.drop_index("ix_calls_livekit_room_name", table_name="calls")
    op.create_unique_constraint("calls_livekit_room_name_key", "calls", ["livekit_room_name"])
