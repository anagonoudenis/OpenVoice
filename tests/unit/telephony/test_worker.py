"""Tests for `openvoice.telephony.worker._transfer_to_human`/`_caller_phone_number`.

The `entrypoint` function itself is integration-glue that needs a live
`JobContext`/`AgentSession`/room (see the module docstring) and isn't
exercised here; `_transfer_to_human` and `_caller_phone_number` are pure
enough (given a mocked `LiveKitAPI`/a fake room) to unit test properly.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import structlog

from openvoice.telephony import worker


def _log() -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(__name__).bind(call_id="test")


async def test_transfer_to_human_does_nothing_without_a_configured_number() -> None:
    await worker._transfer_to_human(room_name="room-1", transfer_number=None, log=_log())
    # No exception, no LiveKitAPI call attempted -- nothing to assert on a mock here,
    # the absence of a raised error/timeout is the behavior under test.


async def test_transfer_to_human_calls_sip_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_api = SimpleNamespace(
        room=SimpleNamespace(
            list_rooms=AsyncMock(return_value=SimpleNamespace(rooms=[SimpleNamespace()])),
            list_participants=AsyncMock(
                return_value=SimpleNamespace(
                    participants=[
                        SimpleNamespace(kind=worker.api.ParticipantInfo.Kind.SIP, identity="sip-1")
                    ]
                )
            ),
        ),
        sip=SimpleNamespace(transfer_sip_participant=AsyncMock()),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(worker.api, "LiveKitAPI", lambda: fake_api)

    await worker._transfer_to_human(room_name="room-1", transfer_number="+15551234567", log=_log())

    fake_api.sip.transfer_sip_participant.assert_awaited_once()
    request = fake_api.sip.transfer_sip_participant.call_args.args[0]
    assert request.room_name == "room-1"
    assert request.participant_identity == "sip-1"
    assert request.transfer_to == "tel:+15551234567"
    fake_api.aclose.assert_awaited_once()


async def test_transfer_to_human_warns_when_room_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_api = SimpleNamespace(
        room=SimpleNamespace(list_rooms=AsyncMock(return_value=SimpleNamespace(rooms=[]))),
        sip=SimpleNamespace(transfer_sip_participant=AsyncMock()),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(worker.api, "LiveKitAPI", lambda: fake_api)

    await worker._transfer_to_human(
        room_name="missing-room", transfer_number="+15551234567", log=_log()
    )

    fake_api.sip.transfer_sip_participant.assert_not_awaited()
    fake_api.aclose.assert_awaited_once()


async def test_transfer_to_human_warns_when_no_sip_participant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_api = SimpleNamespace(
        room=SimpleNamespace(
            list_rooms=AsyncMock(return_value=SimpleNamespace(rooms=[SimpleNamespace()])),
            list_participants=AsyncMock(
                return_value=SimpleNamespace(
                    participants=[SimpleNamespace(kind=worker.api.ParticipantInfo.Kind.STANDARD)]
                )
            ),
        ),
        sip=SimpleNamespace(transfer_sip_participant=AsyncMock()),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(worker.api, "LiveKitAPI", lambda: fake_api)

    await worker._transfer_to_human(room_name="room-1", transfer_number="+15551234567", log=_log())

    fake_api.sip.transfer_sip_participant.assert_not_awaited()


async def test_transfer_to_human_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise RuntimeError("livekit api down")

    monkeypatch.setattr(worker.api, "LiveKitAPI", _raise)

    # Must not raise: a failed transfer must never crash the call.
    await worker._transfer_to_human(room_name="room-1", transfer_number="+15551234567", log=_log())


def _ctx_with_participants(*attributes_list: dict[str, str]) -> SimpleNamespace:
    participants = {
        str(i): SimpleNamespace(attributes=attrs) for i, attrs in enumerate(attributes_list)
    }
    return SimpleNamespace(room=SimpleNamespace(remote_participants=participants))


def test_caller_phone_number_reads_sip_attribute() -> None:
    ctx = _ctx_with_participants({"sip.phoneNumber": "+15551230000"})
    assert worker._caller_phone_number(ctx) == "+15551230000"  # type: ignore[arg-type]


def test_caller_phone_number_returns_none_when_absent() -> None:
    ctx = _ctx_with_participants({"other.attr": "x"})
    assert worker._caller_phone_number(ctx) is None  # type: ignore[arg-type]


def test_caller_phone_number_returns_none_for_no_participants() -> None:
    ctx = _ctx_with_participants()
    assert worker._caller_phone_number(ctx) is None  # type: ignore[arg-type]
