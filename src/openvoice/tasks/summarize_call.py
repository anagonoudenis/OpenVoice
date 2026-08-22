"""Post-call summary generation, run asynchronously via Celery so it never
delays call teardown or blocks the telephony worker's event loop.

`summarize_call` is the testable core (dependencies injected); the Celery
task wraps it with the real settings-derived DB session/LLM provider and
`asyncio.run`, since Celery tasks are synchronous entry points.
"""

import asyncio
import uuid

import structlog
from celery import Task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from openvoice.config import get_settings
from openvoice.db.models import Call
from openvoice.db.session import get_sessionmaker
from openvoice.llm.base import BaseLLMProvider, LLMMessage, LLMMessageRole, LLMProviderError
from openvoice.llm.factory import get_llm_provider
from openvoice.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)

_SUMMARY_SYSTEM_PROMPT = (
    "Summarize the following phone call transcript in 2-3 sentences for a "
    "staff member who wasn't on the call. Focus on what the caller wanted "
    "and what was resolved or still needs follow-up."
)


async def summarize_call(
    *, db_session: AsyncSession, llm: BaseLLMProvider, call_id: uuid.UUID
) -> None:
    """Generate and persist a post-call summary for `call_id`.

    A no-op (logged, not raised) if the call doesn't exist or has no
    transcript yet -- there's nothing to summarize.
    """
    result = await db_session.execute(
        select(Call).where(Call.id == call_id).options(selectinload(Call.transcripts))
    )
    call = result.scalar_one_or_none()
    if call is None:
        logger.warning("summarize_call_not_found", call_id=str(call_id))
        return

    transcript_text = "\n".join(f"{t.role.value}: {t.text}" for t in call.transcripts)
    if not transcript_text:
        logger.info("summarize_call_empty_transcript", call_id=str(call_id))
        return

    response = await llm.generate(
        [LLMMessage(role=LLMMessageRole.USER, content=transcript_text)],
        system_prompt=_SUMMARY_SYSTEM_PROMPT,
        max_tokens=200,
    )

    call.summary = response.content
    await db_session.commit()
    logger.info("call_summarized", call_id=str(call_id))


async def _run_summarize_call(call_id: uuid.UUID) -> None:
    settings = get_settings()
    sessionmaker = get_sessionmaker()
    llm = get_llm_provider(settings)
    async with sessionmaker() as db_session:
        await summarize_call(db_session=db_session, llm=llm, call_id=call_id)


# celery ships no type stubs, so this decorator is untyped (Any) under mypy strict.
@celery_app.task(  # type: ignore[untyped-decorator]
    name="openvoice.summarize_call", bind=True, max_retries=3, default_retry_delay=30
)
def summarize_call_task(self: Task, call_id: str) -> None:
    """Celery entrypoint: generate and persist a post-call summary for `call_id`."""
    try:
        asyncio.run(_run_summarize_call(uuid.UUID(call_id)))
    except LLMProviderError as exc:
        logger.error("summarize_call_llm_failed", call_id=call_id, error=str(exc))
        raise self.retry(exc=exc) from exc
