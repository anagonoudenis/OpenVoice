"""Fixtures for integration tests that need a real PostgreSQL database.

These tests talk to an actual Postgres instance (via `docker compose up
postgres`, or the `postgres` service container in CI) rather than mocking
the database, because the ORM constraints (unique, check, cascade) are
themselves part of what's under test.
"""

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from openvoice.db.models import Base

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://openvoice:openvoice@localhost:5432/openvoice_test",
)
# Deliberately a *different* database name than the app's real dev DB
# (`openvoice`), not just a different env var: `db_engine` below runs
# `Base.metadata.drop_all` after every test. Pointing this at the same
# database as real local development -- e.g. by exporting
# `TEST_DATABASE_URL` to "convenient" defaults -- silently wipes real
# dev/demo data (this happened once while testing this project for real;
# see CHANGELOG). Create it once with, e.g.:
#   psql -U postgres -c "CREATE DATABASE openvoice_test OWNER openvoice;"


@pytest.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    """Function-scoped, not session-scoped: a session-scoped async engine
    would need its own `loop_scope="session"` (pytest-asyncio) to share an
    event loop with function-scoped test coroutines, and asyncpg
    connections can't cross event loops -- rather than take on that
    cross-loop complexity, each test gets a fresh engine. Slightly more
    setup/teardown per test, but correct and simple.
    """
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    sessionmaker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with sessionmaker() as session:
        yield session
    # Reset state between tests without paying for a full drop/recreate.
    async with db_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())
