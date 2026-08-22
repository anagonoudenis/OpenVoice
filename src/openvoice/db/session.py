"""Async SQLAlchemy engine and session management.

The engine/sessionmaker are created lazily and cached at module level so
that importing this module never opens a connection, and tests can force
a rebuild (e.g. against a different `DATABASE_URL`) via `reset_engine()`.
"""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from openvoice.config import Settings, get_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def build_engine(settings: Settings) -> AsyncEngine:
    """Create a new async engine from settings. Does not open a connection."""
    return create_async_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        echo=settings.database_echo,
        pool_pre_ping=True,
    )


def get_engine() -> AsyncEngine:
    """Return the process-wide cached engine, building it on first use."""
    global _engine
    if _engine is None:
        _engine = build_engine(get_settings())
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide cached session factory."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def reset_engine() -> None:
    """Dispose of the cached engine/sessionmaker. Used by tests and shutdown hooks."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a transactional session per request."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        yield session
