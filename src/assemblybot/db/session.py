from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from assemblybot.db.base import Base


def build_sqlite_url(db_path: str | Path) -> str:
    """Build a SQLAlchemy URL for a local SQLite database file."""
    return f"sqlite:///{Path(db_path).expanduser().resolve()}"


def create_sqlite_engine(
    db_path: str | Path,
    *,
    echo: bool = False,
) -> Engine:
    """Create an engine for the AssemblyBot SQLite database."""
    return create_engine(build_sqlite_url(db_path), echo=echo, future=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a typed SQLAlchemy session factory."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def create_all_tables(engine: Engine) -> None:
    """Create all currently registered AssemblyBot database tables."""
    import assemblybot.db.schema.chunk  # noqa: F401
    import assemblybot.db.schema.diarization  # noqa: F401
    import assemblybot.db.schema.person  # noqa: F401
    import assemblybot.db.schema.pipeline_run  # noqa: F401
    import assemblybot.db.schema.session  # noqa: F401
    import assemblybot.db.schema.speaker  # noqa: F401
    import assemblybot.db.schema.transcript  # noqa: F401
    import assemblybot.db.schema.turn  # noqa: F401

    Base.metadata.create_all(engine)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a transactional scope around a series of database operations."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
