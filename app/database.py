"""Two backends, one call-site dialect.

SQLite is the zero-config default -- every fresh checkout and the test suite
get a working database with no setup. It is wrong for exactly one thing:
production. This backend deploys to Vercel, whose filesystem is read-only
outside `/tmp`, and `/tmp` is wiped on every cold start and not shared
between concurrent instances. A SQLite file there is not a database, it is a
scratchpad that resets itself -- every signup, uploaded calendar, diagnosis
record and contact message would be gone the moment that instance recycled,
silently, with nothing in the logs to say so.

`DATABASE_URL` (a standard `postgresql://...` connection string, e.g. from
Neon, Supabase, or Vercel Postgres -- use the *pooled* connection string
those providers publish, not the direct one, since a serverless function can
spin up many concurrent instances) switches the whole app to Postgres. The
64+ call sites elsewhere in this codebase were written once, against
sqlite3.Connection's `?`-placeholder, single-shot `.execute()` API, and
nothing about that has to change: `_PostgresConnection` below adapts a
pooled psycopg connection to speak the same dialect -- `?` becomes `%s`, and
an INSERT grows a `RETURNING id` clause so `.lastrowid` keeps working -- so
the dialect difference lives here, once, instead of at every call site.

The one behavior change: SQLite was never asked to enforce foreign keys
(`PRAGMA foreign_keys` defaults off), so every `ON DELETE CASCADE` in the
schema below has been silently inert -- deleting a calendar has always left
its activities behind as orphan rows. Postgres enforces foreign keys
unconditionally. Rather than let dev (SQLite) and prod (Postgres) disagree
about something as basic as "does deleting a calendar delete its activities",
`_connect()` now turns the pragma on, so SQLite finally does what the schema
already said it would.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "agromet.db"

# Set once, at startup, from `DATABASE_URL`. Empty means "use the SQLite
# file at DATABASE_PATH" -- the zero-config default every local checkout and
# the test suite gets. A Postgres URL is what makes writes survive a
# serverless cold start; see the module docstring below.
DATABASE_URL: str = ""

_pool: ConnectionPool | None = None


def set_database_path(path: str | Path) -> None:
    global DATABASE_PATH
    DATABASE_PATH = Path(path)


def set_database_url(url: str | None) -> None:
    """Point the database layer at Postgres, or back at SQLite if `url` is
    falsy. Also drops any pool from a previous URL (tests swap this)."""
    global DATABASE_URL, _pool
    DATABASE_URL = url or ""
    if _pool is not None:
        _pool.close()
        _pool = None


def _is_postgres() -> bool:
    return bool(DATABASE_URL)


def _connect() -> sqlite3.Connection | "_PostgresConnection":
    if _is_postgres():
        pool = _get_pool()
        return _PostgresConnection(pool.getconn(), pool)

    connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        # Small on purpose: a serverless function handles a handful of
        # concurrent requests per instance, and free-tier Postgres plans cap
        # total connections in the tens, shared across every warm instance
        # the platform happens to have running.
        min_size = int(os.getenv("DATABASE_POOL_MIN", "0"))
        max_size = int(os.getenv("DATABASE_POOL_MAX", "5"))
        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row, "autocommit": False},
            open=False,
        )
        _pool.open(wait=True, timeout=10)
    return _pool


def close_all_connections() -> None:
    """Releases the Postgres pool. A no-op on SQLite, which has no pool to
    leak. Called from the app's shutdown hook so a local `uvicorn --reload`
    or the test suite doesn't accumulate one pool per reload."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


class _PostgresCursor:
    """`sqlite3.Cursor` has a `.lastrowid` no Postgres driver provides, since
    there is no server-side concept of "the last row this cursor inserted" --
    only `RETURNING`. `_PostgresConnection.execute` appends that clause to a
    bare INSERT and reads the id back into this attribute, so the 15 call
    sites in main.py that read `cursor.lastrowid` do not need to know their
    connection is no longer SQLite.
    """

    def __init__(self, cursor: psycopg.Cursor, lastrowid: int | None) -> None:
        self._cursor = cursor
        self.lastrowid = lastrowid

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._cursor)


def _translate_statement(sql: str) -> tuple[str, bool]:
    """`?` -> `%s`, skipping any `?` inside a single-quoted string literal
    (SQL escapes a literal quote by doubling it -- `''` never toggles
    string state). Also reports whether this is a bare INSERT that wants a
    `RETURNING id` clause appended, for `.lastrowid` -- added at the very
    end, so it is valid whether or not the statement already carries an
    `ON CONFLICT ... DO UPDATE` (as the market-data upserts do).
    """
    out: list[str] = []
    in_string = False
    i = 0
    length = len(sql)
    while i < length:
        char = sql[i]
        if char == "'":
            if in_string and sql[i : i + 2] == "''":
                out.append("''")
                i += 2
                continue
            in_string = not in_string
            out.append(char)
        elif char == "?" and not in_string:
            out.append("%s")
        else:
            out.append(char)
        i += 1

    translated = "".join(out)
    stripped_upper = translated.strip().upper()
    wants_id = stripped_upper.startswith("INSERT") and "RETURNING" not in stripped_upper
    if wants_id:
        translated = translated.rstrip()
        if translated.endswith(";"):
            translated = translated[:-1]
        translated += "\nRETURNING id"

    return translated, wants_id


def _split_statements(script: str) -> list[str]:
    """`executescript`'s Postgres path. Splitting on bare `;` is safe here
    specifically because it is only ever fed this module's own DDL, whose
    string-literal defaults (`DEFAULT 'active'`, `DEFAULT '[]'`, ...) are all
    semicolon-free. Not a general-purpose SQL splitter, and not used as one.
    """
    return [statement.strip() for statement in script.split(";") if statement.strip()]


class _PostgresConnection:
    """Adapts a pooled psycopg connection to sqlite3.Connection's shape --
    `.execute()` returning a cursor directly, `.executescript()`,
    `.commit()`/`.rollback()`/`.close()` -- so `get_connection()` below can
    hand either one to a caller that only ever knew sqlite3's API.
    """

    def __init__(self, raw: psycopg.Connection, pool: ConnectionPool) -> None:
        self._raw = raw
        self._pool = pool

    def execute(self, sql: str, params: Iterable[Any] = ()) -> _PostgresCursor:
        translated, wants_id = _translate_statement(sql)
        cursor = self._raw.cursor()
        cursor.execute(translated, tuple(params))
        lastrowid = None
        if wants_id:
            row = cursor.fetchone()
            lastrowid = row["id"] if row else None
        return _PostgresCursor(cursor, lastrowid)

    def executescript(self, script: str) -> None:
        for statement in _split_statements(script):
            self._raw.execute(statement)

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        # Returning the connection to the pool *is* closing it, from the
        # caller's point of view -- the pool decides whether to actually
        # tear down the socket or keep it warm for the next request.
        self._pool.putconn(self._raw)


@contextmanager
def get_connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _add_column_if_missing(connection, table: str, column: str, definition: str) -> None:
    """SQLite has no ADD COLUMN IF NOT EXISTS, and CREATE TABLE IF NOT EXISTS
    silently skips an existing table -- so a column added to the schema above
    never reaches a database that already exists. This closes that gap.
    Handles both dialects: SQLite's `PRAGMA table_info`, Postgres's
    `information_schema.columns`."""
    if _is_postgres():
        existing = {
            row["column_name"]
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (table,),
            ).fetchall()
        }
    else:
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}

    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# The schema, written once against SQLite's dialect. `_postgres_ddl` derives
# the Postgres version from this same string rather than maintaining a second
# copy by hand -- every table here declares its primary key the same way
# (`id INTEGER PRIMARY KEY AUTOINCREMENT`, checked by
# tests/test_database_dialects.py), so a single substitution is both correct
# and impossible for the two schemas to drift apart on.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    name TEXT,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agricultural_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id INTEGER,
    data_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    file_name TEXT,
    file_content_type TEXT,
    file_size INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(owner_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_agricultural_records_type
ON agricultural_records(data_type);

CREATE TABLE IF NOT EXISTS calendars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_record_id INTEGER,
    title TEXT NOT NULL,
    description TEXT,
    calendar_type TEXT NOT NULL,
    crop TEXT NOT NULL,
    region_code TEXT,
    region TEXT,
    district_code TEXT,
    district TEXT,
    year INTEGER,
    total_weeks INTEGER,
    cycle_duration INTEGER,
    breed_type TEXT,
    sample_activities_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_record_id) REFERENCES agricultural_records(id)
);

CREATE INDEX IF NOT EXISTS idx_calendars_type
ON calendars(calendar_type);

CREATE INDEX IF NOT EXISTS idx_calendars_location
ON calendars(region_code, district_code, crop, year);

CREATE TABLE IF NOT EXISTS calendar_activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calendar_id INTEGER NOT NULL,
    activity_code TEXT NOT NULL,
    activity_name TEXT NOT NULL,
    start_week INTEGER NOT NULL,
    end_week INTEGER NOT NULL,
    production_week INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(calendar_id) REFERENCES calendars(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_calendar_activities_calendar
ON calendar_activities(calendar_id, start_week, end_week);

CREATE TABLE IF NOT EXISTS weekly_advisories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_record_id INTEGER,
    advisory_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    region_code TEXT,
    region TEXT,
    district_code TEXT,
    district TEXT,
    crop TEXT,
    commodity_code TEXT,
    poultry_type_code TEXT,
    breed_code TEXT,
    year INTEGER,
    season TEXT,
    summary TEXT,
    weather_forecast_json TEXT NOT NULL DEFAULT '{}',
    advisories_json TEXT NOT NULL DEFAULT '[]',
    sms_advisory TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_record_id) REFERENCES agricultural_records(id)
);

CREATE INDEX IF NOT EXISTS idx_weekly_advisories_lookup
ON weekly_advisories(advisory_type, region_code, district_code, crop, year);

CREATE TABLE IF NOT EXISTS weekly_advisory_activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    advisory_id INTEGER NOT NULL,
    activity TEXT NOT NULL,
    week_label TEXT,
    activity_type TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(advisory_id) REFERENCES weekly_advisories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_weekly_advisory_activities_advisory
ON weekly_advisory_activities(advisory_id);

CREATE TABLE IF NOT EXISTS production_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calendar_id INTEGER NOT NULL,
    batch_name TEXT NOT NULL,
    commodity TEXT NOT NULL,
    start_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    initial_quantity INTEGER NOT NULL DEFAULT 0,
    current_quantity INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    -- Copied from the calendar at creation time, because
    -- serialize_cycle divides by it to get currentWeek and
    -- progressPercent. Without the column every cycle reported
    -- "week 1 of 1, 100%" regardless of its calendar's length.
    -- Snapshotted rather than joined so a later edit to the
    -- calendar cannot silently move a running batch's finish
    -- date.
    total_duration_weeks INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(calendar_id) REFERENCES calendars(id)
);

CREATE INDEX IF NOT EXISTS idx_production_cycles_status
ON production_cycles(status, commodity);

CREATE TABLE IF NOT EXISTS diagnosis_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id INTEGER,
    provider_product TEXT,
    status TEXT NOT NULL,
    plant TEXT,
    disease TEXT,
    confidence REAL,
    severity TEXT,
    source TEXT,
    crop_context TEXT,
    region_context TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(owner_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_diagnosis_records_owner
ON diagnosis_records(owner_id, created_at DESC);

CREATE TABLE IF NOT EXISTS commodities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    unit TEXT NOT NULL,
    trend TEXT NOT NULL DEFAULT 'stable',
    demand TEXT NOT NULL DEFAULT 'moderate',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_commodities_slug
ON commodities(slug);

CREATE INDEX IF NOT EXISTS idx_commodities_category
ON commodities(category);

CREATE TABLE IF NOT EXISTS commodity_trends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity_slug TEXT NOT NULL UNIQUE,
    month_prices_json TEXT NOT NULL DEFAULT '[]',
    seasonal_pattern TEXT,
    peak_months_json TEXT NOT NULL DEFAULT '[]',
    low_months_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(commodity_slug) REFERENCES commodities(slug)
);

CREATE INDEX IF NOT EXISTS idx_commodity_trends_slug
ON commodity_trends(commodity_slug);

CREATE TABLE IF NOT EXISTS market_centers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region TEXT NOT NULL UNIQUE,
    major_markets_json TEXT NOT NULL DEFAULT '[]',
    transport_access TEXT NOT NULL DEFAULT 'fair',
    price_premium REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_market_centers_region
ON market_centers(region);

-- Published GMet flood/drought bulletins. These override the
-- computed index for a region, because local knowledge (dam
-- spillage, a failed drain, an evacuation order) is not visible to
-- a global hydrological model. The computed value is still shown
-- alongside so an override is never silent.
CREATE TABLE IF NOT EXISTS hazard_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region TEXT NOT NULL,
    hazard TEXT NOT NULL CHECK (hazard IN ('flood', 'drought')),
    band TEXT NOT NULL,
    headline TEXT,
    advisory_json TEXT NOT NULL DEFAULT '[]',
    issued_by TEXT NOT NULL DEFAULT 'GMet',
    effective_from TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    effective_to TEXT,
    created_by INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(created_by) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_hazard_overrides_lookup
ON hazard_overrides(region, hazard, effective_from);

-- Messages sent from the apps' Contact screen.
--
-- Stored rather than emailed straight out: an SMTP failure would
-- lose a farmer's message with nothing to recover it from, and a
-- table can be read by whoever is on duty regardless of whose
-- mailbox is configured. `handled_at` is the whole workflow --
-- unset means nobody has answered it yet.
CREATE TABLE IF NOT EXISTS contact_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    subject TEXT NOT NULL,
    message TEXT NOT NULL,
    -- Which app it came from, so a bug report can be reproduced.
    source TEXT NOT NULL DEFAULT 'mobile',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    handled_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_contact_messages_unhandled
ON contact_messages(handled_at, created_at);
"""


def _postgres_ddl(sqlite_ddl: str) -> str:
    return sqlite_ddl.replace(
        "INTEGER PRIMARY KEY AUTOINCREMENT",
        "INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY",
    )


def init_db() -> None:
    if not _is_postgres():
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        connection.executescript(_postgres_ddl(_SCHEMA) if _is_postgres() else _SCHEMA)

        # Migrations for databases created before a column was added.
        _add_column_if_missing(connection, "production_cycles", "total_duration_weeks", "INTEGER")


def row_to_dict(row: sqlite3.Row | dict | None) -> dict | None:
    return dict(row) if row is not None else None


def encode_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True)


def decode_payload(value: str | None) -> dict:
    if not value:
        return {}
    return json.loads(value)
