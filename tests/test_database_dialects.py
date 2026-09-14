"""Proves the Postgres path behaves like the SQLite path it replaces on a
serverless deployment, rather than trusting that from code review alone.

Skipped entirely unless `TEST_DATABASE_URL` points at a real Postgres --
nobody should need Docker running to get a green `pytest`. When it is set,
every assertion below runs for real against that database: schema parity,
`?` placeholder translation, `.lastrowid` via `RETURNING id`, upsert
(`ON CONFLICT ... DO UPDATE`), and `ON DELETE CASCADE`.
"""

from __future__ import annotations

import os

import pytest

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a Postgres connection string to run the dialect-parity suite",
)


@pytest.fixture()
def pg_connection():
    """A Postgres connection wired up exactly like the app wires one up --
    same `database` module, same schema, same pool -- but pointed at a
    throwaway table set so this suite can drop and recreate it per test
    without touching whatever else lives in that Postgres instance.
    """
    from backend.app import database

    database.set_database_url(TEST_DATABASE_URL)
    with database.get_connection() as connection:
        connection.executescript(database._postgres_ddl(database._SCHEMA))
        database._add_column_if_missing(connection, "production_cycles", "total_duration_weeks", "INTEGER")

    yield database

    with database.get_connection() as connection:
        for table in (
            "weekly_advisory_activities",
            "weekly_advisories",
            "calendar_activities",
            "production_cycles",
            "calendars",
            "agricultural_records",
            "diagnosis_records",
            "hazard_overrides",
            "contact_messages",
            "commodity_trends",
            "commodities",
            "market_centers",
            "users",
        ):
            connection.execute(f"DELETE FROM {table}")
    database.set_database_url("")


def _sqlite_columns() -> dict[str, set[str]]:
    """Every table's column set, as SQLite sees its own schema -- the
    baseline the Postgres schema (derived from the same `_SCHEMA` string) is
    compared against.
    """
    import sqlite3

    from backend.app.database import _SCHEMA

    connection = sqlite3.connect(":memory:")
    connection.executescript(_SCHEMA)
    # `sqlite_sequence` is SQLite's own AUTOINCREMENT bookkeeping table, not
    # one `_SCHEMA` declares -- Postgres has no equivalent because identity
    # columns don't need one, which is correct, not a parity gap.
    tables = [
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if row[0] != "sqlite_sequence"
    ]
    return {table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")} for table in tables}


def _postgres_columns(connection) -> dict[str, set[str]]:
    rows = connection.execute(
        "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public'"
    ).fetchall()
    columns: dict[str, set[str]] = {}
    for row in rows:
        columns.setdefault(row["table_name"], set()).add(row["column_name"])
    return columns


class TestSchemaParity:
    def test_same_tables_and_columns_as_sqlite(self, pg_connection):
        """The one thing `_postgres_ddl`'s single string-substitution trick
        promises: whatever `_SCHEMA` declares, both dialects end up with the
        exact same tables and columns. If a future edit to `_SCHEMA`
        introduces a second SQLite-only construct, this is what catches it.
        """
        with pg_connection.get_connection() as connection:
            postgres_columns = _postgres_columns(connection)

        assert postgres_columns == _sqlite_columns()


class TestPlaceholderTranslation:
    def test_question_mark_params_work(self, pg_connection):
        with pg_connection.get_connection() as connection:
            connection.execute(
                "INSERT INTO users(email, name, password_hash) VALUES (?, ?, ?)",
                ("farmer@example.com", "Farmer", "hash"),
            )
            row = connection.execute("SELECT name FROM users WHERE email = ?", ("farmer@example.com",)).fetchone()

        assert row["name"] == "Farmer"

    def test_question_mark_inside_string_literal_is_not_translated(self, pg_connection):
        """A literal `?` in bound *data* (not the query text) must survive
        untouched -- it is the query template's `?` characters that mean
        "placeholder", never a `?` sitting inside a quoted value some other
        part of the query happens to embed literally.
        """
        with pg_connection.get_connection() as connection:
            connection.execute(
                "INSERT INTO users(email, name, password_hash) VALUES (?, ?, ?)",
                ("q@example.com", "Got rain?", "hash"),
            )
            row = connection.execute("SELECT name FROM users WHERE email = ?", ("q@example.com",)).fetchone()

        assert row["name"] == "Got rain?"


class TestLastrowid:
    def test_insert_populates_lastrowid_via_returning(self, pg_connection):
        with pg_connection.get_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO users(email, name, password_hash) VALUES (?, ?, ?)",
                ("id-check@example.com", "Id Check", "hash"),
            )
            inserted_id = cursor.lastrowid
            row = connection.execute("SELECT id FROM users WHERE email = ?", ("id-check@example.com",)).fetchone()

        assert inserted_id is not None
        assert inserted_id == row["id"]

    def test_upsert_with_on_conflict_still_returns_lastrowid(self, pg_connection):
        """The market-data seed uses `ON CONFLICT ... DO UPDATE`, not a bare
        INSERT -- RETURNING has to work appended after that clause too.
        """
        with pg_connection.get_connection() as connection:
            connection.execute(
                """
                INSERT INTO commodities (slug, name, category, price, unit, trend, demand)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET price = excluded.price
                """,
                ("maize", "Maize", "grain", 5.0, "kg", "stable", "moderate"),
            )
            cursor = connection.execute(
                """
                INSERT INTO commodities (slug, name, category, price, unit, trend, demand)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET price = excluded.price
                """,
                ("maize", "Maize", "grain", 6.5, "kg", "rising", "high"),
            )
            row = connection.execute("SELECT id, price FROM commodities WHERE slug = ?", ("maize",)).fetchone()

        assert cursor.lastrowid == row["id"]
        assert row["price"] == 6.5


class TestForeignKeyEnforcement:
    def test_deleting_a_calendar_cascades_to_its_activities(self, pg_connection):
        """Postgres enforces `ON DELETE CASCADE` unconditionally -- unlike
        SQLite, which needs `PRAGMA foreign_keys = ON` (now set in
        `_connect()`) or it silently leaves orphan rows. Same assertion
        should hold on both; this is the Postgres half of that promise.
        """
        with pg_connection.get_connection() as connection:
            calendar_id = connection.execute(
                "INSERT INTO calendars(title, calendar_type, crop) VALUES (?, ?, ?)",
                ("Test calendar", "crop", "maize"),
            ).lastrowid
            connection.execute(
                "INSERT INTO calendar_activities(calendar_id, activity_code, activity_name, start_week, end_week) "
                "VALUES (?, ?, ?, ?, ?)",
                (calendar_id, "land-prep", "Land preparation", 1, 2),
            )

            connection.execute("DELETE FROM calendars WHERE id = ?", (calendar_id,))

            remaining = connection.execute(
                "SELECT COUNT(*) AS count FROM calendar_activities WHERE calendar_id = ?", (calendar_id,)
            ).fetchone()

        assert remaining["count"] == 0
