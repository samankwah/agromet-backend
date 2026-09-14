"""Isolates every test run from the developer's real database.

Every test module does `from backend.app import main` (or imports something
that does), and that import resolves `DATABASE_PATH`/`DATABASE_URL` and runs
`init_db()` + `seed_market_data()` as a side effect of import. Without this
file, that resolves to `backend/agromet.db` -- the same file a locally
running `uvicorn` writes real dev data to -- so running the suite would seed
and mutate a developer's own database. pytest imports the `conftest.py`
nearest a test file before collecting it, so setting the environment here
runs before any test module's import does.

Both overrides are unconditional, not `setdefault`: a developer's shell may
already export a real `DATABASE_URL` for running the app locally against
Postgres, and inheriting that into the test process would point destructive
INSERT/UPDATE/DELETE assertions at a real database. The suite decides its
own database; it does not trust the ambient environment for one.

Set `TEST_DATABASE_URL` to run this exact suite against Postgres instead of
SQLite -- the two dialects' parity proven by real assertions, not just
review. `docker run -e POSTGRES_PASSWORD=test -p 5432:5432 postgres:16-alpine`
plus `TEST_DATABASE_URL=postgresql://postgres:test@localhost:5432/postgres`
is enough to try it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_DB_DIR = tempfile.mkdtemp(prefix="agromet-test-db-")

os.environ["DATABASE_PATH"] = str(Path(_TEST_DB_DIR) / "test.db")
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", "")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SECRET_KEY", "test-secret")
