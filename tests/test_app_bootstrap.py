"""Guards the one behavior no other test exercises: that importing
`backend.app.main` seeds market data into the database `DATABASE_PATH`
actually names, not wherever `database.py`'s own hardcoded default points.

This is a real bug the router split introduced and every other test missed.
`main.py` used to be one file, executed top to bottom, so "configure the
database, then seed it" was true by construction. Splitting `market.py`
out meant its module-level `seed_market_data()` call now runs at *import*
time -- and for one bad ordering, `from .routers import (...)` ran before
`set_database_path`/`set_database_url`, so the seed landed in
`database.py`'s default `agromet.db` while `init_db()` (which ran after the
routers were imported) correctly built the *configured* database's schema
right next to a market table nobody had populated. Every existing test
still passed, because none of them asserted the seed landed anywhere in
particular.

Needs a subprocess, not an in-process import: `backend.app.main` is only
ever imported once per interpreter (Python caches modules), so a second
`import` in the same test process would not re-run its module-level code
and would prove nothing about ordering.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "backend" / "agromet.db"


class SeedDataLandsInTheConfiguredDatabaseTests(unittest.TestCase):
    def test_importing_main_seeds_the_configured_sqlite_file_not_the_default(self):
        default_existed_before = DEFAULT_DB_PATH.exists()
        default_mtime_before = DEFAULT_DB_PATH.stat().st_mtime if default_existed_before else None

        with tempfile.TemporaryDirectory() as tmp_dir:
            configured_path = Path(tmp_dir) / "configured.db"
            probe_script = Path(tmp_dir) / "probe.py"
            probe_script.write_text(
                "from backend.app.main import app\n"
                "from backend.app.database import get_connection\n"
                "with get_connection() as conn:\n"
                "    row = conn.execute('SELECT COUNT(*) AS n FROM commodities').fetchone()\n"
                "print(row['n'])\n",
                encoding="utf-8",
            )

            env = dict(os.environ)
            env["DATABASE_PATH"] = str(configured_path)
            env["DATABASE_URL"] = ""
            env["APP_ENV"] = "test"
            env["SECRET_KEY"] = "test-secret"
            # `cwd=` alone doesn't put the repo root on sys.path for a script
            # file the way it does for `python -c` -- a script's own directory
            # takes that slot instead. Same fix conftest.py documents for
            # pytest itself.
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

            result = subprocess.run(
                [sys.executable, str(probe_script)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, msg=f"subprocess failed:\n{result.stderr}")
            self.assertTrue(configured_path.exists(), "init_db() should have created the configured SQLite file")

            seeded_count = int(result.stdout.strip().splitlines()[-1])
            self.assertEqual(
                seeded_count,
                15,
                "seed_market_data() should have populated the *configured* database "
                "(DATABASE_PATH), not database.py's hardcoded default",
            )

        # The default file must be untouched by this run -- either it still
        # doesn't exist, or (on a machine where a developer has run the app
        # locally before) its modification time didn't just change.
        if not default_existed_before:
            self.assertFalse(
                DEFAULT_DB_PATH.exists(),
                "a subprocess configured to use a different DATABASE_PATH created "
                "database.py's default agromet.db -- the seed ran before the "
                "configured path took effect",
            )
        else:
            self.assertEqual(
                DEFAULT_DB_PATH.stat().st_mtime,
                default_mtime_before,
                "the default agromet.db was modified by a run configured to use a "
                "different DATABASE_PATH -- the seed ran before the configured path "
                "took effect",
            )


if __name__ == "__main__":
    unittest.main()
