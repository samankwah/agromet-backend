"""Make the app's own log records go somewhere.

`main.py` has always held a `logging.getLogger(__name__)` and used it in the one
place that matters most -- the chat path, where an upstream failure is otherwise
indistinguishable from a missing API key. But nothing ever configured logging,
so under a bare uvicorn those records went to a root logger with no handler and
the warning that explained a degraded answer was thrown away.

Deliberately small: a level, a format, and a handler on the root logger, applied
once and never over an existing configuration. Anything richer (JSON lines, a
request id, a shipping destination) belongs with whoever runs this, and guessing
at it here would only be something to undo.
"""

from __future__ import annotations

import logging
import os

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Attach a handler to the root logger unless one is already there.

    The guard is what keeps this safe under uvicorn and gunicorn, which install
    their own handlers: adding a second one prints every line twice, and having
    watched a log file fill with duplicates is how that becomes obvious.
    """
    root = logging.getLogger()
    resolved = (level or os.getenv("LOG_LEVEL") or "INFO").upper()

    if root.handlers:
        root.setLevel(resolved)
    else:
        logging.basicConfig(level=resolved, format=DEFAULT_FORMAT)

    # httpx logs a line per request at INFO, which at this app's level means one
    # line for every forecast tile, every hazard refresh and every model call.
    # That buries the handful of records worth reading, and the URLs it prints
    # carry query parameters. Failures still come through at WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
