"""The composition root: build the app, wire the database, mount every
router.

This file used to be almost 3,000 lines -- every route, every domain helper,
every config constant, all in one module that every change had a chance of
colliding with. It is now what a composition root should be: config loading,
the database, the FastAPI app and its middleware, and a list of routers. The
actual endpoints live in `routers/`, one module per domain; the config
values they read live in `config.py`; the auth dependency they share lives
in `deps.py`; the calendar/advisory insert logic two different routers both
need lives in `records.py`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .database import close_all_connections, init_db, set_database_path, set_database_url
from .logging_config import configure_logging

configure_logging()

# Database config has to be wired up *before* any router is imported, not
# just before the app starts serving: `routers/market.py` calls
# `seed_market_data()` as a side effect of being imported, and that write
# needs `database.py` already pointed at the real target. Import a router
# earlier than this and its seed lands in the SQLite default instead of
# whichever database `DATABASE_URL` actually names -- silently, since
# nothing here raises when that happens, it just seeds the wrong file.
set_database_path(config.DATABASE_PATH)
set_database_url(config.DATABASE_URL)
init_db()

from .routers import (  # noqa: E402 -- see the comment above; this ordering is load-bearing
    accounts,
    advisories,
    agricultural_data,
    calendars,
    chat,
    content,
    dashboard,
    diagnosis,
    hazards,
    health,
    market,
    outlook,
    production_cycles,
    weather,
)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    yield
    # No-op on SQLite. On Postgres, releases the pool's connections instead
    # of leaking them across every `uvicorn --reload` and test run.
    close_all_connections()


app = FastAPI(title=config.APP_NAME, debug=config.DEBUG, lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.FRONTEND_ORIGINS,
    allow_origin_regex=config.LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registration order matters only where two routers could otherwise both
# match the same request; FastAPI/Starlette tries routes in the order they
# were added. Within `advisories`, `calendars` and `hazards` a literal path
# (e.g. `/api/weekly-advisories/activities`) is always registered ahead of
# the `{id}` route it would otherwise be shadowed by -- see each router's own
# ordering for why. Across routers here, paths don't overlap, so this list's
# order is just alphabetical for anyone scanning it.
app.include_router(accounts.router)
app.include_router(advisories.router)
app.include_router(advisories.poultry_router)
app.include_router(agricultural_data.router)
app.include_router(calendars.router)
app.include_router(chat.router)
app.include_router(content.router)
app.include_router(dashboard.router)
app.include_router(diagnosis.router)
app.include_router(hazards.router)
app.include_router(health.router)
app.include_router(market.router)
app.include_router(outlook.router)
app.include_router(production_cycles.router)
app.include_router(weather.router)
