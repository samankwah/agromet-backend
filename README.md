# AgroMet Backend

REST API server for the AgroMet agricultural meteorological advisory platform. Handles authentication, agricultural data management, crop calendars, weekly advisories, crop disease diagnosis, and market intelligence for Ghana's agricultural sector.

## Tech Stack

- **Framework:** FastAPI
- **Runtime:** Python 3.11+, Uvicorn (ASGI)
- **Database:** SQLite (local dev, zero config) or Postgres (production -- see [The database](#the-database))
- **Auth:** JWT (PyJWT) with OAuth2 bearer tokens
- **HTTP Client:** httpx (async, for external API calls)
- **External APIs:** OpenAI, Kindwise (crop health + plant ID), Google Translate fallback, Ambee (weather)

## API Endpoints

### Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Create account |
| POST | `/api/v1/auth/login` | Login, returns JWT |
| GET | `/api/v1/auth/me` | Get current user profile |

### Agricultural Data

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/agricultural-data/upload` | Upload agricultural records (Excel/JSON) |
| GET | `/api/agricultural-data/{dataType}` | Retrieve records by type |
| DELETE | `/api/agricultural-data/{dataType}/{recordId}` | Delete a record |

### Crop Calendars

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/crop-calendars/create` | Create a crop or poultry calendar |
| GET | `/api/crop-calendars/district/{district}` | Get calendars by district |
| GET | `/api/crop-calendars/search` | Search calendars with filters |
| GET | `/api/crop-calendars/stats` | Calendar statistics |

### Weekly Advisories

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/weekly-advisories/upload` | Upload advisory bulletin |

### Market Intelligence

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/market/commodities` | List all commodities with prices |
| GET | `/api/market/commodities/{slug}` | Get single commodity data |
| GET | `/api/market/trends` | Historical price trends |
| GET | `/api/market/trends/{slug}` | Trend data for a commodity |
| GET | `/api/market/regions` | Market centers by region |
| GET | `/api/market/regions/{region}` | Single region market data |

> Market endpoints are currently being implemented. Database schema is in place.

### File Management

| Method | Endpoint | Description |
|---|---|---|
| POST | `/user/files/upload` | Upload a file |
| GET | `/user/files` | List user files |
| DELETE | `/user/files/{fileId}` | Delete a file |
| GET | `/user/files/{fileId}/download` | Download a file |

### Dashboard

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/user/dashboard/stats` | Aggregated dashboard statistics |

## Database Schema

The same schema either way (see [The database](#the-database)) -- these tables:

- **users** -- Account credentials and profile
- **agricultural_records** -- Uploaded agricultural data (JSON payloads)
- **calendars** -- Crop and poultry calendar definitions
- **calendar_activities** -- Activities within a calendar (start/end weeks)
- **weekly_advisories** -- Agro-meteorological advisory bulletins
- **weekly_advisory_activities** -- Individual advisory activities
- **production_cycles** -- Active production cycle tracking
- **diagnosis_records** -- Crop disease diagnosis results
- **commodities** -- Market commodity prices, trends, demand levels
- **commodity_trends** -- Historical price data and seasonal patterns
- **market_centers** -- Regional market information and price premiums

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/samankwah/agromet-backend.git
cd agromet-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `SECRET_KEY` | JWT signing secret (change in production) |
| `DATABASE_PATH` | SQLite database path (default: `./agromet.db`); ignored when `DATABASE_URL` is set |
| `DATABASE_URL` | Postgres connection string. **Required in production** -- see [The database](#the-database) |
| `DATABASE_POOL_MIN` / `DATABASE_POOL_MAX` | Postgres connection pool size (default `0` / `5`) |
| `FRONTEND_ORIGINS` | Allowed CORS origins |
| `OPENAI_API_KEY` | OpenAI API key (chatbot) |
| `KINDWISE_API_KEY` | Kindwise crop health API key |
| `AMBEE_API_KEY` | Ambee weather data API key |

### Running

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs` (Swagger UI).

### Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Runs from `backend/` or from the repo root (`python -m pytest backend/tests`);
`pytest.ini` puts the repo root on `sys.path` either way, because every test
module imports `backend.app....`. The suite is unittest-style, so
`python -m unittest discover -s tests -t ..` works without installing pytest at
all. No API keys are needed: the tests that cover the assistant fake the
provider.

The suite runs against a throwaway SQLite file, never a developer's own
`agromet.db` (`tests/conftest.py`). Set `TEST_DATABASE_URL` to a Postgres
connection string to additionally run `test_database_dialects.py` -- schema
parity, `RETURNING`, upsert, and cascade-delete, for real, against Postgres.
A local `docker run -e POSTGRES_PASSWORD=test -p 5432:5432 postgres:16-alpine`
plus `TEST_DATABASE_URL=postgresql://postgres:test@localhost:5432/postgres`
is enough. Skipped, not failed, when unset.

## Project Structure

```
app/
  main.py              # Composition root: build the app, wire the database, mount every router
  config.py            # Every setting this service reads from its environment, in one place
  deps.py              # Shared FastAPI dependencies -- who's asking (get_current_user & co.)
  records.py           # Calendar/advisory insert logic two different routers both need
  database.py          # SQLite/Postgres connection, schema initialization
  schemas.py           # Pydantic request/response models
  auth.py              # JWT token creation and verification
  domain.py            # Business logic (calendars, advisories, cycles)
  diagnosis.py         # Crop disease diagnosis integration
  spreadsheet_parser.py # Excel upload parsing and preview
  chat_prompt.py       # What AgroMet AI is told, and how a turn is assembled
  chat_context.py      # The live forecast/hazard/price figures it answers from
  rate_limit.py        # The quota in front of the route that spends money
  logging_config.py    # Somewhere for log records to actually go
  routers/             # One module per domain -- see below
    accounts.py         # Register, log in, /auth/me
    chat.py              # The assistant, transcription, translation, TTS stubs
    weather.py            # Ambee proxy + the Open-Meteo bundle
    diagnosis.py           # Crop disease diagnosis endpoints + history
    content.py               # FAQ, Terms/Privacy, the Contact form
    agricultural_data.py      # The generic record upload/list/delete path
    calendars.py                # Crop/poultry calendar preview+commit, enhanced-calendars
    advisories.py                 # Weekly advisory preview+commit, list/get/delete
    production_cycles.py           # Batches tracked against a calendar
    dashboard.py                    # Aggregate counts for the admin dashboard
    market.py                        # Commodity prices, trends, market centers
    hazards.py                        # Flood/drought summary, regions, overrides
    outlook.py                        # Subseasonal outlook + precipitation field
    health.py                          # Liveness and integration-status
tests/
  conftest.py               # Isolates the suite from a developer's real database
  test_chat.py              # The assistant: prompt, model call, validation, quota
  test_chat_context.py      # Grounding: intent routing and the rendered figures
  test_rate_limit.py        # The quota, including the shared-address case
  test_diagnosis.py         # Diagnosis module tests
  test_database_dialects.py # SQLite/Postgres parity -- skipped without TEST_DATABASE_URL
```

`main.py` was a single ~2,900-line file until this split -- every route, every
domain helper and every config constant in one module that every change had
a chance of colliding with. Splitting it changed no behavior: the same 72
routes exist at the same paths, verified by diffing the method+path pairs
before and after, and the full test suite (SQLite and Postgres both) passes
unchanged. Config constants live in `config.py` and are read as
`config.NAME` (module attribute access) rather than imported by name, on
purpose -- it is what lets `unittest.mock.patch("backend.app.config.NAME", ...)`
reach whichever router actually reads it at request time, regardless of
which one that turns out to be.

### The database

SQLite is the default because it needs no setup: clone the repo, run
`uvicorn`, and `agromet.db` appears next to it. That default is correct for
local dev and wrong for this app's actual production target. The backend
deploys to Vercel (`vercel.json`) as a serverless function, and Vercel's
filesystem is read-only outside `/tmp` -- which is itself wiped on every cold
start and not shared between concurrent instances. Point SQLite there (which
`resolve_database_path` in `main.py` does automatically, so it at least
doesn't crash) and every signup, uploaded calendar, diagnosis record and
contact message survives only until that particular instance recycles, then
is gone, with nothing in the logs to say so.

Setting `DATABASE_URL` to a Postgres connection string switches the whole app
off that path -- **do this before trusting the deployment with real farmer
data.** Neon, Supabase and Vercel Postgres all publish a free tier and a
*pooled* connection string (use that one, not the direct one: a serverless
function can run several concurrent instances, each wanting its own
connection). `database.py`'s module docstring covers how one code path
serves both databases; `test_database_dialects.py` proves it against a real
Postgres rather than asserting it from review.

Every other piece of state in this service already lives outside process
memory except one, and it already knows it: the chat rate limiter, next.

### The assistant

`/api/chat` is unauthenticated, so it is metered rather than gated: a quota per
device and a looser one per address (`CHAT_RATE_LIMIT`, `CHAT_DAILY_LIMIT`),
plus hard ceilings on question length and on `max_output_tokens`. The quota
counter lives in process memory, which is exact on a long-running host and best
effort on a serverless one -- see the module docstring in `rate_limit.py`.

Answers are grounded: before the model is called, `chat_context` gathers the
forecast, flood and drought bands, and market prices for the farmer's region,
but only the ones the question actually needs. Every source is bounded and
optional, so a slow upstream costs a section of the answer rather than the
answer.

With no `OPENAI_API_KEY` the route still answers, with a plainly worded fallback
and `degraded: true`; `degradedReason` says which failure it was (`no_key`,
`timeout`, `upstream_error`, `empty_output`).

## Related

- **Frontend:** [samankwah/agromet-frontend](https://github.com/samankwah/agromet-frontend)

## License

MIT
