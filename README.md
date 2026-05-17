# AgroMet Backend

REST API server for the AgroMet agricultural meteorological advisory platform. Handles authentication, agricultural data management, crop calendars, weekly advisories, crop disease diagnosis, and market intelligence for Ghana's agricultural sector.

## Tech Stack

- **Framework:** FastAPI
- **Runtime:** Python 3.11+, Uvicorn (ASGI)
- **Database:** SQLite
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

SQLite database with the following tables:

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
| `DATABASE_PATH` | SQLite database path (default: `./agromet.db`) |
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
python -m pytest tests/
```

## Project Structure

```
app/
  main.py              # FastAPI app, routes, middleware
  database.py          # SQLite connection, schema initialization
  schemas.py           # Pydantic request/response models
  auth.py              # JWT token creation and verification
  domain.py            # Business logic (calendars, advisories, cycles)
  diagnosis.py         # Crop disease diagnosis integration
  spreadsheet_parser.py # Excel upload parsing and preview
tests/
  test_diagnosis.py    # Diagnosis module tests
```

## Related

- **Frontend:** [samankwah/agromet-frontend](https://github.com/samankwah/agromet-frontend)

## License

MIT
