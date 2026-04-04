from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "agromet.db"


def set_database_path(path: str | Path) -> None:
    global DATABASE_PATH
    DATABASE_PATH = Path(path)


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


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


def init_db() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        connection.executescript(
            """
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
            """
        )


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def encode_payload(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True)


def decode_payload(value: str | None) -> dict:
    if not value:
        return {}
    return json.loads(value)
