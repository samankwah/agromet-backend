"""Commodity prices, trends and regional market centers.

`seed_market_data()` runs at import time, same as it always has -- this
module is imported once, when `main.py` registers the router, so the timing
is unchanged even though the call no longer sits in `main.py` itself.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from ..database import get_connection
from ..schemas import CommodityResponse, CommodityTrendResponse, MarketCenterResponse

router = APIRouter(prefix="/api/market", tags=["market"])

SEED_COMMODITIES = [
    ("yellow-maize", "Yellow Maize", "Maize", 299.99, "per bag", "stable", "high"),
    ("white-maize", "White Maize", "Maize", 289.99, "per bag", "rising", "high"),
    ("rice", "Rice", "Rice", 159.99, "per bag", "stable", "very-high"),
    ("yam", "Yam", "Yam", 389.99, "per bag", "rising", "high"),
    ("cassava", "Cassava", "Cassava", 129.99, "per bag", "stable", "moderate"),
    ("tomatoes", "Tomatoes", "Tomatoes", 149.99, "per crate", "volatile", "high"),
    ("pepper", "Pepper", "Pepper", 59.99, "per bag", "rising", "high"),
    ("onion", "Onion", "Onion", 89.99, "per bag", "seasonal", "moderate"),
    ("plantain", "Plantain", "Plantain", 79.99, "per bunch", "stable", "high"),
    ("beans", "Beans", "Beans", 199.99, "per bag", "rising", "moderate"),
    ("soybeans", "Soybeans", "Soybeans", 399.99, "per bag", "stable", "growing"),
    ("sorghum", "Sorghum", "Sorghum", 189.99, "per bag", "stable", "low"),
    ("groundnuts", "Groundnuts", "Groundnuts", 249.99, "per bag", "rising", "moderate"),
    ("cocoa", "Cocoa", "Cocoa", 850.00, "per bag", "volatile", "export"),
    ("poultry", "Poultry", "Poultry", 45.00, "per kg", "rising", "very-high"),
]

# Every commodity above carries a trend entry. The market UI draws a sparkline
# on each card and a full price chart on the commodity page, so a commodity
# without a trend is a commodity with a visibly broken page. The last point of
# each series is the commodity's current price, by construction.
SEED_TRENDS = {
    "yellow-maize": {
        "6months": [280, 285, 290, 295, 298, 299.99],
        "seasonal_pattern": "Low during harvest (July-August), High during planting (March-April)",
        "peak_months": [3, 4, 5],
        "low_months": [7, 8, 9],
    },
    "white-maize": {
        "6months": [265, 270, 276, 282, 287, 289.99],
        "seasonal_pattern": "Tracks yellow maize, but firmer when household demand for banku and kenkey is strong",
        "peak_months": [3, 4, 5],
        "low_months": [8, 9, 10],
    },
    "rice": {
        "6months": [150, 152, 155, 157, 158, 159.99],
        "seasonal_pattern": "Stable year-round, slight increase during festivals",
        "peak_months": [12, 1],
        "low_months": [6, 7, 8],
    },
    "tomatoes": {
        "6months": [120, 140, 160, 180, 170, 149.99],
        "seasonal_pattern": "Very volatile, peaks during dry season",
        "peak_months": [1, 2, 3],
        "low_months": [6, 7, 8],
    },
    "yam": {
        "6months": [350, 360, 370, 380, 385, 389.99],
        "seasonal_pattern": "Peaks before harvest, drops after new yam season",
        "peak_months": [6, 7, 8],
        "low_months": [9, 10, 11],
    },
    "cassava": {
        "6months": [126, 127, 128, 128.5, 129, 129.99],
        "seasonal_pattern": "Flat year-round; roots can be left in the ground until they are needed",
        "peak_months": [2, 3],
        "low_months": [8, 9],
    },
    "pepper": {
        "6months": [48, 51, 54, 57, 59, 59.99],
        "seasonal_pattern": "Climbs through the dry season as irrigated volumes thin out",
        "peak_months": [12, 1, 2],
        "low_months": [6, 7, 8],
    },
    "onion": {
        "6months": [110, 102, 95, 90, 88, 89.99],
        "seasonal_pattern": "Strongly seasonal; falls once northern and Sahel stock arrives",
        "peak_months": [4, 5, 6],
        "low_months": [10, 11, 12],
    },
    "plantain": {
        "6months": [72, 75, 82, 85, 81, 79.99],
        "seasonal_pattern": "Cannot be stored, so the price follows that week's arrivals",
        "peak_months": [1, 2, 3],
        "low_months": [7, 8, 9],
    },
    "beans": {
        "6months": [178, 183, 189, 194, 197, 199.99],
        "seasonal_pattern": "Stores well, so the price rises steadily through the lean season",
        "peak_months": [4, 5, 6],
        "low_months": [11, 12],
    },
    "soybeans": {
        "6months": [372, 380, 388, 393, 397, 399.99],
        "seasonal_pattern": "Crusher demand outruns local supply, so harvest dips stay shallow",
        "peak_months": [2, 3, 4],
        "low_months": [11, 12],
    },
    "sorghum": {
        "6months": [180, 182, 185, 187, 188, 189.99],
        "seasonal_pattern": "Steady brewer and feed-mill demand; thin volumes move slowly",
        "peak_months": [3, 4],
        "low_months": [10, 11],
    },
    "groundnuts": {
        "6months": [225, 231, 238, 243, 247, 249.99],
        "seasonal_pattern": "Rises through the lean season once the northern harvest is sold down",
        "peak_months": [4, 5, 6],
        "low_months": [10, 11, 12],
    },
    "cocoa": {
        "6months": [790, 815, 870, 905, 862, 850.00],
        "seasonal_pattern": "Volatile; set by the world price and the announced farmgate rate",
        "peak_months": [10, 11, 12],
        "low_months": [5, 6, 7],
    },
    "poultry": {
        "6months": [41, 42, 43, 44, 44.5, 45.00],
        "seasonal_pattern": "Spikes in December and around Easter; feed-grain cost sets the floor",
        "peak_months": [12, 4],
        "low_months": [6, 7, 8],
    },
}

SEED_MARKET_CENTERS = [
    ("Greater Accra", ["Tema Market", "Kaneshie Market", "Makola Market"], "excellent", 1.1),
    ("Ashanti", ["Kumasi Central Market", "Kejetia Market"], "good", 1.05),
    ("Northern", ["Tamale Market", "Yendi Market"], "fair", 0.95),
    ("Western", ["Takoradi Market", "Tarkwa Market"], "good", 1.02),
]


def seed_market_data() -> None:
    """Bring the market tables up to date with the seed data above.

    This upserts rather than bailing out on a non-empty table. The earlier
    "insert only if empty" guard meant that any commodity or trend added to
    the seeds after first run never reached an existing agromet.db, which is
    how the database ended up serving four trends for fourteen commodities.
    There is no write API for market data, so there is no operator-entered
    state here to protect.
    """
    with get_connection() as conn:
        for slug, name, category, price, unit, trend, demand in SEED_COMMODITIES:
            conn.execute(
                """
                INSERT INTO commodities (slug, name, category, price, unit, trend, demand)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug) DO UPDATE SET
                    name = excluded.name,
                    category = excluded.category,
                    price = excluded.price,
                    unit = excluded.unit,
                    trend = excluded.trend,
                    demand = excluded.demand,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (slug, name, category, price, unit, trend, demand),
            )

        for slug, data in SEED_TRENDS.items():
            conn.execute(
                """
                INSERT INTO commodity_trends (commodity_slug, month_prices_json, seasonal_pattern, peak_months_json, low_months_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(commodity_slug) DO UPDATE SET
                    month_prices_json = excluded.month_prices_json,
                    seasonal_pattern = excluded.seasonal_pattern,
                    peak_months_json = excluded.peak_months_json,
                    low_months_json = excluded.low_months_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (slug, json.dumps(data["6months"]), data["seasonal_pattern"], json.dumps(data["peak_months"]), json.dumps(data["low_months"])),
            )

        for region, markets, transport, premium in SEED_MARKET_CENTERS:
            conn.execute(
                """
                INSERT INTO market_centers (region, major_markets_json, transport_access, price_premium)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(region) DO UPDATE SET
                    major_markets_json = excluded.major_markets_json,
                    transport_access = excluded.transport_access,
                    price_premium = excluded.price_premium,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (region, json.dumps(markets), transport, premium),
            )


seed_market_data()


# The market collections are serialized as maps keyed by slug/region, which is
# the shape both clients already read, so the schemas below cannot be used as
# FastAPI `response_model=`. They are applied to each entry instead: the row
# still has to satisfy the declared contract before it goes out, and every
# entry now carries its own slug/region rather than relying on the map key.


def _commodity_payload(row) -> dict:
    return CommodityResponse(**dict(row)).model_dump()


def _trend_payload(row) -> dict:
    return CommodityTrendResponse(
        commodity_slug=row["commodity_slug"],
        seasonal_pattern=row["seasonal_pattern"],
        peak_months=json.loads(row["peak_months_json"]),
        low_months=json.loads(row["low_months_json"]),
        **{"6months": json.loads(row["month_prices_json"])},
    ).model_dump(by_alias=True)


def _market_center_payload(row) -> dict:
    return MarketCenterResponse(
        region=row["region"],
        major_markets=json.loads(row["major_markets_json"]),
        transport_access=row["transport_access"],
        price_premium=row["price_premium"],
    ).model_dump()


COMMODITY_COLUMNS = "slug, name, category, price, unit, trend, demand"
TREND_COLUMNS = "commodity_slug, month_prices_json, seasonal_pattern, peak_months_json, low_months_json"
MARKET_CENTER_COLUMNS = "region, major_markets_json, transport_access, price_premium"


@router.get("/commodities")
def get_commodities():
    with get_connection() as conn:
        rows = conn.execute(f"SELECT {COMMODITY_COLUMNS} FROM commodities ORDER BY name").fetchall()
    return {"success": True, "data": {row["slug"]: _commodity_payload(row) for row in rows}}


@router.get("/commodities/{slug}")
def get_commodity(slug: str):
    with get_connection() as conn:
        row = conn.execute(f"SELECT {COMMODITY_COLUMNS} FROM commodities WHERE slug = ?", (slug,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Commodity not found")
    return {"success": True, "data": _commodity_payload(row)}


@router.get("/trends")
def get_trends():
    with get_connection() as conn:
        rows = conn.execute(f"SELECT {TREND_COLUMNS} FROM commodity_trends").fetchall()
    return {"success": True, "data": {row["commodity_slug"]: _trend_payload(row) for row in rows}}


@router.get("/trends/{slug}")
def get_trend(slug: str):
    with get_connection() as conn:
        row = conn.execute(f"SELECT {TREND_COLUMNS} FROM commodity_trends WHERE commodity_slug = ?", (slug,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Trend data not found")
    return {"success": True, "data": _trend_payload(row)}


@router.get("/regions")
def get_regions():
    with get_connection() as conn:
        rows = conn.execute(f"SELECT {MARKET_CENTER_COLUMNS} FROM market_centers ORDER BY region").fetchall()
    return {"success": True, "data": {row["region"]: _market_center_payload(row) for row in rows}}


@router.get("/regions/{region}")
def get_region(region: str):
    with get_connection() as conn:
        row = conn.execute(f"SELECT {MARKET_CENTER_COLUMNS} FROM market_centers WHERE region = ?", (region,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Region not found")
    return {"success": True, "data": _market_center_payload(row)}
