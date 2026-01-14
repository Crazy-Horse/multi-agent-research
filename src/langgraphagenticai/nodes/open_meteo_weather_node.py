import json
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError

from langgraphagenticai.nodes.yahoo_data_extraction_node import _ensure_dir

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class Location(BaseModel):
    name: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class OpenMeteoRequest(BaseModel):
    locations: List[Location] = Field(..., min_length=1)
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    daily_vars: List[str] = Field(
        default_factory=lambda: [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]
    )
    timezone: str = Field(default="UTC")
    use_archive: Optional[bool] = Field(default=None)

def _utc_now():
    return datetime.now(timezone.utc)

def _choose_endpoint(end_date: date, use_archive: Optional[bool]) -> str:
    if use_archive is True:
        return ARCHIVE_URL
    if use_archive is False:
        return FORECAST_URL
    today = date.today()
    # If entirely in the past (buffer), archive is safer
    if end_date <= (today - timedelta(days=2)):
        return ARCHIVE_URL
    return FORECAST_URL


def _call(url: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo failed: {r.status_code} {r.reason}. "
            f"Params={json.dumps(params)}. Body={r.text[:500]}"
        )
    return r.json()


def _payload_to_df(location_name: str, payload: Dict[str, Any]) -> pd.DataFrame:
    daily = payload.get("daily")
    if not daily or "time" not in daily:
        reason = payload.get("reason") or payload.get("message") or "No daily data returned."
        raise RuntimeError(f"Open-Meteo returned no daily data for {location_name}. Reason: {reason}")
    df = pd.DataFrame(daily).rename(columns={"time": "date"})
    df.insert(1, "location", location_name)
    return df

def fetch_open_meteo_daily_df(
    locations: List[dict],
    start_date: str,
    end_date: str,
    daily_vars: Optional[List[str]] = None,
    timezone: str = "UTC",
    use_archive: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Fetch daily weather from Open-Meteo for one or more locations and return CSV text.
    Input locations: [{"name":"MinasGerais","latitude":-18.5,"longitude":-44.6}, ...]
    """
    try:
        req = OpenMeteoRequest(
            locations=locations,
            start_date=start_date,
            end_date=end_date,
            daily_vars=daily_vars or OpenMeteoRequest.model_fields["daily_vars"].default_factory(),  # type: ignore
            timezone=timezone,
            use_archive=use_archive,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid Open-Meteo inputs: {e}") from e

    s = date.fromisoformat(req.start_date)
    e = date.fromisoformat(req.end_date)
    if e < s:
        raise ValueError("end_date must be >= start_date")

    endpoint = _choose_endpoint(e, req.use_archive)

    frames: List[pd.DataFrame] = []
    for loc in req.locations:
        params = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "daily": ",".join(req.daily_vars),
            "timezone": req.timezone,
        }
        payload = _call(endpoint, params)
        frames.append(_payload_to_df(loc.name, payload))

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    out = out.sort_values(["location", "date"])

    # numeric coercion
    for c in out.columns:
        if c in ("date", "location"):
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def open_meteo_weather_node(state):
    topic = state.topic

    # Require location if enable_exogenous; otherwise skip
    if not state.enable_exogenous:
        return {}

    locations = getattr(state, "weather_locations", None) or []
    if not locations:
        locations = [
            {"name": "Brazil_MinasGerais", "latitude": -18.5, "longitude": -44.6},
            {"name": "Brazil_EspiritoSanto", "latitude": -19.5, "longitude": -40.6},
            {"name": "Vietnam_CentralHighlands", "latitude": 12.7, "longitude": 108.1},
            {"name": "Colombia_Huila", "latitude": 2.5, "longitude": -75.6},
            {"name": "Indonesia_Lampung", "latitude": -5.4, "longitude": 105.3},
            {"name": "Ethiopia_Oromia", "latitude": 8.5, "longitude": 39.0},
        ]

    try:
        df = fetch_open_meteo_daily_df(
            locations=locations,
            start_date=state.start_date,
            end_date=state.end_date,
            timezone="UTC",
            use_archive=True,  # historical stability
        )
    except Exception as e:
        return {
            "has_weather_data": False,
            "warnings": state.warnings + [f"Open-Meteo weather extraction failed: {str(e)[:200]}"],
        }

    if df is None or df.empty:
        return {
            "has_weather_data": False,
            "warnings": state.warnings + ["Open-Meteo returned no daily weather data."],
        }

    # Ensure ISO dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df = df.dropna(subset=["date"]).sort_values(["location", "date"]).reset_index(drop=True)

    out_csv = Path(f"data/raw/{topic}_openmeteo_daily.csv")
    meta_path = Path(f"data/raw/{topic}_openmeteo_extraction_metadata.json")
    _ensure_dir(out_csv.parent)

    df.to_csv(out_csv, index=False, na_rep="")

    daily_vars = [c for c in df.columns if c not in ("date", "location")]
    meta = {
        "as_of_utc": _utc_now().isoformat(),
        "provider": "open-meteo",
        "dataset_id": "daily_weather",
        "locations": locations,
        "start_date": state.start_date,
        "end_date": state.end_date,
        "daily_vars": daily_vars,
        "row_count": int(df.shape[0]),
        "notes": "Open-Meteo archive daily weather for multiple coffee regions; aggregated downstream to date-level features.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    raw_paths = dict(state.raw_data_paths)
    raw_paths["weather"] = str(out_csv)

    extraction_meta = dict(state.extraction_metadata)
    extraction_meta["open_meteo"] = {
        "provider": meta["provider"],
        "dataset_id": meta["dataset_id"],
        "symbols": None,
        "start_date": meta["start_date"],
        "end_date": meta["end_date"],
        "as_of_utc": datetime.fromisoformat(meta["as_of_utc"]),
        "row_count": meta["row_count"],
        "notes": meta["notes"],
    }

    return {
        "raw_data_paths": raw_paths,
        "extraction_metadata": extraction_meta,
        "has_weather_data": True,
    }