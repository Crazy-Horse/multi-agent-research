from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from fredapi import Fred

from langgraphagenticai.state.commodity_state import ExtractionMeta


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_worldbank_exchange_rate(country_code: str, currency_name: str) -> pd.DataFrame:
    """
    Get exchange rate from World Bank for currencies not available in FRED.
    Returns annual data forward-filled to daily frequency.

    Output columns:
        - date (datetime64)
        - {currency_name}_usd (float)
    """
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/PA.NUS.FCRF"
        f"?format=json&per_page=500&date=2015:2025"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if len(data) > 1 and data[1]:
            records = []
            for item in data[1]:
                val = item.get("value")
                if val is None:
                    continue
                year = int(item["date"])
                # Place annual value mid-year, then forward-fill daily
                d = pd.Timestamp(f"{year}-06-30")
                records.append({"date": d, f"{currency_name}_usd": float(val)})

            if records:
                df = pd.DataFrame(records).sort_values("date")
                df = df.set_index("date").resample("D").ffill().reset_index()
                return df

    except Exception:
        pass

    return pd.DataFrame()


def get_daily_macro(fred: Fred, start_date: str) -> pd.DataFrame:
    """
    Fetch macro indicators and FX rates from:
      - FRED (daily-ish)
      - World Bank (annual -> daily forward fill)

    Returns a single daily dataframe with:
      - date (datetime64)
      - multiple numeric columns
    """
    fred_series: Dict[str, str] = {
        "DEXBZUS": "brl_usd",
        "DEXINUS": "inr_usd",
        "DEXCHUS": "cny_usd",
        "DEXTHUS": "thb_usd",
        "DEXMXUS": "mxn_usd",
        "DEXUSAL": "aud_usd",
        "DEXUSEU": "eur_usd",
        "DEXSFUS": "zar_usd",
        "DEXJPUS": "jpy_usd",
        "DEXSZUS": "chf_usd",
        "DEXKOUS": "krw_usd",
        "DEXUSUK": "gbp_usd",
        "DCOILWTICO": "oil_wti",
        "DTWEXBGS": "usd_index",
        "DGS10": "us_10yr_rate",
    }

    dfs: List[pd.DataFrame] = []

    # FRED series
    for code, col_name in fred_series.items():
        try:
            s = fred.get_series(code, observation_start=start_date)
            df = pd.DataFrame({"date": s.index, col_name: s.values})
            dfs.append(df)
            time.sleep(0.25)  # polite pacing
        except Exception:
            # Non-fatal: just skip missing series
            continue

    # World Bank currencies
    worldbank_currencies: List[Tuple[str, str]] = [
        ("VNM", "vnd"),
        ("COL", "cop"),
        ("IDN", "idr"),
        ("ETH", "etb"),
        ("HND", "hnl"),
        ("UGA", "ugx"),
        ("PER", "pen"),
        ("CAF", "xaf"),
        ("GTM", "gtq"),
        ("GIN", "gnf"),
        ("NIC", "nio"),
        ("CRI", "crc"),
        ("TZA", "tzs"),
        ("KEN", "kes"),
        ("LAO", "lak"),
        ("PAK", "pkr"),
        ("PHL", "php"),
        ("EGY", "egp"),
        ("CUB", "cup"),
        ("ARG", "ars"),
        ("RUS", "rub"),
        ("TUR", "try"),
        ("UKR", "uah"),
        ("IRN", "irr"),
        ("BLR", "byn"),
    ]

    for country_code, currency in worldbank_currencies:
        wb_df = get_worldbank_exchange_rate(country_code, currency)
        if not wb_df.empty:
            dfs.append(wb_df)

    if not dfs:
        return pd.DataFrame()

    # Merge on date
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(result, df, on="date", how="outer")

    # Normalize and sort
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    start_dt = pd.to_datetime(start_date, errors="coerce")
    if pd.notna(start_dt):
        result = result[result["date"] >= start_dt]

    # Force numeric coercion for all non-date columns
    for c in result.columns:
        if c == "date":
            continue
        result[c] = pd.to_numeric(result[c], errors="coerce")

    return result

def fred_data_extraction_node(state):
    """
    LangGraph deterministic node:
      - pulls macro + FX daily series
      - writes CSV + metadata JSON
      - returns a partial state update dict
    """
    topic = state.topic
    start_date = state.start_date
    end_date = state.end_date  # treat as inclusive

    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        return {
            "errors": state.errors + [
                {
                    "node": "fred_data_extraction_node",
                    "message": "Missing FRED_API_KEY in environment.",
                    "severity": "fatal",
                    "suggested_action": "Set FRED_API_KEY in your shell or .env and restart the kernel.",
                }
            ]
        }

    out_csv = Path(f"data/raw/{topic}_fred_macro_fx_daily.csv")
    meta_path = Path(f"data/raw/{topic}_fred_extraction_metadata.json")
    _ensure_dir(out_csv.parent)

    fred = Fred(api_key=fred_key)
    df = get_daily_macro(fred, start_date=start_date)

    if df is None or df.empty:
        return {
            "warnings": state.warnings + ["FRED/WorldBank macro extraction returned no data."],
            "has_macro_data": False,
            "has_fx_data": False,
        }

    # ---- Ensure df['date'] is datetime and filter to end_date (inclusive) ----
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    end_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.notna(end_dt):
        df = df[df["date"] <= end_dt]

    df = df.sort_values("date").reset_index(drop=True)

    # ---- Write CSV with ISO dates ----
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.date.astype(str)
    df_out.insert(0, "commodity", topic)

    # FX columns are daily exchange rates (either FRED or WorldBank forward-filled)
    fx_cols = [c for c in df_out.columns if c.endswith("_usd")]

    # Macro columns (exclude identifiers and exclude FX)
    macro_cols = [c for c in df_out.columns if c not in {"commodity", "date"} and not c.endswith("_usd")]

    # Coverage tests: consider data "available" if at least 1 column has > X% non-null
    def _has_usable_data(cols, min_non_null_frac=0.10):
        if not cols:
            return False
        non_null_fracs = [(df_out[c].notna().mean()) for c in cols]
        return max(non_null_fracs) >= min_non_null_frac

    has_fx = _has_usable_data(fx_cols, min_non_null_frac=0.10)
    has_macro = _has_usable_data(macro_cols, min_non_null_frac=0.10)

    df_out.to_csv(out_csv, index=False, na_rep="")

    # ---- Column classification ----
    fx_cols = [c for c in df_out.columns if c.endswith("_usd")]
    # exclude identifier columns
    macro_cols = [c for c in df_out.columns if c not in {"commodity", "date"}]

    # ---- Metadata: use a real datetime for state; write ISO string to disk ----
    as_of = datetime.now(timezone.utc).replace(microsecond=0)

    meta_json = {
        "as_of_utc": as_of.isoformat(),
        "provider": "fredapi+worldbank",
        "dataset_id": "macro_fx_daily",
        "symbols": None,
        "start_date": start_date,
        "end_date": end_date,
        "row_count": int(df_out.shape[0]),
        "series_count": int(len(macro_cols)),
        "fx_series_count": int(len(fx_cols)),
        "notes": "FRED provides daily-ish series; World Bank series are annual and forward-filled to daily.",
    }
    meta_path.write_text(json.dumps(meta_json, indent=2, sort_keys=True), encoding="utf-8")

    # ---- State updates (typed ExtractionMeta) ----
    raw_paths = dict(state.raw_data_paths)
    raw_paths["macro_fx"] = str(out_csv)

    updated_meta = dict(state.extraction_metadata)
    updated_meta["fred"] = ExtractionMeta(
        provider=meta_json["provider"],
        dataset_id=meta_json["dataset_id"],
        symbols=None,
        start_date=meta_json["start_date"],
        end_date=meta_json["end_date"],
        as_of_utc=as_of,
        row_count=meta_json["row_count"],
        notes=meta_json["notes"],
    )

    return {
        "raw_data_paths": raw_paths,
        "extraction_metadata": updated_meta,
        "has_macro_data": has_macro,
        "has_fx_data": has_fx,
    }
