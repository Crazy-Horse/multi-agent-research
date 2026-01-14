from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

from langgraphagenticai.state.commodity_state import ExtractionMeta


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def yahoo_data_extraction_node(state):
    """
    Deterministic node: downloads OHLCV and writes raw CSV + metadata.
    Returns a partial state update dict.
    """
    topic = state.topic
    ticker = state.canonical_ticker
    start_date = state.start_date
    end_date = state.end_date

    out_csv = Path(f"data/raw/{topic}_yahoo_ohlcv_1d.csv")
    meta_path = Path(f"data/raw/{topic}_extraction_metadata.json")
    _ensure_dir(out_csv.parent)

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",   # helps keep OHLCV as top-level columns
    )

    if df is None or df.empty:
        return {
            "has_price_data": False,
            "errors": state.errors + [{
                "node": "yahoo_extract_node",
                "message": f"No data returned for ticker={ticker} in {start_date}..{end_date}.",
                "severity": "fatal",
                "suggested_action": "Verify ticker, date range, and network access to Yahoo Finance."
            }]
        }

    # ---- Robust column flattening (MultiIndex-safe) ----
    if isinstance(df.columns, pd.MultiIndex):
        # Common cases:
        # 1) ('Close', 'KC=F') or ('KC=F','Close')
        # Keep any level that matches OHLCV field names.
        field_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        new_cols = []
        for col in df.columns:
            # col is a tuple
            parts = [str(x) for x in col if x is not None]
            # choose the part that looks like a field name if present
            chosen = next((p for p in parts if p in field_names), parts[0])
            new_cols.append(chosen)
        df.columns = new_cols
    else:
        df.columns = [str(c) for c in df.columns]

    df = df.reset_index()

    # ---- Normalize date column ----
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    elif "date" not in df.columns:
        return {
            "has_price_data": False,
            "errors": state.errors + [{
                "node": "yahoo_extract_node",
                "message": f"Yahoo response missing date column. Columns={list(df.columns)}",
                "severity": "fatal",
                "suggested_action": "Inspect Yahoo payload; yfinance may be returning an unexpected schema."
            }]
        }

    # Keep date as datetime in raw stage; stringify only at the end
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ---- Normalize OHLCV names ----
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)

    # Close fallback to adj_close if needed
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    if "close" not in df.columns:
        return {
            "has_price_data": False,
            "errors": state.errors + [{
                "node": "yahoo_extract_node",
                "message": f"Missing close/adj_close after normalization. Columns={list(df.columns)}",
                "severity": "fatal",
                "suggested_action": "Print df.head() and df.columns; adjust normalization logic for your yfinance version."
            }]
        }

    # Keep only expected cols if present
    keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()

    # Numeric coercion
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    # Add identifiers
    df["commodity"] = topic
    df["ticker"] = ticker

    # Write CSV with ISO date strings
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.date.astype(str)
    df_out.to_csv(out_csv, index=False, na_rep="")

    # ---- Metadata (file JSON uses string; state uses datetime) ----
    as_of = datetime.now(timezone.utc).replace(microsecond=0)

    meta_json = {
        "as_of_utc": as_of.isoformat(),
        "provider": "yahoo_finance",
        "dataset_id": "ohlcv_1d",
        "symbols": [ticker],
        "start_date": start_date,
        "end_date": end_date,
        "schema": "ohlcv-1d",
        "row_count": int(df.shape[0]),
        "notes": "Yahoo Finance OHLCV is used as a free proxy; may differ from official settlement.",
    }
    meta_path.write_text(json.dumps(meta_json, indent=2, sort_keys=True), encoding="utf-8")

    # Build typed ExtractionMeta for state
    yahoo_meta = ExtractionMeta(
        provider=meta_json["provider"],
        dataset_id=meta_json["dataset_id"],
        symbols=meta_json["symbols"],
        start_date=meta_json["start_date"],
        end_date=meta_json["end_date"],
        as_of_utc=as_of,
        row_count=meta_json["row_count"],
        notes=meta_json["notes"],
    )

    updated_raw_paths = dict(state.raw_data_paths)
    updated_raw_paths["price_ohlcv"] = str(out_csv)

    updated_extraction_meta = dict(state.extraction_metadata)
    updated_extraction_meta["yahoo"] = yahoo_meta

    return {
        "raw_data_paths": updated_raw_paths,
        "extraction_metadata": updated_extraction_meta,
        "has_price_data": True,
    }