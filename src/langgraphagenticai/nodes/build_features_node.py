from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence, Iterable

import pandas as pd
import numpy as np

from langgraphagenticai.nodes.yahoo_data_extraction_node import _ensure_dir

WX_SCHEMA_COLUMNS: list[str] = [
    # Base
    "wx_location_count",
    "wx_temperature_2m_max_wavg",
    "wx_temperature_2m_min_wavg",
    "wx_precipitation_sum_wavg",
    "wx_wind_speed_10m_max_wavg",

    # Rollups
    "wx_temperature_2m_max_wavg_mean_7d",
    "wx_temperature_2m_max_wavg_mean_14d",
    "wx_temperature_2m_max_wavg_mean_30d",

    "wx_temperature_2m_min_wavg_mean_7d",
    "wx_temperature_2m_min_wavg_mean_14d",
    "wx_temperature_2m_min_wavg_mean_30d",

    "wx_precipitation_sum_wavg_sum_7d",
    "wx_precipitation_sum_wavg_sum_14d",
    "wx_precipitation_sum_wavg_sum_30d",

    "wx_wind_speed_10m_max_wavg_mean_7d",
    "wx_wind_speed_10m_max_wavg_mean_14d",
    "wx_wind_speed_10m_max_wavg_mean_30d",

    # Lags
    "wx_location_count_lag_7d",
    "wx_location_count_lag_14d",

    "wx_temperature_2m_max_wavg_lag_7d",
    "wx_temperature_2m_max_wavg_lag_14d",

    "wx_temperature_2m_min_wavg_lag_7d",
    "wx_temperature_2m_min_wavg_lag_14d",

    "wx_precipitation_sum_wavg_lag_7d",
    "wx_precipitation_sum_wavg_lag_14d",

    "wx_wind_speed_10m_max_wavg_lag_7d",
    "wx_wind_speed_10m_max_wavg_lag_14d",
]

WX_COMPACT_SCHEMA_COLUMNS: list[str] = [
    # Base
    "wx_location_count",
    "wx_temperature_2m_max_wavg",
    "wx_temperature_2m_min_wavg",
    "wx_precipitation_sum_wavg",
    "wx_wind_speed_10m_max_wavg",

    # Rollups
    "wx_temperature_2m_max_wavg_mean_7d",
    "wx_temperature_2m_max_wavg_mean_14d",
    "wx_temperature_2m_max_wavg_mean_30d",

    "wx_temperature_2m_min_wavg_mean_7d",
    "wx_temperature_2m_min_wavg_mean_14d",
    "wx_temperature_2m_min_wavg_mean_30d",

    "wx_precipitation_sum_wavg_sum_7d",
    "wx_precipitation_sum_wavg_sum_14d",
    "wx_precipitation_sum_wavg_sum_30d",

    "wx_wind_speed_10m_max_wavg_mean_7d",
    "wx_wind_speed_10m_max_wavg_mean_14d",
    "wx_wind_speed_10m_max_wavg_mean_30d",

    # Lags (7/14 only)
    "wx_location_count_lag_7d",
    "wx_location_count_lag_14d",

    "wx_temperature_2m_max_wavg_lag_7d",
    "wx_temperature_2m_max_wavg_lag_14d",

    "wx_temperature_2m_min_wavg_lag_7d",
    "wx_temperature_2m_min_wavg_lag_14d",

    "wx_precipitation_sum_wavg_lag_7d",
    "wx_precipitation_sum_wavg_lag_14d",

    "wx_wind_speed_10m_max_wavg_lag_7d",
    "wx_wind_speed_10m_max_wavg_lag_14d",
]


def enforce_compact_weather_schema(
    df: pd.DataFrame,
    *,
    keep_weather: bool = True,
    wx_schema: Sequence[str] = tuple(WX_COMPACT_SCHEMA_COLUMNS),
    extra_keep: Iterable[str] = (),
    reorder: bool = True,
    target_col: str = "target_price_t_plus_14",
    target_last: bool = True,
) -> pd.DataFrame:
    """
    Enforce the *compact* wx schema (7/14 lags only), dropping any other wx_ columns.

    Behavior:
      - If keep_weather=False: drops all wx_ columns
      - Else:
          - keeps only wx_ columns in wx_schema (+ extra_keep)
          - optionally reorders: non-wx columns preserve original order, then wx columns in schema order
          - optionally moves target_col to the end (common for modeling)

    Notes:
      - Missing wx columns are tolerated (silently skipped).
      - Non-wx columns are never dropped by this function.
    """
    out = df.copy()

    wx_cols_present = [c for c in out.columns if c.startswith("wx_")]
    if not wx_cols_present:
        # Still optionally move target last
        if target_last and target_col in out.columns:
            cols = [c for c in out.columns if c != target_col] + [target_col]
            return out.loc[:, cols]
        return out

    if not keep_weather:
        out = out.drop(columns=wx_cols_present)
        if target_last and target_col in out.columns:
            cols = [c for c in out.columns if c != target_col] + [target_col]
            out = out.loc[:, cols]
        return out

    allow = set(wx_schema) | set(extra_keep)

    # Drop any wx_ not explicitly allowlisted
    drop_cols = [c for c in wx_cols_present if c not in allow]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    if reorder:
        non_wx = [c for c in out.columns if not c.startswith("wx_") and c != target_col]
        wx_ordered = [c for c in wx_schema if c in out.columns]
        # deterministic extras (wx_ kept via extra_keep but not in wx_schema)
        wx_extras = sorted([c for c in out.columns if c.startswith("wx_") and c not in wx_ordered])

        cols = non_wx + wx_ordered + wx_extras

        # Put target at end if requested
        if target_col in out.columns:
            if target_last:
                cols = cols + [target_col]
            else:
                cols = [target_col] + cols

        out = out.loc[:, cols]

    return out

def enforce_exact_weather_schema(
    df: pd.DataFrame,
    *,
    keep_weather: bool = True,
    reorder: bool = True,
    weather_cols: Sequence[str] = tuple(WX_SCHEMA_COLUMNS),
    extra_keep: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Enforce the exact `wx_` weather schema you provided.

    Behavior:
      - If keep_weather=False: drop ALL columns that start with 'wx_'
      - Else:
          - drop any 'wx_' columns not in weather_cols (plus extra_keep)
          - optionally reorder to place wx columns in the schema order,
            while preserving non-wx column order.

    Notes:
      - Does not drop non-wx columns (price, macro/FX, target, etc.).
      - If a wx column in the schema is missing, it is simply skipped (no error).
    """
    df_out = df.copy()

    wx_cols_present = [c for c in df_out.columns if c.startswith("wx_")]

    if not wx_cols_present:
        return df_out

    if not keep_weather:
        return df_out.drop(columns=wx_cols_present)

    allow = set(weather_cols) | set(extra_keep)

    # Drop unlisted wx_ columns
    drop_cols = [c for c in wx_cols_present if c not in allow]
    if drop_cols:
        df_out = df_out.drop(columns=drop_cols)

    if not reorder:
        return df_out

    # Reorder: keep non-wx columns in original order, then wx columns in schema order
    non_wx = [c for c in df_out.columns if not c.startswith("wx_")]
    wx_ordered = [c for c in weather_cols if c in df_out.columns]  # only those that exist
    # plus any extra_keep wx columns not in schema, appended deterministically
    wx_extras = sorted([c for c in df_out.columns if c.startswith("wx_") and c not in wx_ordered])

    final_cols = non_wx + wx_ordered + wx_extras
    return df_out.loc[:, final_cols]

def _true_range(high, low, prev_close):
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _wx_allowlist(
    *,
    windows: Sequence[int] = (7, 14, 30),
    lags: Sequence[int] = (7, 14),
) -> set[str]:
    """
    Linear-regression weather schema allowlist.
    Keeps:
      - base weighted aggregates
      - rolling means (temp/wind)
      - rolling sums (precip)
      - explicit lags for base signals only
    """
    allow = set()

    # Base (date-level) weighted aggregates
    base = [
        "wx_temperature_2m_max_wavg",
        "wx_temperature_2m_min_wavg",
        "wx_precipitation_sum_wavg",
        "wx_wind_speed_10m_max_wavg",
        "wx_location_count",
        # Optional dispersion signals (uncomment if you want them)
        # "wx_temperature_2m_max_std",
        # "wx_precipitation_sum_std",
    ]
    allow.update(base)

    # Rolling: temp/wind -> mean; precip -> sum
    for w in windows:
        allow.add(f"wx_temperature_2m_max_wavg_mean_{w}d")
        allow.add(f"wx_temperature_2m_min_wavg_mean_{w}d")
        allow.add(f"wx_wind_speed_10m_max_wavg_mean_{w}d")
        allow.add(f"wx_precipitation_sum_wavg_sum_{w}d")

    # Lags: base signals only (NOT rolling outputs)
    for lag in lags:
        allow.add(f"wx_temperature_2m_max_wavg_lag_{lag}d")
        allow.add(f"wx_temperature_2m_min_wavg_lag_{lag}d")
        allow.add(f"wx_wind_speed_10m_max_wavg_lag_{lag}d")
        allow.add(f"wx_precipitation_sum_wavg_lag_{lag}d")
        allow.add(f"wx_location_count_lag_{lag}d")

    return allow


def enforce_linear_weather_schema(
    df: pd.DataFrame,
    *,
    keep_weather: bool = True,
    windows: Sequence[int] = (7, 14, 30),
    lags: Sequence[int] = (7, 14),
    drop_unlisted_wx: bool = True,
    extra_keep: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Enforce a compact weather feature schema for linear regression.

    - Keeps all non-`wx_` columns as-is.
    - If keep_weather=True:
        keeps only allowlisted `wx_` columns (+ extra_keep).
      Else:
        removes all `wx_` columns.

    Args:
      df: feature dataframe
      keep_weather: whether to keep any weather features at all
      windows: rolling windows to allow
      lags: lag horizons to allow
      drop_unlisted_wx: if True, drops any `wx_` column not allowlisted
      extra_keep: additional columns to retain even if they start with wx_

    Returns:
      A new dataframe with filtered columns, preserving original order where possible.
    """
    df_out = df.copy()

    wx_cols = [c for c in df_out.columns if c.startswith("wx_")]
    if not wx_cols:
        return df_out  # nothing to do

    if not keep_weather:
        return df_out.drop(columns=wx_cols)

    allow = _wx_allowlist(windows=windows, lags=lags)
    allow.update(extra_keep)

    if drop_unlisted_wx:
        drop_cols = [c for c in wx_cols if c not in allow]
        if drop_cols:
            df_out = df_out.drop(columns=drop_cols)

    # Optional: ensure we didn't accidentally keep forbidden “lag of rolling” patterns
    # (defensive guardrail if upstream naming changes)
    forbidden = re.compile(r"_mean_\d+d_lag_\d+d|_sum_\d+d_lag_\d+d")
    bad = [c for c in df_out.columns if c.startswith("wx_") and forbidden.search(c)]
    if bad:
        df_out = df_out.drop(columns=bad)

    return df_out

def _apply_exogenous_missingness_policy(
    df: pd.DataFrame,
    *,
    exogenous_cols: list[str],
    protected_cols: set[str],
    max_missing_pct: float = 40.0,
    do_backfill: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Deterministic:
      - drops exogenous columns with missing% > threshold
      - forward-fills (and optional back-fills) remaining exogenous columns
    Returns:
      (df, summary) where summary includes dropped cols + missingness snapshot.
    """
    summary = {"dropped_exogenous_cols": [], "exogenous_missing_pct": {}}
    exogenous_cols = [c for c in exogenous_cols if c in df.columns]
    if not exogenous_cols:
        return df, summary

    # Compute missingness on exogenous columns only
    miss = (df[exogenous_cols].isna().mean() * 100.0).to_dict()
    summary["exogenous_missing_pct"] = {k: float(v) for k, v in miss.items()}

    # Drop columns above threshold (but never protected)
    to_drop = [
        c for c in exogenous_cols
        if c not in protected_cols and miss.get(c, 0.0) > max_missing_pct
    ]
    if to_drop:
        df = df.drop(columns=to_drop)
        summary["dropped_exogenous_cols"] = to_drop

    remaining = [c for c in exogenous_cols if c in df.columns and c not in protected_cols]
    if remaining:
        # If you ever run multiple commodities, this is safer:
        if "commodity" in df.columns:
            df[remaining] = df.groupby("commodity", sort=False)[remaining].ffill()
            if do_backfill:
                df[remaining] = df.groupby("commodity", sort=False)[remaining].bfill()
        else:
            df[remaining] = df[remaining].ffill()
            if do_backfill:
                df[remaining] = df[remaining].bfill()

    return df, summary

def build_features_node(state):
    """
    Deterministic node (drop-in replacement):
      - reads raw OHLCV (Yahoo)
      - computes market-derived features
      - optionally merges macro/FX daily data (FRED + World Bank)
      - optionally merges multi-region weather (Open-Meteo) and engineers:
          * equal-weight daily aggregates (mean/std/min/max)
          * production-weighted daily aggregates (wavg)
          * rolling weather features (7/14/30)
          * explicit weather lags (7/14/30)
      - applies deterministic exogenous missingness policy (drop >40% missing, ffill/bfill)
      - creates target_price_t_plus_14
      - writes:
          data/processed/features_daily.csv
          data/processed/features_daily_data_dictionary.md
    """

    # ----------------------------
    # Local deterministic helpers (kept inside for drop-in portability)
    # ----------------------------
    def _weighted_weather_aggregate(weather_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
        """Return one row per date with per-variable wavg + location_count."""
        if weather_df.empty:
            return pd.DataFrame()

        w = weather_df.copy()
        w["weight"] = w["location"].map(weights).fillna(0.0)

        base_vars = [c for c in w.columns if c not in {"date", "location", "weight"}]
        out_parts = []

        # Weighted average per date for each base var
        for col in base_vars:
            # If all weights are zero on a date, result will be 0.0; fix to NaN via denom check.
            num = (w[col] * w["weight"]).groupby(w["date"]).sum()
            den = w["weight"].groupby(w["date"]).sum()
            wavg = (num / den.replace(0.0, np.nan)).rename(f"wx_{col}_wavg")
            out_parts.append(wavg)

        out = pd.concat(out_parts, axis=1).reset_index()
        out["wx_location_count"] = w.groupby("date")["location"].nunique().values
        return out

    def _equal_weight_weather_stats(weather_df: pd.DataFrame) -> pd.DataFrame:
        """Return one row per date with mean/std/min/max across locations for each base var."""
        if weather_df.empty:
            return pd.DataFrame()

        base_vars = [c for c in weather_df.columns if c not in {"date", "location"}]
        g = weather_df.groupby("date", sort=True)

        out = pd.DataFrame({"date": sorted(g.groups.keys())})
        out = out.set_index("date")

        for col in base_vars:
            s = g[col]
            out[f"wx_{col}_mean"] = s.mean()
            out[f"wx_{col}_std"] = s.std(ddof=0)
            out[f"wx_{col}_min"] = s.min()
            out[f"wx_{col}_max"] = s.max()

        out = out.reset_index()
        out["wx_location_count"] = g["location"].nunique().values
        return out

    def _add_weather_rollups(df_in: pd.DataFrame, weather_cols: list[str], windows: tuple[int, ...]) -> pd.DataFrame:
        roll_cols = {}

        for col in weather_cols:
            for w in windows:
                if "precipitation" in col or "precip" in col:
                    roll_cols[f"{col}_sum_{w}d"] = df_in[col].rolling(w, min_periods=1).sum()
                else:
                    roll_cols[f"{col}_mean_{w}d"] = df_in[col].rolling(w, min_periods=1).mean()

        if not roll_cols:
            return df_in

        roll_df = pd.DataFrame(roll_cols, index=df_in.index)
        return pd.concat([df_in, roll_df], axis=1)

    def _add_weather_lags(df_in: pd.DataFrame, weather_cols: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
        lagged_cols = {}
        for col in weather_cols:
            for lag in lags:
                lagged_cols[f"{col}_lag_{lag}d"] = df_in[col].shift(lag)

        if not lagged_cols:
            return df_in

        lag_df = pd.DataFrame(lagged_cols, index=df_in.index)
        return pd.concat([df_in, lag_df], axis=1)

    # ----------------------------
    # Guardrails
    # ----------------------------
    if not state.has_price_data or "price_ohlcv" not in state.raw_data_paths:
        return {
            "errors": list(getattr(state, "errors", [])) + [{
                "node": "build_features_node",
                "message": "Missing price OHLCV input; cannot build features.",
                "severity": "fatal",
                "suggested_action": "Ensure yahoo_extract_node succeeded and raw_data_paths['price_ohlcv'] exists."
            }]
        }

    warnings = list(getattr(state, "warnings", []))
    raw_path = Path(state.raw_data_paths["price_ohlcv"])

    # Keep original raw row count separate from processed df
    try:
        raw_rows_count = int(_safe_read_csv(raw_path).shape[0])
    except Exception:
        raw_rows_count = 0

    df = _safe_read_csv(raw_path)

    # ----------------------------
    # Parse + coerce OHLCV
    # ----------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # Ensure commodity always exists early (useful for groupwise fills)
    df["commodity"] = state.topic

    # Canonical series for ML
    df["front_month_price"] = df["close"].astype(float)

    # Market features
    df["ret_1d"] = df["front_month_price"].pct_change()
    df["logret_1d"] = np.log(df["front_month_price"]).diff()
    df["hl_range"] = (df["high"] - df["low"]) / df["front_month_price"]
    df["oc_range"] = (df["close"] - df["open"]) / df["front_month_price"]
    df["vol_10d"] = df["logret_1d"].rolling(10).std()
    df["vol_20d"] = df["logret_1d"].rolling(20).std()

    prev_close = df["close"].shift(1)
    tr = _true_range(df["high"].astype(float), df["low"].astype(float), prev_close.astype(float))
    df["atr_14d"] = pd.Series(tr, index=df.index).rolling(14).mean()

    # ----------------------------
    # Optional merge: macro/FX daily (FRED + World Bank)
    # ----------------------------
    merged_exogenous_cols: list[str] = []

    if getattr(state, "has_macro_data", False) and "macro_fx" in state.raw_data_paths:
        macro_path = Path(state.raw_data_paths["macro_fx"])
        if macro_path.exists():
            macro = _safe_read_csv(macro_path)
            macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
            macro = macro.dropna(subset=["date"]).sort_values("date")

            for c in macro.columns:
                if c in {"date", "commodity"}:
                    continue
                macro[c] = pd.to_numeric(macro[c], errors="coerce")

            if "commodity" in macro.columns:
                macro["commodity"] = macro["commodity"].astype(str)
                df["commodity"] = df["commodity"].astype(str)
                df = df.merge(macro, on=["commodity", "date"], how="left", suffixes=("", "_macro"))
            else:
                df = df.merge(macro, on="date", how="left", suffixes=("", "_macro"))

            merged_exogenous_cols.extend([c for c in macro.columns if c not in {"date", "commodity"}])
        else:
            warnings.append(f"macro_fx path not found: {str(macro_path)}")

    # ----------------------------
    # Optional merge: multi-region weather (Open-Meteo)
    #   Expect raw weather file columns: date, location, <daily vars...>
    #   Engineer:
    #     - equal-weight stats per date: wx_<var>_{mean,std,min,max}
    #     - weighted average per date: wx_<var>_wavg
    #     - rolling windows on a selected subset
    #     - explicit lags on the selected subset
    # ----------------------------
    weather_engineered_cols: list[str] = []

    if getattr(state, "has_weather_data", False) and "weather" in state.raw_data_paths:
        weather_path = Path(state.raw_data_paths["weather"])
        if weather_path.exists():
            wdf = _safe_read_csv(weather_path)

            # Normalize weather schema
            if "date" not in wdf.columns or "location" not in wdf.columns:
                warnings.append(
                    f"weather file missing required columns (date, location). Columns={list(wdf.columns)}"
                )
            else:
                wdf["date"] = pd.to_datetime(wdf["date"], errors="coerce")
                wdf = wdf.dropna(subset=["date", "location"]).sort_values(["location", "date"]).reset_index(drop=True)

                # Coerce base vars numeric
                for c in wdf.columns:
                    if c in {"date", "location"}:
                        continue
                    wdf[c] = pd.to_numeric(wdf[c], errors="coerce")

                # Default weights (you can override by setting state.weather_region_weights)
                default_weights = {
                    "Brazil_MinasGerais": 0.35,
                    "Brazil_EspiritoSanto": 0.15,
                    "Vietnam_CentralHighlands": 0.20,
                    "Colombia_Huila": 0.15,
                    "Indonesia_Lampung": 0.10,
                    "Ethiopia_Oromia": 0.05,
                }
                weights = getattr(state, "weather_region_weights", None) or default_weights

                # Equal-weight daily stats
                wx_stats = _equal_weight_weather_stats(wdf)

                # Weighted daily averages
                wx_wavg = _weighted_weather_aggregate(wdf, weights=weights)

                # Merge stats + wavg (avoid duplicate wx_location_count by prefer stats then overwrite if needed)
                wx = wx_stats.merge(wx_wavg, on="date", how="outer", suffixes=("", "_w"))
                if "wx_location_count_w" in wx.columns:
                    wx["wx_location_count"] = wx["wx_location_count"].fillna(wx["wx_location_count_w"])
                    wx = wx.drop(columns=["wx_location_count_w"])

                # Merge into main df
                df = df.merge(wx, on="date", how="left")

                # Identify engineered weather columns (everything we just added except date)
                weather_engineered_cols = [c for c in wx.columns if c != "date"]

                # Rolling + lag ONLY on stable aggregates (means + wavg + precip sums)
                # Keep the set tight to avoid unnecessary feature explosion.
                base_for_roll_lag = [
                    c for c in weather_engineered_cols
                    if (
                        c.endswith("_mean") or c.endswith("_wavg")
                        or "precipitation_sum" in c  # includes mean/min/max/std but we’ll handle by name below
                    )
                ]

                # Add rolling windows
                df = _add_weather_rollups(df, weather_cols=base_for_roll_lag, windows=(7, 14, 30))

                # Add explicit lags
                roll_lag_cols = [c for c in df.columns if c.startswith("wx_")]
                df = _add_weather_lags(df, weather_cols=roll_lag_cols, lags=(7, 14, 30))

                # Track new columns added for dictionary / missingness policy
                weather_engineered_cols = [c for c in df.columns if c.startswith("wx_")]
        else:
            warnings.append(f"weather path not found: {str(weather_path)}")

    # Combine exogenous cols list (macro + weather)
    merged_exogenous_cols = list(dict.fromkeys(merged_exogenous_cols + weather_engineered_cols))

    # ----------------------------
    # Exogenous missingness policy (drop + impute)
    # ----------------------------
    protected_cols = {
        # identity
        "commodity", "ticker", "date",
        # price + engineered market features
        "front_month_price", "ret_1d", "logret_1d", "vol_10d", "vol_20d",
        "atr_14d", "hl_range", "oc_range",
        # target
        "target_price_t_plus_14",
        # raw OHLCV
        "open", "high", "low", "close", "volume", "adj_close",
    }

    df, exo_policy_summary = _apply_exogenous_missingness_policy(
        df,
        exogenous_cols=merged_exogenous_cols,
        protected_cols=protected_cols,
        max_missing_pct=40.0,
        do_backfill=True,
    )

    if exo_policy_summary.get("dropped_exogenous_cols"):
        dropped = exo_policy_summary["dropped_exogenous_cols"]
        warnings.append(
            f"Dropped {len(dropped)} exogenous columns with >40% missingness: "
            + ", ".join(dropped[:10])
            + ("..." if len(dropped) > 10 else "")
        )

    # Keep only exogenous columns that survived
    merged_exogenous_cols = [c for c in merged_exogenous_cols if c in df.columns]

    # ----------------------------
    # Target: front_month_price shifted by +N calendar days
    # ----------------------------
    horizon = int(state.target_horizon_days)
    df_key = df[["date", "front_month_price"]].copy()
    df_key["date_shifted"] = df_key["date"] - pd.Timedelta(days=horizon)

    df = df.merge(
        df_key[["date_shifted", "front_month_price"]].rename(
            columns={"date_shifted": "date", "front_month_price": "target_price_t_plus_14"}
        ),
        on="date",
        how="left",
    )

    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=horizon)
    df.loc[df["date"] > cutoff, "target_price_t_plus_14"] = np.nan

    # df = enforce_linear_weather_schema(
    #     df,
    #     keep_weather=True,
    #     windows=(7, 14),
    #     lags=(7, 14),
    #     drop_unlisted_wx=True,
    # )
    #df = enforce_exact_weather_schema(df, keep_weather=True, reorder=True)
    df = enforce_compact_weather_schema(df, keep_weather=True, reorder=True, target_last=True)

    # Final shaping
    df["ticker"] = state.canonical_ticker
    df["date"] = df["date"].dt.date.astype(str)

    # ----------------------------
    # Write outputs
    # ----------------------------
    out_csv = Path("data/processed/features_daily.csv")
    out_dd = Path("data/processed/features_daily_data_dictionary.md")
    _ensure_dir(out_csv.parent)

    df.to_csv(out_csv, index=False, na_rep="")

    # Data dictionary
    lines = [
        "# Data Dictionary: features_daily.csv",
        "",
        f"- commodity: str ({state.topic})",
        "- date: ISO-8601 date (YYYY-MM-DD)",
        f"- ticker: str ({state.canonical_ticker})",
        f"- front_month_price: float (Yahoo proxy from {state.canonical_ticker} close)",
        "- ret_1d: float (pct change, t vs t-1)",
        "- logret_1d: float (log return, t vs t-1)",
        "- vol_10d: float (rolling std of logret_1d over 10 rows)",
        "- vol_20d: float (rolling std of logret_1d over 20 rows)",
        "- atr_14d: float (rolling mean of true range over 14 rows)",
        "- hl_range: float ((high-low)/price)",
        "- oc_range: float ((close-open)/price)",
        f"- target_price_t_plus_14: float (front_month_price at t+{horizon} calendar days; blank for last {horizon} days)",
        "",
    ]

    macro_cols = [c for c in merged_exogenous_cols if not c.startswith("wx_")]
    wx_cols = [c for c in merged_exogenous_cols if c.startswith("wx_")]

    if macro_cols:
        lines += ["## Exogenous macro/FX columns (merged on date)", ""]
        for c in macro_cols:
            lines.append(f"- {c}: float (macro/FX series; source: FRED/World Bank)")
        lines += [""]

    if wx_cols:
        lines += [
            "## Exogenous weather columns (multi-region Open-Meteo)",
            "",
            "- Weather raw data is per (location, date). It is aggregated to date-level features.",
            "- Equal-weight aggregates are provided as wx_<var>_{mean,std,min,max}.",
            "- Production-weighted aggregates are provided as wx_<var>_wavg (weights configurable).",
            "- Rolling windows add *_mean_7d/14d/30d (or *_sum_7d/14d/30d for precipitation-like vars).",
            "- Explicit lags add *_lag_7d/14d/30d for leakage-safe modeling.",
            "",
        ]
        # Keep dictionary readable (don’t list hundreds of engineered columns exhaustively)
        lines += [f"- weather_feature_count: {len(wx_cols)} (columns prefixed with wx_)", ""]

    lines += [
        "## Data Quality Rules",
        "",
        "- Exogenous features are merged on date and then forward-filled; initial gaps may be back-filled.",
        "- Exogenous columns with >40% missingness are dropped automatically to keep the dataset ML-ready.",
        "- Target is never imputed and is forced blank for the most recent horizon window.",
        "",
        "Notes:",
        "- Yahoo Finance provides a continuous proxy series; roll mechanics are provider-defined.",
        "- Features are computed on trading days; target uses calendar-day alignment via date shifting and join.",
        "- Missing values are blank in CSV output.",
    ]

    out_dd.write_text("\n".join(lines), encoding="utf-8")

    # State updates
    updated_processed = dict(getattr(state, "processed_data_paths", {}))
    updated_processed["features_daily"] = str(out_csv)
    updated_processed["data_dictionary"] = str(out_dd)

    row_counts = dict(getattr(state, "row_counts", {}))
    row_counts["raw_rows"] = int(raw_rows_count)
    row_counts["final_rows"] = int(df.shape[0])

    return {
        "processed_data_paths": updated_processed,
        "row_counts": row_counts,
        "warnings": warnings,
    }

