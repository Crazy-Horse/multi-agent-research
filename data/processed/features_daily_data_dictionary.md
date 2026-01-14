# Data Dictionary: features_daily.csv

- commodity: str (coffee)
- date: ISO-8601 date (YYYY-MM-DD)
- ticker: str (KC=F)
- front_month_price: float (Yahoo proxy from KC=F close)
- ret_1d: float (pct change, t vs t-1)
- logret_1d: float (log return, t vs t-1)
- vol_10d: float (rolling std of logret_1d over 10 rows)
- vol_20d: float (rolling std of logret_1d over 20 rows)
- atr_14d: float (rolling mean of true range over 14 rows)
- hl_range: float ((high-low)/price)
- oc_range: float ((close-open)/price)
- target_price_t_plus_14: float (front_month_price at t+14 calendar days; blank for last 14 days)

## Exogenous macro/FX columns (merged on date)

- brl_usd: float (macro/FX series; source: FRED/World Bank)
- inr_usd: float (macro/FX series; source: FRED/World Bank)
- cny_usd: float (macro/FX series; source: FRED/World Bank)
- thb_usd: float (macro/FX series; source: FRED/World Bank)
- mxn_usd: float (macro/FX series; source: FRED/World Bank)
- aud_usd: float (macro/FX series; source: FRED/World Bank)
- eur_usd: float (macro/FX series; source: FRED/World Bank)
- zar_usd: float (macro/FX series; source: FRED/World Bank)
- jpy_usd: float (macro/FX series; source: FRED/World Bank)
- chf_usd: float (macro/FX series; source: FRED/World Bank)
- krw_usd: float (macro/FX series; source: FRED/World Bank)
- gbp_usd: float (macro/FX series; source: FRED/World Bank)
- oil_wti: float (macro/FX series; source: FRED/World Bank)
- usd_index: float (macro/FX series; source: FRED/World Bank)
- us_10yr_rate: float (macro/FX series; source: FRED/World Bank)
- vnd_usd: float (macro/FX series; source: FRED/World Bank)
- cop_usd: float (macro/FX series; source: FRED/World Bank)
- idr_usd: float (macro/FX series; source: FRED/World Bank)
- etb_usd: float (macro/FX series; source: FRED/World Bank)
- hnl_usd: float (macro/FX series; source: FRED/World Bank)
- ugx_usd: float (macro/FX series; source: FRED/World Bank)
- pen_usd: float (macro/FX series; source: FRED/World Bank)
- xaf_usd: float (macro/FX series; source: FRED/World Bank)
- gtq_usd: float (macro/FX series; source: FRED/World Bank)
- nio_usd: float (macro/FX series; source: FRED/World Bank)
- crc_usd: float (macro/FX series; source: FRED/World Bank)
- tzs_usd: float (macro/FX series; source: FRED/World Bank)
- kes_usd: float (macro/FX series; source: FRED/World Bank)
- lak_usd: float (macro/FX series; source: FRED/World Bank)
- pkr_usd: float (macro/FX series; source: FRED/World Bank)
- php_usd: float (macro/FX series; source: FRED/World Bank)
- egp_usd: float (macro/FX series; source: FRED/World Bank)
- ars_usd: float (macro/FX series; source: FRED/World Bank)
- rub_usd: float (macro/FX series; source: FRED/World Bank)
- try_usd: float (macro/FX series; source: FRED/World Bank)
- uah_usd: float (macro/FX series; source: FRED/World Bank)
- irr_usd: float (macro/FX series; source: FRED/World Bank)
- byn_usd: float (macro/FX series; source: FRED/World Bank)

## Exogenous weather columns (multi-region Open-Meteo)

- Weather raw data is per (location, date). It is aggregated to date-level features.
- Equal-weight aggregates are provided as wx_<var>_{mean,std,min,max}.
- Production-weighted aggregates are provided as wx_<var>_wavg (weights configurable).
- Rolling windows add *_mean_7d/14d/30d (or *_sum_7d/14d/30d for precipitation-like vars).
- Explicit lags add *_lag_7d/14d/30d for leakage-safe modeling.

- weather_feature_count: 216 (columns prefixed with wx_)

## Data Quality Rules

- Exogenous features are merged on date and then forward-filled; initial gaps may be back-filled.
- Exogenous columns with >40% missingness are dropped automatically to keep the dataset ML-ready.
- Target is never imputed and is forced blank for the most recent horizon window.

Notes:
- Yahoo Finance provides a continuous proxy series; roll mechanics are provider-defined.
- Features are computed on trading days; target uses calendar-day alignment via date shifting and join.
- Missing values are blank in CSV output.