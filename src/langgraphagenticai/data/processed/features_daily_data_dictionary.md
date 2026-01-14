# Data Dictionary: features_daily.csv

- commodity: str (coffee)
- date: ISO-8601 date (YYYY-MM-DD)
- front_month_price: float (Yahoo proxy from KC=F close)
- ret_1d: float (pct change, t vs t-1)
- logret_1d: float (log return, t vs t-1)
- vol_10d: float (rolling std of logret_1d over 10 rows)
- vol_20d: float (rolling std of logret_1d over 20 rows)
- atr_14d: float (rolling mean of true range over 14 rows)
- hl_range: float ((high-low)/price)
- oc_range: float ((close-open)/price)
- target_price_t_plus_14: float (front_month_price at t+14 calendar days; blank for last 14 days)

Notes:
- Yahoo Finance provides a continuous proxy series; roll mechanics are provider-defined.
- Features are computed on trading days; target uses calendar-day alignment via date shifting and join.