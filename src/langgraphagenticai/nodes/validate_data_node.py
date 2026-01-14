from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# If your ErrorRecord is in langgraphagenticai.state (or similar), import it.
# Adjust the import path if needed.
from langgraphagenticai.state.commodity_state import ErrorRecord


def _as_error_record(e: Any) -> ErrorRecord:
    """
    Normalize any error-like object into an ErrorRecord.

    Supports:
      - ErrorRecord (already)
      - dict with keys: node, message, severity, suggested_action, details
      - str (treated as message, node='unknown')
      - any object with .model_dump() (Pydantic model) or attributes
    """
    if isinstance(e, ErrorRecord):
        return e

    # dict-style error
    if isinstance(e, dict):
        return ErrorRecord(
            node=str(e.get("node", "unknown")),
            message=str(e.get("message", "")),
            severity=e.get("severity", "fatal"),
            suggested_action=e.get("suggested_action"),
            details=e.get("details"),
        )

    # pydantic-like
    if hasattr(e, "model_dump"):
        d = e.model_dump()
        return ErrorRecord(
            node=str(d.get("node", "unknown")),
            message=str(d.get("message", "")),
            severity=d.get("severity", "fatal"),
            suggested_action=d.get("suggested_action"),
            details=d.get("details"),
        )

    # attribute-like
    node = getattr(e, "node", "unknown")
    msg = getattr(e, "message", None)
    if msg is None:
        # plain string or unknown object
        return ErrorRecord(node=str(node), message=str(e), severity="fatal")

    return ErrorRecord(
        node=str(node),
        message=str(msg),
        severity=getattr(e, "severity", "fatal"),
        suggested_action=getattr(e, "suggested_action", None),
        details=getattr(e, "details", None),
    )


def _append_error(errors: List[ErrorRecord], *, node: str, message: str, severity: str = "fatal", suggested_action: str | None = None, details: Dict[str, Any] | None = None) -> None:
    errors.append(
        ErrorRecord(
            node=node,
            message=message,
            severity=severity,  # type: ignore[arg-type]
            suggested_action=suggested_action,
            details=details,
        )
    )


def validate_node(state):
    """
    Deterministic node: validates features_daily.csv and writes summary into state.

    Standardizes:
      - state.errors -> List[ErrorRecord]
      - returned 'errors' -> List[ErrorRecord]
    """
    # Normalize inbound errors to ErrorRecord
    errors: List[ErrorRecord] = [_as_error_record(e) for e in list(getattr(state, "errors", []))]
    warnings: List[str] = list(getattr(state, "warnings", []))

    processed = getattr(state, "processed_data_paths", {}) or {}
    if "features_daily" not in processed:
        _append_error(
            errors,
            node="validate_node",
            message="Missing features_daily output; cannot validate.",
            severity="fatal",
            suggested_action="Ensure build_features_node ran successfully.",
        )
        return {
            "validation_checks": dict(getattr(state, "validation_checks", {}) or {}),
            "null_rate_summary": dict(getattr(state, "null_rate_summary", {}) or {}),
            "warnings": warnings,
            "errors": errors,
            "run_summary": "\n".join(
                [
                    "# Run Summary",
                    f"- topic: {getattr(state, 'topic', '')}",
                    f"- ticker: {getattr(state, 'canonical_ticker', '')}",
                    f"- date range: {getattr(state, 'start_date', '')} to {getattr(state, 'end_date', '')}",
                    "",
                    "## Validation Checks",
                    "- features_daily_present: False",
                    "",
                    "## Errors",
                    f"- validate_node: Missing features_daily output; cannot validate.",
                ]
            ),
        }

    path = Path(processed["features_daily"])
    df = pd.read_csv(path)

    checks: Dict[str, bool] = {}

    # ISO date check
    try:
        pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
        checks["iso_dates"] = True
    except Exception:
        checks["iso_dates"] = False
        _append_error(
            errors,
            node="validate_node",
            message="Date column is not ISO-8601 (YYYY-MM-DD) parseable.",
            severity="fatal",
            suggested_action="Ensure build_features_node writes df['date'] as YYYY-MM-DD strings.",
        )

    # No duplicates per (commodity, date)
    if "commodity" in df.columns and "date" in df.columns:
        dup = df.duplicated(subset=["commodity", "date"]).any()
        checks["no_duplicate_dates"] = (not dup)
        if dup:
            _append_error(
                errors,
                node="validate_node",
                message="Duplicate rows detected for (commodity, date).",
                severity="fatal",
                suggested_action="Drop duplicates and ensure deterministic aggregation before writing CSV.",
            )
    else:
        checks["no_duplicate_dates"] = False
        _append_error(
            errors,
            node="validate_node",
            message="Missing commodity/date columns needed for dedupe validation.",
            severity="fatal",
            suggested_action="Ensure build_features_node outputs commodity and date columns.",
        )

    # Numeric float-castable check (exclude known string columns)
    non_numeric = {"commodity", "ticker", "date"}
    numeric_cols = [c for c in df.columns if c not in non_numeric]
    cast_ok = True
    for c in numeric_cols:
        try:
            pd.to_numeric(df[c], errors="raise")
        except Exception:
            cast_ok = False
            _append_error(
                errors,
                node="validate_node",
                message=f"Column '{c}' is not fully float-castable.",
                severity="fatal",
                suggested_action="Ensure numeric columns are written as numbers or blanks (not strings).",
                details={"column": c},
            )
    checks["numeric_cast_success"] = cast_ok

    # Target null tail check
    horizon = int(getattr(state, "target_horizon_days", 14))
    if "target_price_t_plus_14" in df.columns:
        df["_date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        max_date = df["_date_dt"].max()
        cutoff = max_date - pd.Timedelta(days=horizon)
        tail = df[df["_date_dt"] > cutoff]
        tail_null_ok = tail["target_price_t_plus_14"].isna().all()
        checks["target_null_tail_ok"] = bool(tail_null_ok)
        if not tail_null_ok:
            warnings.append("target_price_t_plus_14 is not null for all rows in the most recent horizon window.")
    else:
        checks["target_null_tail_ok"] = False
        warnings.append("target_price_t_plus_14 missing; target validation skipped.")

    # Null rate summary
    null_rates: Dict[str, float] = {}
    for c in df.columns:
        null_rates[c] = float(df[c].isna().mean() * 100.0)

    # Human-readable summary
    summary_lines = [
        "# Run Summary",
        f"- topic: {getattr(state, 'topic', '')}",
        f"- ticker: {getattr(state, 'canonical_ticker', '')}",
        f"- date range: {getattr(state, 'start_date', '')} to {getattr(state, 'end_date', '')}",
        "",
        "## Validation Checks",
    ]
    for k, v in checks.items():
        summary_lines.append(f"- {k}: {v}")

    if warnings:
        summary_lines.append("")
        summary_lines.append("## Warnings")
        summary_lines.extend([f"- {w}" for w in warnings])

    if errors:
        summary_lines.append("")
        summary_lines.append("## Errors")
        for e in errors:
            summary_lines.append(f"- {e.node}: {e.message}")

    # Merge checks into existing state dicts
    merged_checks = dict(getattr(state, "validation_checks", {}) or {})
    merged_checks.update(checks)

    merged_null_rates = dict(getattr(state, "null_rate_summary", {}) or {})
    merged_null_rates.update(null_rates)

    return {
        "validation_checks": merged_checks,
        "null_rate_summary": merged_null_rates,
        "warnings": warnings,
        "errors": errors,  # <-- always List[ErrorRecord]
        "run_summary": "\n".join(summary_lines),
    }
