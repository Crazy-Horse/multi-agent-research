from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


Severity = Literal["fatal", "non_fatal"]

class ExtractionMeta(BaseModel):
    """Reproducibility metadata for a data extraction step."""
    model_config = ConfigDict(extra="allow")

    provider: str
    dataset_id: Optional[str] = None
    symbols: Optional[List[str]] = None
    start_date: str
    end_date: str
    as_of_utc: datetime
    row_count: Optional[int] = None
    notes: Optional[str] = None


class TargetDefinition(BaseModel):
    """Explicit definition of the ML regression target."""
    name: str = "target_price_t_plus_14"
    source_column: str = "front_month_price"
    shift_days: int = 14
    alignment_rule: str = "calendar_day_forward_join"
    null_policy: str = "null_for_most_recent_shift_window"


class ErrorRecord(BaseModel):
    """Structured error reporting for node failures and partial failures."""
    node: str
    message: str
    severity: Severity = "fatal"
    suggested_action: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class CommodityResearchState(BaseModel):
    """
    LangGraph shared state for commodity research + extraction + feature spec + dataset build.

    Notes:
    - Keep DataFrames out of state for durability; store file paths + summaries instead.
    - Availability flags are derived/mutable, and help conditional routing.
    """
    model_config = ConfigDict(extra="allow")

    # -------------------------------------------------------------------------
    # A) Run Configuration (inputs)
    # -------------------------------------------------------------------------
    topic: str
    canonical_ticker: str

    start_date: str
    end_date: str
    target_horizon_days: int = 14

    weather_locations: list[dict] = Field(default_factory=list)

    enable_research: bool = True
    enable_feature_spec: bool = True
    enable_exogenous: bool = True

    # -------------------------------------------------------------------------
    # B) Availability Flags (mutable / derived)
    # -------------------------------------------------------------------------
    has_price_data: bool = False
    has_macro_data: bool = False
    has_fx_data: bool = False
    has_weather_data: bool = False
    has_positioning_data: bool = False

    # -------------------------------------------------------------------------
    # C) Raw Data Artifacts
    # -------------------------------------------------------------------------
    raw_data_paths: Dict[str, str] = Field(default_factory=dict)
    # Example keys: "price_ohlcv", "macro", "fx", "weather", "positioning"

    # -------------------------------------------------------------------------
    # D) Extraction Metadata
    # -------------------------------------------------------------------------
    extraction_metadata: Dict[str, ExtractionMeta] = Field(default_factory=dict)
    # Keyed by provider/source name: "yahoo", "fred", "open_meteo", "cot", etc.

    # -------------------------------------------------------------------------
    # E) Research Outputs (optional)
    # -------------------------------------------------------------------------
    research_brief_markdown: Optional[str] = None
    research_sources: List[str] = Field(default_factory=list)

    # -------------------------------------------------------------------------
    # F) Feature Specification Outputs (optional)
    # -------------------------------------------------------------------------
    feature_spec_markdown: Optional[str] = None
    target_definition: TargetDefinition = Field(default_factory=TargetDefinition)

    # -------------------------------------------------------------------------
    # G) Processed / Final Dataset Artifacts
    # -------------------------------------------------------------------------
    processed_data_paths: Dict[str, str] = Field(default_factory=dict)
    # Example keys: "features_daily", "data_dictionary", "continuous_series", etc.

    row_counts: Dict[str, int] = Field(default_factory=dict)
    # Example keys: "raw_rows", "final_rows", "dropped_rows"

    # -------------------------------------------------------------------------
    # H) Validation Results
    # -------------------------------------------------------------------------
    validation_checks: Dict[str, bool] = Field(default_factory=dict)
    # Example keys: "no_duplicate_dates", "numeric_cast_success", "iso_dates", ...

    null_rate_summary: Dict[str, float] = Field(default_factory=dict)
    # Column -> % missing (0..100)

    # -------------------------------------------------------------------------
    # I) Errors and Warnings
    # -------------------------------------------------------------------------
    errors: List[ErrorRecord] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # -------------------------------------------------------------------------
    # J) Run Summary
    # -------------------------------------------------------------------------
    run_summary: Optional[str] = None
