# src/langgraphagenticai/main.py
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import argparse

from langgraphagenticai.graph import build_graph
from langgraphagenticai.state import CommodityResearchState


def run() -> int:
    parser = argparse.ArgumentParser(description="Run commodity feature pipeline (Yahoo -> Build -> Validate).")
    parser.add_argument("--topic", default="coffee")
    parser.add_argument("--ticker", default="KC=F")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--enable-exogenous", action="store_true")
    parser.add_argument("--weather-6-regions", action="store_true")

    args = parser.parse_args()

    app = build_graph()

    weather_locations = []
    if args.weather_6_regions:
        weather_locations = [
            {"name": "Brazil_MinasGerais", "latitude": -18.5, "longitude": -44.6},
            {"name": "Brazil_EspiritoSanto", "latitude": -19.5, "longitude": -40.6},
            {"name": "Vietnam_CentralHighlands", "latitude": 12.7, "longitude": 108.1},
            {"name": "Colombia_Huila", "latitude": 2.5, "longitude": -75.6},
            {"name": "Indonesia_Lampung", "latitude": -5.4, "longitude": 105.3},
            {"name": "Ethiopia_Oromia", "latitude": 8.5, "longitude": 39.0},
        ]

    state = CommodityResearchState(
        topic=args.topic,
        canonical_ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        target_horizon_days=args.horizon,
        enable_research=False,
        enable_feature_spec=False,
        enable_exogenous=bool(args.enable_exogenous),
        weather_locations=weather_locations,
    )

    result = app.invoke(state)

    # LangGraph often returns dict-like state updates; normalize to object access if needed
    run_summary = result.get("run_summary") if isinstance(result, dict) else getattr(result, "run_summary", None)
    errors = result.get("errors") if isinstance(result, dict) else getattr(result, "errors", [])

    if run_summary:
        print(run_summary)
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"- {e}")

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(run())