from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from langgraphagenticai.nodes.yahoo_data_extraction_node import yahoo_data_extraction_node
from langgraphagenticai.nodes.fred_data_extraction_node import fred_data_extraction_node
from langgraphagenticai.nodes.open_meteo_weather_node import open_meteo_weather_node
from langgraphagenticai.nodes.build_features_node import build_features_node
from langgraphagenticai.nodes.validate_data_node import validate_node

from langgraphagenticai.state import CommodityResearchState


def build_graph(
    *,
    name_prefix: str = "",
    include_validate: bool = True,
    return_graph: bool = False,
):
    graph = StateGraph(CommodityResearchState)

    yahoo_node_name = f"{name_prefix}yahoo_extract"
    fred_node_name = f"{name_prefix}fred_extract"
    open_meteo_node_name = f"{name_prefix}open_meteo_extract"
    build_node_name = f"{name_prefix}build_features"
    validate_node_name = f"{name_prefix}validate"

    graph.add_node(yahoo_node_name, yahoo_data_extraction_node)
    graph.add_node(fred_node_name, fred_data_extraction_node)
    graph.add_node(open_meteo_node_name, open_meteo_weather_node)
    graph.add_node(build_node_name, build_features_node)

    if include_validate:
        graph.add_node(validate_node_name, validate_node)

    def route_after_yahoo(state: CommodityResearchState) -> str:
        return fred_node_name if state.enable_exogenous else build_node_name

    graph.add_edge(START, yahoo_node_name)

    # Conditional transition
    graph.add_conditional_edges(
        yahoo_node_name,
        route_after_yahoo,
        {
            fred_node_name: fred_node_name,
            build_node_name: build_node_name,
        },
    )

    # Exogenous chain
    graph.add_edge(fred_node_name, open_meteo_node_name)
    graph.add_edge(open_meteo_node_name, build_node_name)

    # Build → Validate → End
    if include_validate:
        graph.add_edge(build_node_name, validate_node_name)
        graph.add_edge(validate_node_name, END)
    else:
        graph.add_edge(build_node_name, END)

    app = graph.compile()
    return (app, graph) if return_graph else app
