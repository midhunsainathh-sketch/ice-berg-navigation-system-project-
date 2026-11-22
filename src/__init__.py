"""
Ice-Risk Aware Route Decision Support System (DSS)
Core package initialization.
"""

from .risk_model import IceRiskModel
from .baseline import generate_great_circle_baseline, haversine_distance, RouteResult
from .routing import IceRoutingGraph, optimize_route
from .evaluation import compare_routes, format_comparison_table
from .visualization import create_folium_map, create_risk_profile_figure, create_metrics_bar_chart

__all__ = [
    "IceRiskModel",
    "generate_great_circle_baseline",
    "haversine_distance",
    "RouteResult",
    "IceRoutingGraph",
    "optimize_route",
    "compare_routes",
    "format_comparison_table",
    "create_folium_map",
    "create_risk_profile_figure",
    "create_metrics_bar_chart",
]
