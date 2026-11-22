"""
Evaluation and Performance Comparison Module
Calculates operational trade-off metrics between the Great-Circle baseline and proposed route.
Conforms to Section 7 (Results and Evaluation) and Table 7 of the project report.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from .baseline import RouteResult


def calculate_tradeoff_metrics(
    baseline: RouteResult,
    proposed: RouteResult
) -> Dict[str, float]:
    """
    Computes comparative operational metrics:
    - Risk reduction (%) = ((baseline risk - proposed risk) / baseline risk) * 100
    - Time penalty (%) = ((proposed time - baseline time) / baseline time) * 100
    - Distance penalty (%) = ((proposed distance - baseline distance) / baseline distance) * 100
    """
    if baseline.cumulative_ice_risk > 1e-6:
        risk_reduction = (
            (baseline.cumulative_ice_risk - proposed.cumulative_ice_risk) / baseline.cumulative_ice_risk
        ) * 100.0
    else:
        risk_reduction = 0.0

    if baseline.total_transit_time_hours > 1e-6:
        time_penalty = (
            (proposed.total_transit_time_hours - baseline.total_transit_time_hours) / baseline.total_transit_time_hours
        ) * 100.0
    else:
        time_penalty = 0.0

    if baseline.total_distance_km > 1e-6:
        dist_penalty = (
            (proposed.total_distance_km - baseline.total_distance_km) / baseline.total_distance_km
        ) * 100.0
    else:
        dist_penalty = 0.0

    if baseline.max_segment_risk > 1e-6:
        max_risk_reduction = (
            (baseline.max_segment_risk - proposed.max_segment_risk) / baseline.max_segment_risk
        ) * 100.0
    else:
        max_risk_reduction = 0.0

    return {
        "risk_reduction_pct": round(risk_reduction, 2),
        "time_penalty_pct": round(time_penalty, 2),
        "distance_penalty_pct": round(dist_penalty, 2),
        "max_risk_reduction_pct": round(max_risk_reduction, 2)
    }


def compare_routes(
    baseline: RouteResult,
    proposed: RouteResult
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates comparison DataFrame matching Table 7 of the report along with trade-off metrics.
    """
    rows = [
        {
            "Route": "Baseline: shortest great-circle",
            "Transit time": f"{baseline.total_transit_time_hours:.1f} hrs",
            "Cumulative ice risk": f"{baseline.cumulative_ice_risk:.1f}",
            "Max. risk": f"{baseline.max_segment_risk:.2f}",
            "Avg. risk": f"{baseline.average_risk:.2f}",
            "Distance (km)": f"{baseline.total_distance_km:.1f}",
            "Distance (NM)": f"{baseline.total_distance_nm:.1f}",
            "Feasibility": "Pass" if baseline.feasibility else "Fail"
        },
        {
            "Route": f"Proposed: risk-aware route ({proposed.algorithm})",
            "Transit time": f"{proposed.total_transit_time_hours:.1f} hrs",
            "Cumulative ice risk": f"{proposed.cumulative_ice_risk:.1f}",
            "Max. risk": f"{proposed.max_segment_risk:.2f}",
            "Avg. risk": f"{proposed.average_risk:.2f}",
            "Distance (km)": f"{proposed.total_distance_km:.1f}",
            "Distance (NM)": f"{proposed.total_distance_nm:.1f}",
            "Feasibility": "Pass" if proposed.feasibility else "Fail"
        }
    ]

    df = pd.DataFrame(rows)
    tradeoff = calculate_tradeoff_metrics(baseline, proposed)

    summary = {
        "comparison_table": df,
        "tradeoffs": tradeoff,
        "baseline_raw": baseline.to_summary_dict(),
        "proposed_raw": proposed.to_summary_dict()
    }

    return df, summary


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a GitHub Flavored Markdown table without external dependencies."""
    headers = list(df.columns)
    lines = []
    # Header row
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    # Separator row
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    # Data rows
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def format_comparison_table(df: pd.DataFrame, tradeoffs: Dict[str, float]) -> str:
    """Format evaluation comparison table as a clean Markdown table for reports."""
    md = dataframe_to_markdown(df)
    notes = (
        f"\n\n**Evaluation Summary:**\n"
        f"- **Cumulative Risk Reduction:** `{tradeoffs['risk_reduction_pct']}%`\n"
        f"- **Maximum Segment Risk Reduction:** `{tradeoffs['max_risk_reduction_pct']}%`\n"
        f"- **Transit Time Penalty:** `{tradeoffs['time_penalty_pct']:+.1f}%`\n"
        f"- **Distance Penalty:** `{tradeoffs['distance_penalty_pct']:+.1f}%`\n"
    )
    return md + notes
