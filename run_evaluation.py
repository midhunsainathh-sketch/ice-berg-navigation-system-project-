"""
Headless Route Evaluation CLI Runner
Generates experimental results against the baseline and prints Table 7 directly.
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.risk_model import IceRiskModel
from src.baseline import generate_great_circle_baseline
from src.routing import optimize_route
from src.evaluation import compare_routes, format_comparison_table
from data.generate_sample_data import generate_ice_dataset


def main():
    print("=" * 70)
    print("  ICE-RISK AWARE ROUTE DECISION SUPPORT SYSTEM (DSS)")
    print("  CA-1 Project Automated Evaluation Runner")
    print("=" * 70)

    data_path = os.path.join(os.path.dirname(__file__), "data", "ice_conditions.csv")
    if not os.path.exists(data_path):
        print(f"[*] Dataset not found at {data_path}. Generating realistic Arctic dataset...")
        generate_ice_dataset(data_path)

    print(f"[*] Loading ice-risk field from {data_path}...")
    model = IceRiskModel(data_path)
    bb = model.get_bounding_box()
    print(f"[*] Operating Area: Lat {bb[0]}°N to {bb[1]}°N, Lon {bb[2]}°E to {bb[3]}°E")

    # Define standard evaluation scenario: Tromso (69.65N, 18.96E) -> Longyearbyen Svalbard (78.22N, 15.65E)
    origin = (69.65, 18.96)
    dest = (78.22, 15.65)
    vessel_speed_knots = 14.0
    risk_threshold = 0.65
    alpha = 1.0
    beta = 60.0

    print(f"\n[*] Evaluating Scenario: Tromsø ({origin}) -> Longyearbyen ({dest})")
    print(f"    - Base vessel speed: {vessel_speed_knots} knots")
    print(f"    - Max feasible risk threshold: {risk_threshold}")
    print(f"    - Cost objective: Cost = {alpha} * Time + {beta} * Risk")

    # 1. Baseline Route
    print("\n[1/2] Computing Shortest Great-Circle Baseline...")
    baseline = generate_great_circle_baseline(
        origin=origin,
        destination=dest,
        risk_model=model,
        base_speed_knots=vessel_speed_knots,
        max_risk_threshold=risk_threshold
    )
    print(f"      Done. Distance: {baseline.total_distance_km:.1f} km | Time: {baseline.total_transit_time_hours:.1f} hrs | Cumulative Risk: {baseline.cumulative_ice_risk:.1f} | Max Risk: {baseline.max_segment_risk:.2f} | Feasibility: {'Pass' if baseline.feasibility else 'Fail'}")

    # 2. Proposed Route (Dijkstra & A*)
    print("\n[2/2] Optimizing Proposed Risk-Aware Route (Dijkstra)...")
    proposed = optimize_route(
        origin=origin,
        destination=dest,
        risk_model=model,
        algorithm="dijkstra",
        base_speed_knots=vessel_speed_knots,
        alpha_time=alpha,
        beta_risk=beta,
        max_risk_threshold=risk_threshold,
        grid_lat_step=0.35,
        grid_lon_step=0.75
    )
    print(f"      Done. Distance: {proposed.total_distance_km:.1f} km | Time: {proposed.total_transit_time_hours:.1f} hrs | Cumulative Risk: {proposed.cumulative_ice_risk:.1f} | Max Risk: {proposed.max_segment_risk:.2f} | Feasibility: {'Pass' if proposed.feasibility else 'Fail'}")

    # 3. Comparative Evaluation
    df_comp, summary = compare_routes(baseline, proposed)
    tradeoffs = summary["tradeoffs"]

    print("\n" + "=" * 70)
    print("  TABLE 7: RESULTS AGAINST THE BASELINE")
    print("=" * 70)
    print(format_comparison_table(df_comp, tradeoffs))
    print("=" * 70)

    # Save to CSV
    output_csv = os.path.join(os.path.dirname(__file__), "data", "table_7_results.csv")
    df_comp.to_csv(output_csv, index=False)
    print(f"\n[+] Results saved to {output_csv}")


if __name__ == "__main__":
    main()
