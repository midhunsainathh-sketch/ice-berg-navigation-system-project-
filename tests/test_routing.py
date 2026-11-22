"""
Automated Unit and Integration Test Suite
Validates Haversine distance, Great-Circle baseline, Risk Model, Graph Routing, and Evaluation.
Runs with standard library unittest or pytest.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline import haversine_distance, sample_great_circle_waypoints, generate_great_circle_baseline
from src.risk_model import IceRiskModel
from src.routing import optimize_route
from src.evaluation import compare_routes, calculate_tradeoff_metrics


class TestIceRoutingDSS(unittest.TestCase):

    def test_haversine_distance(self):
        # 1 degree of latitude at equator is approximately 111.19 km
        d = haversine_distance(0.0, 0.0, 1.0, 0.0)
        self.assertTrue(110.5 < d < 111.5, f"Expected ~111.19 km, got {d}")

        # Distance to self must be 0
        self.assertEqual(haversine_distance(70.0, 20.0, 70.0, 20.0), 0.0)

    def test_sample_great_circle_waypoints(self):
        origin = (70.0, 20.0)
        dest = (78.0, 15.0)
        waypoints = sample_great_circle_waypoints(origin, dest, num_points=10)
        
        self.assertEqual(len(waypoints), 10)
        self.assertTrue(np.isclose(waypoints[0][0], origin[0], atol=1e-4))
        self.assertTrue(np.isclose(waypoints[0][1], origin[1], atol=1e-4))
        self.assertTrue(np.isclose(waypoints[-1][0], dest[0], atol=1e-4))
        self.assertTrue(np.isclose(waypoints[-1][1], dest[1], atol=1e-4))

    def test_risk_model_normalization(self):
        df = pd.DataFrame([
            {"latitude": 70.0, "longitude": 20.0, "ice_concentration": 0.0, "ice_thickness_m": 0.0, "ice_type": "Open Water"},
            {"latitude": 75.0, "longitude": 25.0, "ice_concentration": 0.9, "ice_thickness_m": 2.0, "ice_type": "Thick First-Year Ice"},
        ])
        model = IceRiskModel(df)
        
        # Check open water risk
        r_open = model.get_risk_at(70.0, 20.0)["ice_risk"]
        self.assertEqual(r_open, 0.0)
        
        # Check heavy ice risk
        r_heavy = model.get_risk_at(75.0, 25.0)["ice_risk"]
        self.assertTrue(0.70 < r_heavy <= 1.0)

    def test_routing_optimization_and_tradeoff(self):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "ice_conditions.csv")
        if not os.path.exists(csv_path):
            from data.generate_sample_data import generate_ice_dataset
            generate_ice_dataset(csv_path)

        model = IceRiskModel(csv_path)

        origin = (70.5, 20.0)
        dest = (76.0, 36.0)

        baseline = generate_great_circle_baseline(
            origin=origin,
            destination=dest,
            risk_model=model,
            base_speed_knots=14.0,
            max_risk_threshold=0.75
        )

        proposed = optimize_route(
            origin=origin,
            destination=dest,
            risk_model=model,
            algorithm="dijkstra",
            base_speed_knots=14.0,
            alpha_time=1.0,
            beta_risk=50.0,
            max_risk_threshold=0.75,
            grid_lat_step=0.5,
            grid_lon_step=1.0
        )

        self.assertGreater(baseline.total_distance_km, 0)
        self.assertGreater(proposed.total_distance_km, 0)

        # Baseline penetrates heavy pack ice barrier while proposed route is feasible
        self.assertFalse(baseline.feasibility)
        self.assertTrue(proposed.feasibility)

        tradeoffs = calculate_tradeoff_metrics(baseline, proposed)
        self.assertGreater(tradeoffs["risk_reduction_pct"], 15.0)

        df_comp, summary = compare_routes(baseline, proposed)
        self.assertEqual(len(df_comp), 2)
        self.assertIn("Feasibility", df_comp.columns)

    def test_astar_vs_dijkstra(self):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "ice_conditions.csv")
        model = IceRiskModel(csv_path)

        origin = (70.0, 20.0)
        dest = (76.0, 25.0)

        route_dijkstra = optimize_route(
            origin=origin,
            destination=dest,
            risk_model=model,
            algorithm="dijkstra",
            grid_lat_step=0.5,
            grid_lon_step=1.0
        )

        route_astar = optimize_route(
            origin=origin,
            destination=dest,
            risk_model=model,
            algorithm="astar",
            grid_lat_step=0.5,
            grid_lon_step=1.0
        )

        self.assertEqual(route_dijkstra.feasibility, route_astar.feasibility)
        self.assertTrue(np.isclose(route_dijkstra.total_distance_km, route_astar.total_distance_km, rtol=0.15))


if __name__ == "__main__":
    unittest.main()
