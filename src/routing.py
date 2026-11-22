"""
Risk-Aware Routing Engine Module
Constructs a weighted spatial graph and optimizes routes using Dijkstra or A*.
Conforms to Section 5.3 and Section 6.2 of the project report.
"""

from typing import Tuple, List, Dict, Any, Optional
import networkx as nx
import numpy as np

from .risk_model import IceRiskModel
from .baseline import (
    haversine_distance,
    calculate_effective_speed,
    RouteResult,
    KM_PER_NAUTICAL_MILE
)


class IceRoutingGraph:
    """
    Weighted spatial routing graph over an operating region with ice risk fields.
    Nodes represent spatial coordinates and edges represent navigable movement segments.
    """

    def __init__(
        self,
        risk_model: IceRiskModel,
        grid_lat_step: float = 0.35,
        grid_lon_step: float = 0.75,
        lat_bounds: Optional[Tuple[float, float]] = None,
        lon_bounds: Optional[Tuple[float, float]] = None
    ):
        self.risk_model = risk_model
        self.lat_step = grid_lat_step
        self.lon_step = grid_lon_step
        
        bb = risk_model.get_bounding_box()
        self.lat_min = lat_bounds[0] if lat_bounds else bb[0]
        self.lat_max = lat_bounds[1] if lat_bounds else bb[1]
        self.lon_min = lon_bounds[0] if lon_bounds else bb[2]
        self.lon_max = lon_bounds[1] if lon_bounds else bb[3]

        self.nodes_list: List[Tuple[float, float]] = []
        self._build_grid_nodes()

    def _build_grid_nodes(self) -> None:
        """Create regular spatial lattice of graph nodes."""
        lats = np.arange(self.lat_min, self.lat_max + self.lat_step / 2.0, self.lat_step)
        lons = np.arange(self.lon_min, self.lon_max + self.lon_step / 2.0, self.lon_step)

        self.nodes_list = [
            (round(float(la), 4), round(float(lo), 4))
            for la in lats
            for lo in lons
        ]

    def build_networkx_graph(
        self,
        base_speed_knots: float = 14.0,
        alpha_time: float = 1.0,
        beta_risk: float = 50.0,
        max_risk_threshold: float = 0.70,
        enforce_hard_constraint: bool = True
    ) -> nx.Graph:
        """
        Builds the NetworkX graph with 8-connectivity.
        Edge cost = alpha * TransitTime + beta * (Risk * Distance) + Penalty.
        """
        G = nx.Graph()

        # Build fast index mapping coordinate to index
        lat_indices = np.round((np.array([n[0] for n in self.nodes_list]) - self.lat_min) / self.lat_step).astype(int)
        lon_indices = np.round((np.array([n[1] for n in self.nodes_list]) - self.lon_min) / self.lon_step).astype(int)
        
        coord_map = {
            (lat_idx, lon_idx): (lat, lon)
            for lat_idx, lon_idx, (lat, lon) in zip(lat_indices, lon_indices, self.nodes_list)
        }

        # Pre-query node risks
        node_risks = {
            coord: self.risk_model.get_risk_at(coord[0], coord[1])["ice_risk"]
            for coord in self.nodes_list
        }

        for coord, r in node_risks.items():
            G.add_node(coord, risk=r, pos=(coord[1], coord[0]))

        # 8-connectivity directions
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        added_edges = set()

        for (la_i, lo_i), u in coord_map.items():
            u_risk = node_risks[u]

            for d_la, d_lo in directions:
                neighbor_key = (la_i + d_la, lo_i + d_lo)
                if neighbor_key not in coord_map:
                    continue

                v = coord_map[neighbor_key]
                edge_id = tuple(sorted([u, v]))
                if edge_id in added_edges:
                    continue
                added_edges.add(edge_id)

                v_risk = node_risks[v]
                mid_lat = (u[0] + v[0]) / 2.0
                mid_lon = (u[1] + v[1]) / 2.0
                edge_risk = self.risk_model.get_risk_at(mid_lat, mid_lon)["ice_risk"]

                # Hard constraint check
                if enforce_hard_constraint and edge_risk > max_risk_threshold:
                    continue  # Prune edge exceeding risk threshold

                dist_km = haversine_distance(u[0], u[1], v[0], v[1])
                eff_speed_knots = calculate_effective_speed(base_speed_knots, edge_risk)
                eff_speed_kmh = eff_speed_knots * KM_PER_NAUTICAL_MILE
                transit_time = dist_km / eff_speed_kmh if eff_speed_kmh > 0 else 0.0

                # Generic composite route objective: Cost = alpha * T + beta * R_exposure + Penalty
                penalty = 10000.0 if edge_risk > max_risk_threshold else 0.0
                cost = (alpha_time * transit_time) + (beta_risk * (edge_risk * dist_km)) + penalty

                G.add_edge(
                    u, v,
                    distance_km=dist_km,
                    edge_risk=edge_risk,
                    speed_knots=eff_speed_knots,
                    transit_time=transit_time,
                    cost=cost
                )

        return G


def connect_endpoint_to_graph(
    G: nx.Graph,
    pt: Tuple[float, float],
    risk_model: IceRiskModel,
    base_speed_knots: float,
    alpha_time: float,
    beta_risk: float,
    max_risk_threshold: float,
    k_neighbors: int = 6
) -> None:
    """Connect an arbitrary coordinate (origin/destination) to nearest valid graph nodes."""
    if pt in G:
        return

    pt_coord = np.array([pt[0], pt[1]])
    all_nodes = list(G.nodes)
    if not all_nodes:
        return

    nodes_arr = np.array(all_nodes)
    dists = np.hypot(nodes_arr[:, 0] - pt_coord[0], (nodes_arr[:, 1] - pt_coord[1]) * np.cos(np.radians(pt[0])))
    nearest_idx = np.argsort(dists)[:k_neighbors]

    G.add_node(pt, risk=risk_model.get_risk_at(pt[0], pt[1])["ice_risk"], pos=(pt[1], pt[0]))

    for idx in nearest_idx:
        v = all_nodes[idx]
        mid_lat = (pt[0] + v[0]) / 2.0
        mid_lon = (pt[1] + v[1]) / 2.0
        edge_risk = risk_model.get_risk_at(mid_lat, mid_lon)["ice_risk"]

        if edge_risk > max_risk_threshold:
            continue

        dist_km = haversine_distance(pt[0], pt[1], v[0], v[1])
        eff_speed_knots = calculate_effective_speed(base_speed_knots, edge_risk)
        eff_speed_kmh = eff_speed_knots * KM_PER_NAUTICAL_MILE
        transit_time = dist_km / eff_speed_kmh if eff_speed_kmh > 0 else 0.0

        cost = (alpha_time * transit_time) + (beta_risk * (edge_risk * dist_km))

        G.add_edge(
            pt, v,
            distance_km=dist_km,
            edge_risk=edge_risk,
            speed_knots=eff_speed_knots,
            transit_time=transit_time,
            cost=cost
        )


def optimize_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    risk_model: IceRiskModel,
    algorithm: str = "dijkstra",
    base_speed_knots: float = 14.0,
    alpha_time: float = 1.0,
    beta_risk: float = 50.0,
    max_risk_threshold: float = 0.70,
    grid_lat_step: float = 0.35,
    grid_lon_step: float = 0.75
) -> RouteResult:
    """
    Finds the optimal risk-aware route using Dijkstra or A* on the spatial ice graph.
    """
    graph_builder = IceRoutingGraph(
        risk_model=risk_model,
        grid_lat_step=grid_lat_step,
        grid_lon_step=grid_lon_step
    )

    # First attempt with hard threshold constraint
    G = graph_builder.build_networkx_graph(
        base_speed_knots=base_speed_knots,
        alpha_time=alpha_time,
        beta_risk=beta_risk,
        max_risk_threshold=max_risk_threshold,
        enforce_hard_constraint=True
    )

    connect_endpoint_to_graph(
        G, origin, risk_model, base_speed_knots, alpha_time, beta_risk, max_risk_threshold
    )
    connect_endpoint_to_graph(
        G, destination, risk_model, base_speed_knots, alpha_time, beta_risk, max_risk_threshold
    )

    path_nodes = None
    feasible = True

    try:
        if nx.has_path(G, origin, destination):
            if algorithm.lower() == "astar":
                def heuristic(u, v):
                    # Admissible lower-bound heuristic: straight-line distance at base speed
                    d = haversine_distance(u[0], u[1], v[0], v[1])
                    return alpha_time * (d / (base_speed_knots * KM_PER_NAUTICAL_MILE))

                path_nodes = nx.astar_path(G, origin, destination, heuristic=heuristic, weight="cost")
            else:
                path_nodes = nx.dijkstra_path(G, origin, destination, weight="cost")
    except nx.NetworkXNoPath:
        path_nodes = None

    # If disconnected due to strict risk threshold, fall back to soft penalty graph
    if path_nodes is None:
        G_soft = graph_builder.build_networkx_graph(
            base_speed_knots=base_speed_knots,
            alpha_time=alpha_time,
            beta_risk=beta_risk,
            max_risk_threshold=max_risk_threshold,
            enforce_hard_constraint=False
        )
        connect_endpoint_to_graph(
            G_soft, origin, risk_model, base_speed_knots, alpha_time, beta_risk, 1.0
        )
        connect_endpoint_to_graph(
            G_soft, destination, risk_model, base_speed_knots, alpha_time, beta_risk, 1.0
        )

        if nx.has_path(G_soft, origin, destination):
            path_nodes = nx.dijkstra_path(G_soft, origin, destination, weight="cost")
            feasible = False  # Soft penalty path forced to violate threshold
        else:
            raise RuntimeError("No feasible route found across the operating graph region.")

    # Reconstruct route result
    total_dist_km = 0.0
    total_time_hours = 0.0
    cumulative_risk = 0.0
    max_risk = 0.0
    segments = []

    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]

        seg_dist = haversine_distance(u[0], u[1], v[0], v[1])
        mid_lat = (u[0] + v[0]) / 2.0
        mid_lon = (u[1] + v[1]) / 2.0
        risk_info = risk_model.get_risk_at(mid_lat, mid_lon)
        seg_risk = risk_info["ice_risk"]

        eff_speed_knots = calculate_effective_speed(base_speed_knots, seg_risk)
        eff_speed_kmh = eff_speed_knots * KM_PER_NAUTICAL_MILE
        seg_time = seg_dist / eff_speed_kmh if eff_speed_kmh > 0 else 0.0

        total_dist_km += seg_dist
        total_time_hours += seg_time
        cumulative_risk += seg_risk * seg_dist
        if seg_risk > max_risk:
            max_risk = seg_risk

        segments.append({
            "segment_idx": i,
            "start": u,
            "end": v,
            "distance_km": seg_dist,
            "ice_risk": seg_risk,
            "ice_concentration": risk_info["ice_concentration"],
            "ice_thickness_m": risk_info["ice_thickness_m"],
            "ice_type": risk_info["ice_type"],
            "speed_knots": eff_speed_knots,
            "time_hours": seg_time
        })

    avg_risk = cumulative_risk / total_dist_km if total_dist_km > 0 else 0.0
    is_feasible = feasible and (max_risk <= max_risk_threshold)

    return RouteResult(
        name=f"Proposed: Risk-Aware Route ({algorithm.upper()})",
        algorithm=algorithm.upper(),
        waypoints=path_nodes,
        total_distance_km=total_dist_km,
        total_distance_nm=total_dist_km / KM_PER_NAUTICAL_MILE,
        total_transit_time_hours=total_time_hours,
        cumulative_ice_risk=cumulative_risk,
        average_risk=avg_risk,
        max_segment_risk=max_risk,
        feasibility=is_feasible,
        segments=segments
    )
