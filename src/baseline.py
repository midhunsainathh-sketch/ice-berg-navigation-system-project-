"""
Baseline Routing Module
Calculates the mandatory Shortest Great-Circle route using the Haversine formula
and evaluates it against the spatial ice-risk field.
Conforms to Section 5.2 of the project report.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np
from .risk_model import IceRiskModel


EARTH_RADIUS_KM = 6371.0088
KM_PER_NAUTICAL_MILE = 1.852


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates great-circle distance between two coordinates in kilometers
    using the Haversine formula.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return float(EARTH_RADIUS_KM * c)


def sample_great_circle_waypoints(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    num_points: int = 100
) -> List[Tuple[float, float]]:
    """
    Generates waypoints along the Great-Circle arc between origin and destination
    using spherical linear interpolation (SLERP).
    """
    lat1, lon1 = np.radians(origin[0]), np.radians(origin[1])
    lat2, lon2 = np.radians(destination[0]), np.radians(destination[1])

    d = haversine_distance(origin[0], origin[1], destination[0], destination[1]) / EARTH_RADIUS_KM
    if d < 1e-9:
        return [origin, destination]

    fractions = np.linspace(0.0, 1.0, num_points)
    waypoints = []

    for f in fractions:
        a = np.sin((1.0 - f) * d) / np.sin(d)
        b = np.sin(f * d) / np.sin(d)

        x = a * np.cos(lat1) * np.cos(lon1) + b * np.cos(lat2) * np.cos(lon2)
        y = a * np.cos(lat1) * np.sin(lon1) + b * np.cos(lat2) * np.sin(lon2)
        z = a * np.sin(lat1) + b * np.sin(lat2)

        lat = np.arctan2(z, np.sqrt(x**2 + y**2))
        lon = np.arctan2(y, x)

        waypoints.append((float(np.degrees(lat)), float(np.degrees(lon))))

    return waypoints


@dataclass
class RouteResult:
    """Represents a computed route and its associated operational evaluation metrics."""
    name: str
    algorithm: str
    waypoints: List[Tuple[float, float]]
    total_distance_km: float
    total_distance_nm: float
    total_transit_time_hours: float
    cumulative_ice_risk: float
    average_risk: float
    max_segment_risk: float
    feasibility: bool
    segments: List[Dict[str, Any]] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "Route": self.name,
            "Transit time (hrs)": round(self.total_transit_time_hours, 2),
            "Cumulative ice risk (risk-km)": round(self.cumulative_ice_risk, 2),
            "Max. risk (0-1)": round(self.max_segment_risk, 3),
            "Average risk (0-1)": round(self.average_risk, 3),
            "Distance (km)": round(self.total_distance_km, 1),
            "Distance (NM)": round(self.total_distance_nm, 1),
            "Feasibility": "Pass" if self.feasibility else "Fail"
        }


def calculate_effective_speed(base_speed_knots: float, segment_risk: float) -> float:
    """
    Calculates vessel effective transit speed through ice.
    Speed decreases with ice risk due to ice-resistance, pack-maneuvering, and safety.
    Clamped to a minimum safe crawling speed (15% of base speed).
    """
    speed_factor = max(0.15, 1.0 - 0.75 * segment_risk)
    return base_speed_knots * speed_factor


def generate_great_circle_baseline(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    risk_model: IceRiskModel,
    base_speed_knots: float = 14.0,
    max_risk_threshold: float = 0.70,
    num_samples: int = 100
) -> RouteResult:
    """
    Constructs the Great-Circle baseline route and evaluates it against the exact same
    risk field as the proposed route optimizer.
    """
    waypoints = sample_great_circle_waypoints(origin, destination, num_points=num_samples)

    total_dist_km = 0.0
    total_time_hours = 0.0
    cumulative_risk = 0.0
    max_risk = 0.0
    segments = []

    for i in range(len(waypoints) - 1):
        pt1 = waypoints[i]
        pt2 = waypoints[i + 1]

        seg_dist = haversine_distance(pt1[0], pt1[1], pt2[0], pt2[1])
        mid_lat = (pt1[0] + pt2[0]) / 2.0
        mid_lon = (pt1[1] + pt2[1]) / 2.0

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
            "start": pt1,
            "end": pt2,
            "distance_km": seg_dist,
            "ice_risk": seg_risk,
            "ice_concentration": risk_info["ice_concentration"],
            "ice_thickness_m": risk_info["ice_thickness_m"],
            "ice_type": risk_info["ice_type"],
            "speed_knots": eff_speed_knots,
            "time_hours": seg_time
        })

    avg_risk = cumulative_risk / total_dist_km if total_dist_km > 0 else 0.0
    is_feasible = max_risk <= max_risk_threshold

    return RouteResult(
        name="Baseline: Shortest Great-Circle",
        algorithm="Great-Circle",
        waypoints=waypoints,
        total_distance_km=total_dist_km,
        total_distance_nm=total_dist_km / KM_PER_NAUTICAL_MILE,
        total_transit_time_hours=total_time_hours,
        cumulative_ice_risk=cumulative_risk,
        average_risk=avg_risk,
        max_segment_risk=max_risk,
        feasibility=is_feasible,
        segments=segments
    )
