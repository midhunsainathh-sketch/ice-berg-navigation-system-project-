"""
Visualization and Mapping Module
Provides interactive geospatial map rendering (Folium) and performance analytics charts (Plotly).
Conforms to Section 6.3 (Demo / User Interface) and Table 6 of the project report.
"""

from typing import Optional, List, Tuple
import folium
from folium import plugins
import plotly.graph_objects as go
import numpy as np

from .baseline import RouteResult
from .risk_model import IceRiskModel


def get_risk_color(risk: float) -> str:
    """Map normalized risk [0, 1] to human-interpretable color code."""
    if risk < 0.20:
        return "#2dc653"  # Low risk - Emerald Green
    elif risk < 0.45:
        return "#e9c46a"  # Moderate risk - Yellow Gold
    elif risk < 0.70:
        return "#f4a261"  # Significant risk - Warm Orange
    else:
        return "#e63946"  # Severe/Impassable risk - Crimson Red


def create_folium_map(
    risk_model: IceRiskModel,
    baseline_route: Optional[RouteResult] = None,
    proposed_route: Optional[RouteResult] = None,
    max_risk_threshold: Optional[float] = 0.70,
    downsample_factor: int = 2
) -> folium.Map:
    """
    Renders an interactive geospatial map with:
    1. Base cartography (CartoDB Positron & OpenStreetMap).
    2. Ice risk spatial layer.
    3. Baseline Great-Circle route overlay (dashed crimson).
    4. Proposed Risk-Aware route overlay (solid royal blue).
    5. Waypoint markers, origin/destination pins, and layer controls.
    """
    # Determine map center
    all_lats = []
    all_lons = []
    if baseline_route and baseline_route.waypoints:
        all_lats.extend([pt[0] for pt in baseline_route.waypoints])
        all_lons.extend([pt[1] for pt in baseline_route.waypoints])
    if proposed_route and proposed_route.waypoints:
        all_lats.extend([pt[0] for pt in proposed_route.waypoints])
        all_lons.extend([pt[1] for pt in proposed_route.waypoints])

    if all_lats:
        center_lat = float(np.mean(all_lats))
        center_lon = float(np.mean(all_lons))
    else:
        bb = risk_model.get_bounding_box()
        center_lat = (bb[0] + bb[1]) / 2.0
        center_lon = (bb[2] + bb[3]) / 2.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron",
        control_scale=True
    )

    # Alternate tile layer
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # 1. Ice Risk Field Layer
    df = risk_model.get_dataframe()
    if not df.empty:
        risk_fg = folium.FeatureGroup(name="Spatial Ice Risk Field", show=True)
        # Downsample slightly for smooth browser rendering
        df_sub = df.iloc[::downsample_factor]
        for _, row in df_sub.iterrows():
            r = float(row["ice_risk"])
            if r <= 0.05:
                continue  # Skip near-zero open ocean to reduce clutter
            color = get_risk_color(r)
            radius = 12000  # meters
            folium.Circle(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.35,
                weight=1,
                tooltip=(
                    f"<b>Ice Observation</b><br>"
                    f"Lat: {row['latitude']:.2f}°, Lon: {row['longitude']:.2f}°<br>"
                    f"Risk: {r:.2f}<br>"
                    f"Concentration: {row['ice_concentration']:.0%}<br>"
                    f"Thickness: {row['ice_thickness_m']:.2f} m<br>"
                    f"Type: {row['ice_type']}"
                )
            ).add_to(risk_fg)
        risk_fg.add_to(m)

    # 2. Baseline Great-Circle Route Overlay
    if baseline_route and baseline_route.waypoints:
        base_fg = folium.FeatureGroup(name="Baseline: Great-Circle", show=True)
        folium.PolyLine(
            locations=baseline_route.waypoints,
            color="#d90429",
            weight=4,
            opacity=0.85,
            dash_array="8, 8",
            tooltip=(
                f"<b>Baseline: Shortest Great-Circle</b><br>"
                f"Distance: {baseline_route.total_distance_km:.1f} km<br>"
                f"Transit Time: {baseline_route.total_transit_time_hours:.1f} hrs<br>"
                f"Cumulative Risk: {baseline_route.cumulative_ice_risk:.1f}<br>"
                f"Max Risk: {baseline_route.max_segment_risk:.2f}"
            )
        ).add_to(base_fg)
        base_fg.add_to(m)

    # 3. Proposed Risk-Aware Route Overlay
    if proposed_route and proposed_route.waypoints:
        prop_fg = folium.FeatureGroup(name=f"Proposed: {proposed_route.algorithm} Route", show=True)
        folium.PolyLine(
            locations=proposed_route.waypoints,
            color="#0077b6",
            weight=5,
            opacity=0.95,
            tooltip=(
                f"<b>Proposed: Risk-Aware Route ({proposed_route.algorithm})</b><br>"
                f"Distance: {proposed_route.total_distance_km:.1f} km<br>"
                f"Transit Time: {proposed_route.total_transit_time_hours:.1f} hrs<br>"
                f"Cumulative Risk: {proposed_route.cumulative_ice_risk:.1f}<br>"
                f"Max Risk: {proposed_route.max_segment_risk:.2f}"
            )
        ).add_to(prop_fg)

        # Subtle waypoint dots
        for i, pt in enumerate(proposed_route.waypoints):
            if i % 2 == 0:
                folium.CircleMarker(
                    location=pt,
                    radius=4,
                    color="#023e8a",
                    fill=True,
                    fill_color="#90e0ef",
                    fill_opacity=0.9,
                    tooltip=f"Waypoint {i+1}: ({pt[0]:.2f}°, {pt[1]:.2f}°)"
                ).add_to(prop_fg)

        prop_fg.add_to(m)

    # 4. Origin and Destination Pin Markers
    endpoints = []
    if proposed_route and proposed_route.waypoints:
        endpoints = [proposed_route.waypoints[0], proposed_route.waypoints[-1]]
    elif baseline_route and baseline_route.waypoints:
        endpoints = [baseline_route.waypoints[0], baseline_route.waypoints[-1]]

    if endpoints:
        start_pt, end_pt = endpoints[0], endpoints[1]
        folium.Marker(
            location=start_pt,
            popup="<b>Route Origin</b>",
            tooltip="Origin",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(m)

        folium.Marker(
            location=end_pt,
            popup="<b>Route Destination</b>",
            tooltip="Destination",
            icon=folium.Icon(color="red", icon="flag", prefix="fa")
        ).add_to(m)

    # Add layer control
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m


def create_risk_profile_figure(
    baseline_route: RouteResult,
    proposed_route: RouteResult,
    max_risk_threshold: float = 0.70
) -> go.Figure:
    """
    Renders distance vs. ice risk profile plot along both journeys,
    highlighting high-risk ice penetration by the baseline and risk-avoidance by the proposed route.
    """
    # Extract baseline cumulative distances and segment risks
    base_dist = [0.0]
    base_risk = [0.0]
    cum_d = 0.0
    for seg in baseline_route.segments:
        cum_d += seg["distance_km"]
        base_dist.append(cum_d)
        base_risk.append(seg["ice_risk"])

    # Extract proposed cumulative distances and segment risks
    prop_dist = [0.0]
    prop_risk = [0.0]
    cum_d = 0.0
    for seg in proposed_route.segments:
        cum_d += seg["distance_km"]
        prop_dist.append(cum_d)
        prop_risk.append(seg["ice_risk"])

    fig = go.Figure()

    # Max risk threshold horizontal dashed line
    max_x = max(base_dist[-1], prop_dist[-1]) * 1.05
    fig.add_trace(go.Scatter(
        x=[0, max_x],
        y=[max_risk_threshold, max_risk_threshold],
        mode="lines",
        name=f"Feasibility Threshold ({max_risk_threshold})",
        line=dict(color="#ef233c", dash="dash", width=2)
    ))

    # Baseline route profile
    fig.add_trace(go.Scatter(
        x=base_dist,
        y=base_risk,
        mode="lines",
        name="Baseline (Great-Circle)",
        line=dict(color="#d90429", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(217, 4, 41, 0.12)"
    ))

    # Proposed route profile
    fig.add_trace(go.Scatter(
        x=prop_dist,
        y=prop_risk,
        mode="lines",
        name=f"Proposed ({proposed_route.algorithm})",
        line=dict(color="#0077b6", width=3.0),
        fill="tozeroy",
        fillcolor="rgba(0, 119, 182, 0.20)"
    ))

    fig.update_layout(
        title="<b>Route Ice Risk Profile (Distance vs. Local Ice Risk)</b>",
        xaxis_title="Cumulative Transit Distance (km)",
        yaxis_title="Normalized Ice Risk Score (0 - 1)",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )

    return fig


def create_metrics_bar_chart(
    baseline_route: RouteResult,
    proposed_route: RouteResult
) -> go.Figure:
    """Side-by-side bar chart comparing key operational performance indicators."""
    metrics = ["Transit Time (hrs)", "Cumulative Risk", "Max Risk (x100)", "Total Distance (100 km)"]
    
    base_vals = [
        baseline_route.total_transit_time_hours,
        baseline_route.cumulative_ice_risk,
        baseline_route.max_segment_risk * 100.0,
        baseline_route.total_distance_km / 100.0
    ]
    prop_vals = [
        proposed_route.total_transit_time_hours,
        proposed_route.cumulative_ice_risk,
        proposed_route.max_segment_risk * 100.0,
        proposed_route.total_distance_km / 100.0
    ]

    fig = go.Figure(data=[
        go.Bar(
            name="Baseline: Great-Circle",
            x=metrics,
            y=base_vals,
            marker_color="#d90429"
        ),
        go.Bar(
            name=f"Proposed: Risk-Aware ({proposed_route.algorithm})",
            x=metrics,
            y=prop_vals,
            marker_color="#0077b6"
        )
    ])

    fig.update_layout(
        title="<b>Operational Metrics Comparison</b>",
        barmode="group",
        template="plotly_white",
        yaxis_title="Metric Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig
