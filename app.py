"""
Ice-Risk Aware Route Decision Support System (DSS)
Streamlit Interactive Dashboard Application.
Conforms to Section 6.3 (Demo / User Interface) and Table 6 of the CA-1 Project Report.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_folium import st_folium

# Add root folder to module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.risk_model import IceRiskModel
from src.baseline import generate_great_circle_baseline, RouteResult
from src.routing import optimize_route
from src.evaluation import compare_routes, calculate_tradeoff_metrics, format_comparison_table
from src.visualization import (
    create_folium_map,
    create_risk_profile_figure,
    create_metrics_bar_chart
)
from data.generate_sample_data import generate_ice_dataset

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ice-Risk Aware Route DSS",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f4c81;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #495057;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #0f4c81;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loader with Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Ice Condition Field...")
def get_risk_model(csv_path: str) -> IceRiskModel:
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        generate_ice_dataset(csv_path)
    return IceRiskModel(csv_path)


# ---------------------------------------------------------
# Presets
# ---------------------------------------------------------
PRESETS = {
    "Tromsø (Norway) -> Longyearbyen (Svalbard)": {
        "origin": (69.65, 18.96),
        "destination": (78.22, 15.65),
        "desc": "Standard Norwegian Arctic passage crossing variable drift ice near Bear Island (Bjørnøya)."
    },
    "Vardø (Barents Sea) -> Franz Josef Land": {
        "origin": (70.37, 31.11),
        "destination": (79.50, 45.00),
        "desc": "Northeastern route crossing active multi-year pack ice barriers in the central Barents Sea."
    },
    "Hammerfest -> Novaya Zemlya Western Approaches": {
        "origin": (70.66, 23.68),
        "destination": (73.50, 48.00),
        "desc": "Eastward transit into the Kara Gate corridor with prominent seasonal ice tongues."
    },
    "Custom Coordinates": {
        "origin": (70.0, 20.0),
        "destination": (77.0, 30.0),
        "desc": "User defined coordinate endpoints within the operating region."
    }
}


# ---------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/iceberg.png", width=64)
st.sidebar.title("Route Configuration")

# 1. Dataset Selection
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
default_csv = os.path.join(data_dir, "ice_conditions.csv")

uploaded_file = st.sidebar.file_uploader("Upload Custom Ice CSV (Optional)", type=["csv"])
if uploaded_file is not None:
    df_custom = pd.read_csv(uploaded_file)
    risk_model = IceRiskModel(df_custom)
    st.sidebar.success(f"Loaded {len(df_custom)} custom observations")
else:
    risk_model = get_risk_model(default_csv)

# 2. Preset Selection
scenario_choice = st.sidebar.selectbox("Voyage Scenario / Endpoints", list(PRESETS.keys()))
scenario_data = PRESETS[scenario_choice]
st.sidebar.caption(f"ℹ️ {scenario_data['desc']}")

if scenario_choice == "Custom Coordinates":
    col_orig1, col_orig2 = st.sidebar.columns(2)
    with col_orig1:
        orig_lat = st.number_input("Origin Lat (°N)", min_value=68.0, max_value=80.0, value=70.0, step=0.25)
    with col_orig2:
        orig_lon = st.number_input("Origin Lon (°E)", min_value=10.0, max_value=50.0, value=20.0, step=0.5)

    col_dest1, col_dest2 = st.sidebar.columns(2)
    with col_dest1:
        dest_lat = st.number_input("Dest Lat (°N)", min_value=68.0, max_value=80.0, value=77.5, step=0.25)
    with col_dest2:
        dest_lon = st.number_input("Dest Lon (°E)", min_value=10.0, max_value=50.0, value=25.0, step=0.5)
    
    origin = (orig_lat, orig_lon)
    destination = (dest_lat, dest_lon)
else:
    origin = scenario_data["origin"]
    destination = scenario_data["destination"]

# 3. Vessel & Optimization Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation & Optimizer Settings")

vessel_speed = st.sidebar.slider(
    "Base Cruise Speed (knots)",
    min_value=8.0,
    max_value=22.0,
    value=14.0,
    step=0.5,
    help="Open water operational cruise speed before ice-retardation."
)

algorithm_choice = st.sidebar.radio(
    "Routing Algorithm",
    ["Dijkstra", "A* Search"],
    index=0,
    help="Graph shortest-path search algorithm."
)

max_risk_threshold = st.sidebar.slider(
    "Max Feasible Risk Threshold ($R_{max}$)",
    min_value=0.20,
    max_value=1.00,
    value=0.65,
    step=0.05,
    help="Hard upper bound for segment ice risk. Segments exceeding this are marked infeasible."
)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    alpha_time = st.number_input(
        "Time Weight (α)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.2,
        help="Weight on transit time."
    )
with col_b:
    beta_risk = st.number_input(
        "Risk Weight (β)",
        min_value=0.0,
        max_value=200.0,
        value=60.0,
        step=10.0,
        help="Weight on cumulative ice exposure."
    )

grid_res = st.sidebar.selectbox(
    "Graph Grid Resolution",
    ["Standard (0.35° lat, 0.75° lon)", "Fine (0.25° lat, 0.50° lon)", "Fast (0.50° lat, 1.00° lon)"],
    index=0
)
res_map = {
    "Standard (0.35° lat, 0.75° lon)": (0.35, 0.75),
    "Fine (0.25° lat, 0.50° lon)": (0.25, 0.50),
    "Fast (0.50° lat, 1.00° lon)": (0.50, 1.00)
}
d_lat, d_lon = res_map[grid_res]


# ---------------------------------------------------------
# Route Computation
# ---------------------------------------------------------
algo_key = "astar" if "A*" in algorithm_choice else "dijkstra"

with st.spinner("Computing Great-Circle baseline and optimizing risk-aware route..."):
    # 1. Baseline Route
    baseline_route = generate_great_circle_baseline(
        origin=origin,
        destination=destination,
        risk_model=risk_model,
        base_speed_knots=vessel_speed,
        max_risk_threshold=max_risk_threshold,
        num_samples=100
    )

    # 2. Proposed Route
    try:
        proposed_route = optimize_route(
            origin=origin,
            destination=destination,
            risk_model=risk_model,
            algorithm=algo_key,
            base_speed_knots=vessel_speed,
            alpha_time=alpha_time,
            beta_risk=beta_risk,
            max_risk_threshold=max_risk_threshold,
            grid_lat_step=d_lat,
            grid_lon_step=d_lon
        )
    except Exception as e:
        st.error(f"Routing Error: {str(e)}")
        st.stop()

# 3. Evaluation & Trade-off Metrics
df_comparison, eval_summary = compare_routes(baseline_route, proposed_route)
tradeoffs = eval_summary["tradeoffs"]


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown("<div class='main-title'>🚢 Ice-Risk Aware Route Decision Support System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>CA-1 Project · Multi-Objective Graph Route Optimization vs. Mandatory Great-Circle Baseline</div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Summary Metrics Cards
# ---------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    time_delta = f"{tradeoffs['time_penalty_pct']:+.1f}% vs Baseline"
    st.metric(
        label="Transit Time",
        value=f"{proposed_route.total_transit_time_hours:.1f} hrs",
        delta=time_delta,
        delta_color="inverse"
    )

with m2:
    risk_delta = f"{tradeoffs['risk_reduction_pct']:.1f}% reduction"
    st.metric(
        label="Cumulative Ice Risk",
        value=f"{proposed_route.cumulative_ice_risk:.1f}",
        delta=risk_delta,
        delta_color="normal"
    )

with m3:
    max_r = proposed_route.max_segment_risk
    st.metric(
        label=f"Max Segment Risk (Limit: {max_risk_threshold})",
        value=f"{max_r:.2f}",
        delta=f"-{(baseline_route.max_segment_risk - max_r):.2f} pts" if baseline_route.max_segment_risk > max_r else "0.00",
        delta_color="normal"
    )

with m4:
    feasibility_status = "PASSED" if proposed_route.feasibility else "THRESHOLD EXCEEDED"
    status_color = "🟢" if proposed_route.feasibility else "🔴"
    st.metric(
        label="Feasibility Status",
        value=f"{status_color} {feasibility_status}",
        delta="Feasible" if proposed_route.feasibility else "Violates Constraint"
    )

st.markdown("---")


# ---------------------------------------------------------
# Map Section (Interactive Folium)
# ---------------------------------------------------------
st.subheader("🗺️ Geospatial Route Overlay & Ice-Risk Field")
st.caption("🔴 Dashed Red Line: Shortest Great-Circle Baseline | 🔵 Solid Blue Line: Proposed Risk-Aware Route | Colored Disks: Spatial Ice Risk")

folium_map = create_folium_map(
    risk_model=risk_model,
    baseline_route=baseline_route,
    proposed_route=proposed_route,
    max_risk_threshold=max_risk_threshold,
    downsample_factor=2
)

# Render Folium map in Streamlit
st_folium(folium_map, width="100%", height=520, returned_objects=[])


# ---------------------------------------------------------
# Tabbed Analysis & Results
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Performance Evaluation (Table 7)",
    "📈 Route Risk Profile & Charts",
    "🧭 Waypoints & Segment Details",
    "❄️ Ice Dataset & Exploratory Analysis"
])

with tab1:
    st.markdown("### Results Against the Baseline (Report Table 7)")
    st.markdown("Direct evaluation of both routes across the exact same spatial ice-risk field:")
    
    def highlight_feasibility(val):
        if val == "Pass":
            return "color: #2dc653; font-weight: bold"
        elif val == "Fail":
            return "color: #d90429; font-weight: bold"
        return ""

    if hasattr(df_comparison.style, "map"):
        styled_df = df_comparison.style.map(highlight_feasibility, subset=["Feasibility"])
    else:
        styled_df = df_comparison.style.applymap(highlight_feasibility, subset=["Feasibility"])
    st.dataframe(styled_df, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.info(
            f"**Operational Trade-off Metrics:**\n"
            f"- **Ice Risk Reduction:** `{tradeoffs['risk_reduction_pct']}%` lower cumulative exposure.\n"
            f"- **Peak Risk Reduction:** `{tradeoffs['max_risk_reduction_pct']}%` reduction in maximum segment risk.\n"
            f"- **Time Penalty:** `{tradeoffs['time_penalty_pct']:+.1f}%` additional transit hours.\n"
            f"- **Distance Penalty:** `{tradeoffs['distance_penalty_pct']:+.1f}%` detour distance ({proposed_route.total_distance_km - baseline_route.total_distance_km:+.1f} km)."
        )
    with col_t2:
        st.success(
            f"**Decision Support Summary:**\n"
            f"- The Great-Circle baseline achieves minimum geometric distance ({baseline_route.total_distance_km:.1f} km), but penetrates high-risk ice (peak risk `{baseline_route.max_segment_risk:.2f}`).\n"
            f"- The proposed `{proposed_route.algorithm}` route maneuvers around the heaviest ice barriers, keeping segment risks below `{max_risk_threshold}` and ensuring operational feasibility."
        )

    # Markdown export for project report submission
    with st.expander("📋 Copy Formatted Markdown Table for Report Submission"):
        st.code(format_comparison_table(df_comparison, tradeoffs), language="markdown")


with tab2:
    col_c1, col_c2 = st.columns([3, 2])
    with col_c1:
        fig_profile = create_risk_profile_figure(baseline_route, proposed_route, max_risk_threshold)
        st.plotly_chart(fig_profile, use_container_width=True)
    with col_c2:
        fig_bars = create_metrics_bar_chart(baseline_route, proposed_route)
        st.plotly_chart(fig_bars, use_container_width=True)


with tab3:
    st.markdown("### Waypoints and Route Segments")
    seg_choice = st.radio("Inspect Waypoints for:", ["Proposed Route", "Baseline Route"], horizontal=True)
    
    target_route = proposed_route if seg_choice == "Proposed Route" else baseline_route
    df_segs = pd.DataFrame(target_route.segments)

    if not df_segs.empty:
        # Display formatted columns
        display_df = pd.DataFrame({
            "Segment": df_segs["segment_idx"] + 1,
            "Start Lat": [f"{pt[0]:.3f}" for pt in df_segs["start"]],
            "Start Lon": [f"{pt[1]:.3f}" for pt in df_segs["start"]],
            "End Lat": [f"{pt[0]:.3f}" for pt in df_segs["end"]],
            "End Lon": [f"{pt[1]:.3f}" for pt in df_segs["end"]],
            "Distance (km)": df_segs["distance_km"].round(1),
            "Ice Risk (0-1)": df_segs["ice_risk"].round(3),
            "Speed (knots)": df_segs["speed_knots"].round(1),
            "Time (hrs)": df_segs["time_hours"].round(2),
            "Ice Type": df_segs["ice_type"]
        })
        st.dataframe(display_df, use_container_width=True, height=350)

        # Download button
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download {seg_choice} Segments (CSV)",
            data=csv_data,
            file_name=f"{seg_choice.lower().replace(' ', '_')}_waypoints.csv",
            mime="text/csv"
        )


with tab4:
    st.markdown("### Exploratory Data Analysis (EDA) of Ice Field")
    raw_df = risk_model.get_dataframe()
    
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Total Observation Points", f"{len(raw_df):,}")
    col_e2.metric("Mean Regional Ice Risk", f"{raw_df['ice_risk'].mean():.2f}")
    col_e3.metric("Max Thickness Recorded", f"{raw_df['ice_thickness_m'].max():.2f} m")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_hist = px.histogram(
            raw_df,
            x="ice_risk",
            nbins=30,
            title="<b>Figure 2: Distribution of Ice-Risk Values</b>",
            color_discrete_sequence=["#0f4c81"]
        )
        fig_hist.update_layout(xaxis_title="Normalized Ice Risk", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_g2:
        fig_scatter = px.scatter(
            raw_df.sample(min(1200, len(raw_df))),
            x="longitude",
            y="latitude",
            color="ice_risk",
            color_continuous_scale="Viridis",
            title="<b>Figure 1: Spatial Ice-Risk Distribution</b>"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Sample Observation Records (Table 2 Fields)")
    st.dataframe(raw_df.head(10), use_container_width=True)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<small style='color: #6c757d;'>"
    "Ice-Risk Aware Route Decision Support System · Course Project CA-1 · "
    "Implemented with Python, NetworkX, Folium, and Streamlit. "
    "Results generated deterministically from the experimental spatial ice model."
    "</small>",
    unsafe_allow_html=True
)
