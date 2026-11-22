# Ice-Risk Aware Route Decision Support System (DSS)

**Course:** CA-1 Project Report Implementation  
**Student:** G Midhun Sainathh  
**Stack:** Python 3.10+, NetworkX, Folium, Streamlit, Plotly, Pandas, NumPy  

---

## 📌 Executive Summary

Navigating polar and sub-polar maritime corridors cannot rely purely on geometric shortest paths. A straight-line great-circle route frequently traverses heavy pack ice, ice ridges, and hazardous concentration tongues, leading to severe hull risk, structural stress, or entrapment.

This project implements the complete **Ice-Risk Aware Route Decision Support System (DSS)** described in the CA-1 report:
1. **Spatial Ice-Risk Field Model**: Ingests, validates, and normalizes multi-factor ice observations (concentration, thickness, ice type) into a continuous $[0, 1]$ risk score.
2. **Mandatory Great-Circle Baseline**: Computes the shortest great-circle arc via the spherical Haversine formula and evaluates it across the exact same spatial ice-risk field.
3. **Graph-Based Route Optimization Engine**: Constructs an 8-connected weighted spatial graph using NetworkX and identifies optimal routes using **Dijkstra's Algorithm** and **A\* Search** under configurable multi-objective weights ($\alpha \cdot \text{Time} + \beta \cdot \text{Risk}$) and hard feasibility risk constraints ($R_{max}$).
4. **Comparative Evaluation**: Directly outputs **Table 7** of the project report, calculating transit time, cumulative ice risk, maximum segment risk, route feasibility, **Risk Reduction (%)**, and **Time Penalty (%)**.
5. **Interactive Dashboard**: A full-featured web DSS built with **Streamlit**, interactive map overlays via **Folium**, and dynamic analytical charts via **Plotly**.

---

## 🗂️ Project Repository Structure

```
ice_route_dss/
├── app.py                     # Streamlit interactive decision support system dashboard
├── run_evaluation.py          # Headless CLI benchmark and Table 7 generator
├── requirements.txt           # Python library dependencies
├── README.md                  # Comprehensive documentation and reproduction guide
├── data/
│   ├── generate_sample_data.py # Realistic Arctic ice dataset generator (Table 2 schema)
│   ├── ice_conditions.csv      # Spatial ice field observations (3,969 grid points)
│   └── table_7_results.csv     # Exported benchmark evaluation results
├── src/
│   ├── __init__.py            # Package initialization
│   ├── risk_model.py          # Ice field ingestion, normalization, and IDW spatial queries
│   ├── baseline.py            # Spherical Great-Circle router (Haversine & SLERP)
│   ├── routing.py             # NetworkX spatial graph constructor, Dijkstra & A* optimizers
│   ├── evaluation.py          # Performance metrics, trade-offs, and Table 7 formatter
│   └── visualization.py       # Folium dual-route overlays & Plotly risk profile charts
└── tests/
    ├── __init__.py
    └── test_routing.py        # Automated test suite (Haversine, Graph, Heuristic, Tradeoffs)
```

---

## 📐 Mathematical Formulations

### 1. Great-Circle Distance (Haversine Formula)
For points $(\phi_1, \lambda_1)$ and $(\phi_2, \lambda_2)$:
$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\Delta\lambda}{2}\right)$$
$$d = 2 R_e \arcsin\left(\min(1.0, \sqrt{a})\right), \quad R_e \approx 6371.0088\text{ km}$$

### 2. Composite Ice-Risk Score
Normalized into $[0, 1]$ from observation variables:
$$R_{ice} = w_c \cdot C_{ice} + w_t \cdot \min\left(\frac{T_{ice}}{T_{ref}}, 1.0\right) + w_y \cdot Y_{type}$$
*Default weights:* $w_c = 0.45$ (concentration), $w_t = 0.35$ (thickness), $w_y = 0.20$ (ice type penalty).

### 3. Vessel Speed Retardation in Ice
Vessel cruising speed is dynamically degraded as a function of local ice resistance:
$$v_{eff}(r) = v_{base} \cdot \max(0.15, 1.0 - 0.75 \cdot r)$$
$$\Delta t_i = \frac{\Delta d_i}{v_{eff}(r_i)}$$

### 4. Graph Edge Cost Objective
For edge $(u, v)$ with distance $\Delta d$ and segment risk $r_{uv} = \frac{r_u + r_v}{2}$:
$$\text{Cost}_{uv} = \alpha \cdot \Delta t_{uv} + \beta \cdot (r_{uv} \cdot \Delta d_{uv}) + P$$
where $P$ is a penalty applied if $r_{uv} > R_{max}$ (when hard constraints are relaxed for soft fallback).

### 5. A* Admissible Heuristic
For node $u$ and destination $T$:
$$h(u, T) = \alpha \cdot \left(\frac{\text{haversine}(u, T)}{v_{base}}\right)$$
Because $v_{eff} \le v_{base}$, $h(u, T)$ never overestimates the actual remaining travel time cost, guaranteeing admissibility and optimality.

### 6. Comparative Trade-off Formulas (Report Section 7.2)
$$\text{Risk Reduction (\%)} = \frac{\text{Baseline Risk} - \text{Proposed Risk}}{\text{Baseline Risk}} \times 100\%$$
$$\text{Time Penalty (\%)} = \frac{\text{Proposed Time} - \text{Baseline Time}}{\text{Baseline Time}} \times 100\%$$

---

## 🚀 Quickstart & How to Run

### 1. Prerequisites
Ensure Python 3.10 or higher is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
Execute the unit and integration tests:
```bash
python -m unittest discover -s tests -v
```

### 4. Run Headless CLI Evaluation (Generates Report Table 7)
```bash
python run_evaluation.py
```

### 5. Launch Interactive Streamlit DSS Dashboard
```bash
streamlit run app.py
```
The application will open in your browser (typically at `http://localhost:8501`).

---

## 📊 Sample Experimental Output (Table 7)

Tested scenario: **Tromsø (69.65°N, 18.96°E) to Svalbard / Central Ice Crossing (76.00°N, 36.00°E)** with $R_{max} = 0.75$:

| Route | Transit time | Cumulative ice risk | Max. risk | Avg. risk | Distance (km) | Distance (NM) | Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline: shortest great-circle** | 68.9 hrs | 481.8 | 0.98 | 0.61 | 792.8 | 428.1 | **Fail** |
| **Proposed: risk-aware route (DIJKSTRA)** | 74.9 hrs | 337.5 | 0.72 | 0.22 | 1518.8 | 820.1 | **Pass** |

### Evaluation Metrics
- **Cumulative Risk Reduction:** `29.95%` lower total ice exposure.
- **Peak Risk Reduction:** `25.82%` drop in maximum segment risk.
- **Average Route Risk:** Reduced from `0.61` down to `0.22`.
- **Transit Time Penalty:** `+8.7%` additional transit duration.
- **Feasibility:** Baseline penetrates impassable 0.98 pack ice (**Fail**), whereas the proposed route safely navigates below the 0.75 threshold (**Pass**).
