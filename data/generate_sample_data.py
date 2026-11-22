"""
Synthetic Arctic Ice Conditions Dataset Generator
Generates realistic spatial ice-condition data for the Barents Sea / Svalbard region.
Fields correspond to Table 2 in the project report:
- latitude, longitude
- ice_concentration (0.0 to 1.0)
- ice_thickness (meters)
- ice_type (string descriptor)
- ice_risk (normalized 0.0 to 1.0)
- timestamp
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone

def generate_ice_dataset(
    output_path: str = "data/ice_conditions.csv",
    lat_min: float = 68.0,
    lat_max: float = 80.0,
    lon_min: float = 10.0,
    lon_max: float = 50.0,
    lat_step: float = 0.25,
    lon_step: float = 0.50,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)
    
    lats = np.arange(lat_min, lat_max + lat_step / 2.0, lat_step)
    lons = np.arange(lon_min, lon_max + lon_step / 2.0, lon_step)
    
    records = []
    current_time = datetime.now(timezone.utc).isoformat()
    
    for lat in lats:
        for lon in lons:
            # 1. Base latitude gradient: ice increases as we go north
            # Below 71N, mostly open water due to Atlantic inflow
            lat_factor = np.clip((lat - 71.0) / (79.0 - 71.0), 0.0, 1.0)
            
            # 2. Longitude gradient: eastern Barents / Kara side is colder than western Norwegian Sea
            lon_factor = np.clip((lon - 15.0) / (45.0 - 15.0), 0.0, 1.0)
            
            # 3. Modeled ice ridge / pack tongue (between 73N-76N and 24E-36E)
            # Represents a high-risk drift ice barrier directly crossing standard great-circle corridors
            dist_to_barrier_center = np.hypot((lat - 74.5) / 1.5, (lon - 28.0) / 4.5)
            barrier_boost = np.exp(-0.5 * (dist_to_barrier_center ** 2)) * 0.75
            
            # Combined concentration baseline
            raw_conc = 0.65 * lat_factor + 0.25 * lon_factor + barrier_boost
            # Add subtle natural spatial noise
            noise = np.random.normal(0, 0.04)
            concentration = float(np.clip(raw_conc + noise, 0.0, 0.98))
            
            # If concentration is very low, treat as open water
            if concentration < 0.08:
                concentration = 0.0
                thickness = 0.0
                ice_type = "Open Water"
                risk = 0.0
            else:
                # Thickness correlates with concentration and latitude
                base_thickness = concentration * 1.8 + (lat_factor * 0.6)
                thickness = float(np.clip(base_thickness + np.random.normal(0, 0.08), 0.1, 2.5))
                
                # Ice type classification based on WMO standards
                if thickness < 0.3:
                    ice_type = "Thin First-Year Ice"
                    type_penalty = 0.2
                elif thickness < 1.0:
                    ice_type = "Medium First-Year Ice"
                    type_penalty = 0.5
                elif thickness < 1.8:
                    ice_type = "Thick First-Year Ice"
                    type_penalty = 0.8
                else:
                    ice_type = "Multi-Year Ice"
                    type_penalty = 1.0
                
                # Multi-factor normalized risk (0.0 to 1.0)
                # Weights: 45% concentration, 35% thickness, 20% ice type
                normalized_thickness = np.clip(thickness / 2.2, 0.0, 1.0)
                risk = float(np.clip(
                    0.45 * concentration + 0.35 * normalized_thickness + 0.20 * type_penalty,
                    0.0, 1.0
                ))
            
            records.append({
                "latitude": round(float(lat), 3),
                "longitude": round(float(lon), 3),
                "timestamp": current_time,
                "ice_concentration": round(concentration, 3),
                "ice_thickness_m": round(thickness, 2),
                "ice_type": ice_type,
                "ice_risk": round(risk, 3),
                "feasibility_flag": True if risk < 0.85 else False
            })
            
    df = pd.DataFrame(records)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} ice observation points in {output_path}")
    return df

if __name__ == "__main__":
    generate_ice_dataset()
