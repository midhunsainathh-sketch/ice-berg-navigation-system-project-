"""
Ice Risk Model Module
Handles ingestion, validation, normalization, and spatial query of ice condition fields.
Conforms to Section 4 (Dataset) and Section 5.3 (Proposed Method) of the project report.
"""

from typing import Dict, Any, Tuple, Optional
import os
import numpy as np
import pandas as pd


class IceRiskModel:
    """
    Spatial Ice Risk Field model.
    Converts ice concentration, thickness, and type into normalized risk values [0, 1]
    and supports spatial queries via nearest-neighbor and Inverse Distance Weighting (IDW).
    """

    def __init__(
        self,
        data_source: Optional[str | pd.DataFrame] = None,
        weight_concentration: float = 0.45,
        weight_thickness: float = 0.35,
        weight_ice_type: float = 0.20,
        ref_thickness_max: float = 2.2
    ):
        self.w_c = weight_concentration
        self.w_t = weight_thickness
        self.w_y = weight_ice_type
        self.ref_thickness_max = ref_thickness_max
        
        self.df: pd.DataFrame = pd.DataFrame()
        self.coords: np.ndarray = np.empty((0, 2))
        self.risk_values: np.ndarray = np.empty(0)
        self.conc_values: np.ndarray = np.empty(0)
        self.thick_values: np.ndarray = np.empty(0)
        self.type_values: np.ndarray = np.empty(0, dtype=object)

        if data_source is not None:
            self.load_data(data_source)

    def load_data(self, data_source: str | pd.DataFrame) -> None:
        """Load and validate ice condition observations from CSV or DataFrame."""
        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Ice condition data file not found: {data_source}")
            df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise TypeError("data_source must be a file path string or pandas DataFrame")

        self.validate_and_process(df)

    def validate_and_process(self, df: pd.DataFrame) -> None:
        """Validate required columns, bounds, and calculate normalized risk field."""
        required = ["latitude", "longitude"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required spatial column in dataset: '{col}'")

        # Range checks
        if not df["latitude"].between(-90, 90).all():
            raise ValueError("Latitude values must be within [-90, 90]")
        if not df["longitude"].between(-180, 180).all():
            raise ValueError("Longitude values must be within [-180, 180]")

        # Provide defaults if specific ice condition attributes are missing
        if "ice_concentration" not in df.columns:
            df["ice_concentration"] = 0.0
        if "ice_thickness_m" not in df.columns:
            df["ice_thickness_m"] = 0.0
        if "ice_type" not in df.columns:
            df["ice_type"] = "Open Water"

        # Calculate or normalize ice_risk
        if "ice_risk" not in df.columns or df["ice_risk"].isnull().any():
            df["ice_risk"] = self.compute_composite_risk(
                df["ice_concentration"],
                df["ice_thickness_m"],
                df["ice_type"]
            )
        else:
            # Ensure risk is strictly clamped to [0, 1]
            df["ice_risk"] = df["ice_risk"].clip(0.0, 1.0)

        self.df = df.reset_index(drop=True)
        self.coords = self.df[["latitude", "longitude"]].to_numpy(dtype=float)
        self.risk_values = self.df["ice_risk"].to_numpy(dtype=float)
        self.conc_values = self.df["ice_concentration"].to_numpy(dtype=float)
        self.thick_values = self.df["ice_thickness_m"].to_numpy(dtype=float)
        self.type_values = self.df["ice_type"].to_numpy(dtype=object)

    def compute_composite_risk(
        self,
        concentration: pd.Series | np.ndarray,
        thickness: pd.Series | np.ndarray,
        ice_type: pd.Series | np.ndarray
    ) -> np.ndarray:
        """
        Calculates normalized risk R in [0, 1] combining concentration, thickness, and ice type:
        R = w_c * C + w_t * (T / T_ref) + w_y * TypePenalty
        """
        conc = np.clip(np.asarray(concentration, dtype=float), 0.0, 1.0)
        thick = np.clip(np.asarray(thickness, dtype=float) / self.ref_thickness_max, 0.0, 1.0)
        
        type_mapping = {
            "open water": 0.0,
            "thin first-year ice": 0.25,
            "medium first-year ice": 0.55,
            "thick first-year ice": 0.85,
            "multi-year ice": 1.00
        }
        
        type_penalty = np.array([
            type_mapping.get(str(t).strip().lower(), 0.5) for t in ice_type
        ], dtype=float)

        risk = (self.w_c * conc) + (self.w_t * thick) + (self.w_y * type_penalty)
        return np.clip(risk, 0.0, 1.0)

    def get_risk_at(self, lat: float, lon: float, method: str = "idw", k: int = 4) -> Dict[str, Any]:
        """
        Query ice conditions and risk score at an arbitrary latitude and longitude.
        Uses Inverse Distance Weighting (IDW) or Nearest Neighbor.
        """
        if len(self.coords) == 0:
            return {
                "ice_risk": 0.0,
                "ice_concentration": 0.0,
                "ice_thickness_m": 0.0,
                "ice_type": "Open Water"
            }

        # Approximate equirectangular distance squared in degrees
        lat_rad = np.radians(lat)
        d_lat = self.coords[:, 0] - lat
        d_lon = (self.coords[:, 1] - lon) * np.cos(lat_rad)
        d2 = d_lat**2 + d_lon**2

        if method == "nearest" or k == 1:
            idx = int(np.argmin(d2))
            return {
                "ice_risk": float(self.risk_values[idx]),
                "ice_concentration": float(self.conc_values[idx]),
                "ice_thickness_m": float(self.thick_values[idx]),
                "ice_type": str(self.type_values[idx])
            }

        # Inverse Distance Weighting over k nearest neighbors
        k_nearest = min(k, len(d2))
        if k_nearest >= len(d2):
            near_indices = np.argsort(d2)[:k_nearest]
        else:
            near_indices = np.argpartition(d2, k_nearest - 1)[:k_nearest]
        near_d2 = d2[near_indices]
        
        # Check if query point is on top of a grid point
        zero_mask = near_d2 < 1e-9
        if np.any(zero_mask):
            exact_idx = near_indices[np.argmax(zero_mask)]
            return {
                "ice_risk": float(self.risk_values[exact_idx]),
                "ice_concentration": float(self.conc_values[exact_idx]),
                "ice_thickness_m": float(self.thick_values[exact_idx]),
                "ice_type": str(self.type_values[exact_idx])
            }

        distances = np.sqrt(near_d2)
        weights = 1.0 / (distances ** 2)
        total_w = np.sum(weights)

        interpolated_risk = float(np.sum(weights * self.risk_values[near_indices]) / total_w)
        interpolated_conc = float(np.sum(weights * self.conc_values[near_indices]) / total_w)
        interpolated_thick = float(np.sum(weights * self.thick_values[near_indices]) / total_w)
        
        # Ice type from nearest point
        nearest_idx = near_indices[int(np.argmin(distances))]
        nearest_type = str(self.type_values[nearest_idx])

        return {
            "ice_risk": float(np.clip(interpolated_risk, 0.0, 1.0)),
            "ice_concentration": float(np.clip(interpolated_conc, 0.0, 1.0)),
            "ice_thickness_m": float(max(0.0, interpolated_thick)),
            "ice_type": nearest_type
        }

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns (lat_min, lat_max, lon_min, lon_max)."""
        if len(self.coords) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(self.df["latitude"].min()),
            float(self.df["latitude"].max()),
            float(self.df["longitude"].min()),
            float(self.df["longitude"].max())
        )

    def get_dataframe(self) -> pd.DataFrame:
        """Return the underlying validated DataFrame."""
        return self.df
