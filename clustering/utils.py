"""
Utility Functions for Clustering Application
Contains reusable helper functions for data processing and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .constants import (
    REQUIRED_COLUMNS,
    COLUMN_KABUPATEN_KOTA,
    COLUMN_TAHUN,
    CLUSTERING_FEATURES,
)


def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize column names to standard format and create mapping.
    
    Args:
        df: Input dataframe with potentially varying column names
        
    Returns:
        Tuple of (normalized dataframe, column mapping dictionary)
    """
    df_columns_lower = [
        col.lower().replace("/", "_").replace(" ", "_") for col in df.columns
    ]
    df_columns_map = {
        col.lower().replace("/", "_").replace(" ", "_"): col for col in df.columns
    }
    
    column_mapping = {}
    for required_col, possible_names in REQUIRED_COLUMNS.items():
        for possible_name in possible_names:
            normalized_name = (
                possible_name.lower().replace("/", "_").replace(" ", "_")
            )
            if normalized_name in df_columns_lower:
                actual_col_name = df_columns_map[normalized_name]
                column_mapping[required_col] = actual_col_name
                break
    
    # Rename columns to standardized names
    df_normalized = df.rename(columns={v: k for k, v in column_mapping.items()})
    
    return df_normalized, column_mapping


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    """
    Validate that all required columns are present in the dataframe.
    
    Args:
        df: Input dataframe to validate
        
    Returns:
        List of missing column names (empty if all required columns present)
    """
    missing_cols = []
    for required_col in REQUIRED_COLUMNS.keys():
        if required_col not in df.columns:
            missing_cols.append(required_col)
    return missing_cols


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate data for clustering.
    
    Args:
        df: Input dataframe
        
    Returns:
        Cleaned dataframe
        
    Raises:
        ValueError: If data validation fails
    """
    # Ensure tahun is numeric
    df[COLUMN_TAHUN] = pd.to_numeric(df[COLUMN_TAHUN], errors="coerce")
    
    # Ensure numeric columns are numeric
    for col in CLUSTERING_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Remove rows with invalid data
    df_clean = df.dropna(subset=[COLUMN_TAHUN] + CLUSTERING_FEATURES)
    
    if df_clean.empty:
        raise ValueError(
            "No valid data after cleaning. Ensure all numeric columns contain valid numbers."
        )
    
    # Ensure kabupaten_kota is string and not empty
    df_clean[COLUMN_KABUPATEN_KOTA] = (
        df_clean[COLUMN_KABUPATEN_KOTA].astype(str).str.strip()
    )
    df_clean = df_clean[df_clean[COLUMN_KABUPATEN_KOTA] != ""]
    
    if df_clean.empty:
        raise ValueError(
            "No valid data after cleaning. Ensure kabupaten_kota column is not empty."
        )
    
    return df_clean


def read_data_file(file_obj) -> pd.DataFrame:
    """
    Read data from uploaded file (CSV or Excel).
    
    Args:
        file_obj: Uploaded file object
        
    Returns:
        DataFrame with file contents
        
    Raises:
        ValueError: If file format is not supported or reading fails
    """
    filename = getattr(file_obj, "name", "").lower()
    
    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(file_obj, engine="openpyxl")
        elif filename.endswith(".csv"):
            return pd.read_csv(file_obj)
        else:
            # Try CSV first, then Excel
            try:
                return pd.read_csv(file_obj)
            except Exception:
                file_obj.seek(0)  # Reset file pointer
                return pd.read_excel(file_obj, engine="openpyxl")
    except Exception as e:
        raise ValueError(
            f"Failed to read file: {str(e)}. Ensure file is in CSV or Excel (.xlsx) format."
        )


def calculate_average_metrics(
    results_per_year: Dict, metric_name: str
) -> Optional[float]:
    """
    Calculate average of a metric across all years.
    
    Args:
        results_per_year: Dictionary of results per year
        metric_name: Name of the metric to average
        
    Returns:
        Average value or None if no valid values
    """
    successful_results = [r for r in results_per_year.values() if "error" not in r]
    
    values = [
        r["evaluation"][metric_name]
        for r in successful_results
        if r["evaluation"][metric_name] is not None
    ]
    
    return float(np.mean(values)) if values else None


def format_clustering_parameters(
    algorithm: str,
    num_clusters: int = None,
    fuzzy_coeff: float = None,
    max_iter: int = None,
    tolerance: float = None,
    min_samples: int = None,
    xi: float = None,
    min_cluster_size: float = None,
    selected_year: str = None,
) -> Dict:
    """
    Format clustering parameters into a standardized dictionary.
    
    Args:
        algorithm: Algorithm name
        num_clusters: Number of clusters (FCM)
        fuzzy_coeff: Fuzzy coefficient (FCM)
        max_iter: Maximum iterations (FCM)
        tolerance: Convergence tolerance (FCM)
        min_samples: Minimum samples (OPTICS)
        xi: Xi parameter (OPTICS)
        min_cluster_size: Minimum cluster size (OPTICS)
        selected_year: Selected year for filtering
        
    Returns:
        Dictionary of parameters
    """
    params = {"algorithm": algorithm}
    
    if num_clusters is not None:
        params["num_clusters"] = num_clusters
    if fuzzy_coeff is not None:
        params["fuzzy_coeff"] = fuzzy_coeff
    if max_iter is not None:
        params["max_iter"] = max_iter
    if tolerance is not None:
        params["tolerance"] = tolerance
    if min_samples is not None:
        params["min_samples"] = min_samples
    if xi is not None:
        params["xi"] = xi
    if min_cluster_size is not None:
        params["min_cluster_size"] = min_cluster_size
    if selected_year is not None:
        params["selected_year"] = selected_year
    
    return params


def safe_float_conversion(value, default: float = 0.0) -> float:
    """
    Safely convert value to float with default fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_int_conversion(value, default: int = 0) -> int:
    """
    Safely convert value to int with default fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default
