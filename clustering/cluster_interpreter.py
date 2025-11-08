"""
Cluster Interpretation Module
Automatically interprets and labels clusters based on their characteristics
Uses NORMALIZED data for comparison across all 3 variables: IPM, Pengeluaran, Garis Kemiskinan
"""

import numpy as np
from typing import Dict, List, Any
from .constants import (
    COLUMN_IPM,
    COLUMN_GARIS_KEMISKINAN,
    COLUMN_PENGELUARAN_PER_KAPITA,
    THRESHOLD_LOW,
    THRESHOLD_HIGH,
    RATIO_BELOW_POVERTY,
    RATIO_SLIGHTLY_ABOVE,
    RATIO_WELL_ABOVE,
)


def interpret_cluster_label(centroid: Dict[str, float], all_centroids: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Interpret cluster based on its centroid values relative to other clusters
    USING NORMALIZED VALUES for IPM, Pengeluaran, AND Garis Kemiskinan
    
    Args:
        centroid: The centroid of the cluster to interpret
        all_centroids: All cluster centroids for comparison
    
    Returns:
        Dictionary with label and description
    """
    
    # Extract values
    ipm = centroid.get(COLUMN_IPM, 0)
    garis_kemiskinan = centroid.get(COLUMN_GARIS_KEMISKINAN, 0)  # rupiah/kapita/bulan
    pengeluaran = centroid.get(COLUMN_PENGELUARAN_PER_KAPITA, 0)  # ribu rupiah/orang/tahun
    
    # Convert pengeluaran to rupiah/kapita/bulan for comparison
    pengeluaran_per_bulan = (pengeluaran * 1000) / 12  # rupiah/kapita/bulan
    
    # Calculate relative values across all clusters (NORMALIZED DATA)
    ipm_values = [c.get(COLUMN_IPM, 0) for c in all_centroids]
    pengeluaran_values = [c.get(COLUMN_PENGELUARAN_PER_KAPITA, 0) for c in all_centroids]
    garis_kemiskinan_values = [c.get(COLUMN_GARIS_KEMISKINAN, 0) for c in all_centroids]
    
    ipm_min = min(ipm_values)
    ipm_max = max(ipm_values)
    ipm_range = ipm_max - ipm_min
    
    pengeluaran_min = min(pengeluaran_values)
    pengeluaran_max = max(pengeluaran_values)
    pengeluaran_range = pengeluaran_max - pengeluaran_min
    
    gk_min = min(garis_kemiskinan_values)
    gk_max = max(garis_kemiskinan_values)
    gk_range = gk_max - gk_min
    
    # Normalize values (0-1 scale) - USING NORMALIZED DATA FOR COMPARISON
    if ipm_range > 0:
        ipm_normalized = (ipm - ipm_min) / ipm_range
    else:
        ipm_normalized = 0.5
    
    if pengeluaran_range > 0:
        pengeluaran_normalized = (pengeluaran - pengeluaran_min) / pengeluaran_range
    else:
        pengeluaran_normalized = 0.5
    
    if gk_range > 0:
        gk_normalized = (garis_kemiskinan - gk_min) / gk_range
    else:
        gk_normalized = 0.5
    
    # Compare pengeluaran with garis kemiskinan (ratio for fallback)
    ratio_to_poverty = pengeluaran_per_bulan / garis_kemiskinan if garis_kemiskinan > 0 else 1.0
    
    # Determine cluster label based on characteristics
    label = ""
    description = ""
    category = ""
    color_code = ""
    
    # Use thresholds from constants
    LOW = THRESHOLD_LOW
    HIGH = THRESHOLD_HIGH
    
    # Poverty line ratio thresholds (untuk fallback case)
    BELOW_POVERTY = RATIO_BELOW_POVERTY
    SLIGHTLY_ABOVE = RATIO_SLIGHTLY_ABOVE
    WELL_ABOVE = RATIO_WELL_ABOVE
    
    # NEW LOGIC: Use normalized values for all variables
    # Case 1: Daerah Maju dengan Biaya Hidup Mahal
    # IPM tinggi + Pengeluaran tinggi + Garis Kemiskinan tinggi
    if ipm_normalized > HIGH and pengeluaran_normalized > HIGH and gk_normalized > HIGH:
        label = "Daerah Maju Biaya Tinggi"
        category = "prosperous_high_cost"
        color_code = "#9333ea"  # Purple
        description = (
            f"Daerah maju dengan IPM tinggi ({ipm:.2f}), "
            f"pengeluaran per kapita tinggi ({pengeluaran:.0f} ribu/tahun atau "
            f"Rp {pengeluaran_per_bulan:,.0f}/bulan), dan "
            f"garis kemiskinan tinggi (Rp {garis_kemiskinan:,.0f}/bulan). "
            f"Daerah ini memiliki standar hidup tinggi dengan biaya hidup yang mahal, "
            f"karakteristik daerah urban maju seperti kota besar."
        )
    
    # Case 2: Daerah Sejahtera Biaya Rendah
    # IPM tinggi + Pengeluaran tinggi + Garis Kemiskinan rendah
    elif ipm_normalized > HIGH and pengeluaran_normalized > HIGH and gk_normalized < LOW:
        label = "Daerah Sejahtera Efisien"
        category = "prosperous"
        color_code = "#48bb78"  # Green
        description = (
            f"Daerah sejahtera dengan IPM tinggi ({ipm:.2f}), "
            f"pengeluaran per kapita tinggi ({pengeluaran:.0f} ribu/tahun atau "
            f"Rp {pengeluaran_per_bulan:,.0f}/bulan), namun "
            f"garis kemiskinan rendah (Rp {garis_kemiskinan:,.0f}/bulan). "
            f"Daerah ini memiliki kesejahteraan baik dengan biaya hidup terjangkau."
        )
    
    # Case 3: Cluster Miskin dengan Biaya Rendah
    # IPM rendah + Pengeluaran rendah + Garis Kemiskinan rendah
    elif ipm_normalized < LOW and pengeluaran_normalized < LOW and gk_normalized < LOW:
        label = "Daerah Tertinggal"
        category = "poor"
        color_code = "#f56565"  # Red
        description = (
            f"Daerah tertinggal dengan IPM rendah ({ipm:.2f}), "
            f"pengeluaran per kapita rendah ({pengeluaran:.0f} ribu/tahun atau "
            f"Rp {pengeluaran_per_bulan:,.0f}/bulan), dan "
            f"garis kemiskinan rendah (Rp {garis_kemiskinan:,.0f}/bulan). "
            f"Daerah ini memerlukan perhatian khusus untuk program pengentasan kemiskinan "
            f"dan peningkatan infrastruktur."
        )
    
    # Case 4: IPM rendah tapi biaya tinggi (problematic)
    # IPM rendah + Garis Kemiskinan tinggi
    elif ipm_normalized < LOW and gk_normalized > HIGH:
        label = "Daerah Rentan Biaya Tinggi"
        category = "vulnerable"
        color_code = "#ed8936"  # Orange
        description = (
            f"Daerah dengan IPM rendah ({ipm:.2f}) namun "
            f"garis kemiskinan tinggi (Rp {garis_kemiskinan:,.0f}/bulan), "
            f"sementara pengeluaran per kapita ({pengeluaran:.0f} ribu/tahun atau "
            f"Rp {pengeluaran_per_bulan:,.0f}/bulan). "
            f"Daerah ini menghadapi tantangan biaya hidup tinggi dengan kualitas SDM rendah, "
            f"memerlukan intervensi komprehensif."
        )
    
    # Case 5: IPM tinggi, pengeluaran sedang, gk tinggi
    elif ipm_normalized > HIGH and pengeluaran_normalized >= LOW and gk_normalized > HIGH:
        label = "Daerah Berkembang Biaya Tinggi"
        category = "developing"
        color_code = "#4299e1"  # Blue
        description = (
            f"Daerah dengan IPM tinggi ({ipm:.2f}), "
            f"pengeluaran per kapita ({pengeluaran:.0f} ribu/tahun atau "
            f"Rp {pengeluaran_per_bulan:,.0f}/bulan), dan "
            f"garis kemiskinan tinggi (Rp {garis_kemiskinan:,.0f}/bulan). "
            f"Daerah berkembang dengan biaya hidup tinggi."
        )
    
    # Case 6: Pengeluaran & GK tinggi (regardless of IPM)
    elif pengeluaran_normalized > HIGH and gk_normalized > HIGH:
        label = "Daerah Biaya Tinggi"
        category = "high_cost"
        color_code = "#a855f7"  # Light Purple
        description = (
            f"Pengeluaran per kapita tinggi ({pengeluaran:.0f} ribu/tahun atau "
            f"Rp {pengeluaran_per_bulan:,.0f}/bulan) dengan "
            f"garis kemiskinan tinggi (Rp {garis_kemiskinan:,.0f}/bulan), "
            f"IPM {ipm:.2f}. "
            f"Daerah dengan biaya hidup tinggi."
        )
    
    # Case 7: IPM tinggi (other cases)
    elif ipm_normalized > HIGH:
        label = "Daerah Berkembang"
        category = "developing"
        color_code = "#4299e1"  # Blue
        description = (
            f"IPM tinggi ({ipm:.2f}), "
            f"pengeluaran per kapita ({pengeluaran:.0f} ribu/tahun atau "
            f"Rp {pengeluaran_per_bulan:,.0f}/bulan), "
            f"garis kemiskinan (Rp {garis_kemiskinan:,.0f}/bulan). "
            f"Daerah dengan potensi pertumbuhan."
        )
    
    # Case 8: IPM rendah dengan pengeluaran di bawah garis kemiskinan
    elif ipm_normalized < LOW and ratio_to_poverty < BELOW_POVERTY:
        label = "Daerah Miskin"
        category = "poor"
        color_code = "#f56565"  # Red
        description = (
            f"IPM rendah ({ipm:.2f}), "
            f"pengeluaran per kapita ({pengeluaran:.0f} ribu/tahun atau "
            f"Rp {pengeluaran_per_bulan:,.0f}/bulan) berada di bawah garis kemiskinan "
            f"(Rp {garis_kemiskinan:,.0f}/bulan). "
            f"Daerah ini memerlukan perhatian khusus untuk program pengentasan kemiskinan."
        )
    
    else:
        # Default: Cluster Menengah
        label = "Daerah Menengah"
        category = "middle"
        color_code = "#ecc94b"  # Yellow
        
        description = (
            f"IPM sedang ({ipm:.2f}), "
            f"pengeluaran per kapita ({pengeluaran:.0f} ribu/tahun atau "
            f"Rp {pengeluaran_per_bulan:,.0f}/bulan), "
            f"garis kemiskinan (Rp {garis_kemiskinan:,.0f}/bulan). "
            f"Daerah dalam kondisi menengah dengan potensi untuk ditingkatkan."
        )
    
    # Additional metrics
    metrics = {
        'ipm_level': _get_ipm_level(ipm_normalized),
        'ipm_normalized': round(ipm_normalized, 3),
        'expenditure_level': _get_level(pengeluaran_normalized),
        'expenditure_normalized': round(pengeluaran_normalized, 3),
        'poverty_line_level': _get_level(gk_normalized),
        'poverty_line_normalized': round(gk_normalized, 3),
        'poverty_status': _get_poverty_status(ratio_to_poverty),
        'ipm_score': ipm,
        'poverty_line_ratio': ratio_to_poverty,
        'expenditure_per_month': pengeluaran_per_bulan,
        'poverty_line': garis_kemiskinan
    }
    
    return {
        'label': label,
        'category': category,
        'description': description,
        'color_code': color_code,
        'metrics': metrics
    }


def _get_ipm_level(ipm_normalized: float) -> str:
    """Get IPM level description"""
    if ipm_normalized < 0.33:
        return "Rendah"
    elif ipm_normalized < 0.67:
        return "Sedang"
    else:
        return "Tinggi"


def _get_level(normalized: float) -> str:
    """Get general level description for normalized value"""
    if normalized < 0.33:
        return "Rendah"
    elif normalized < 0.67:
        return "Sedang"
    else:
        return "Tinggi"


def _get_poverty_status(ratio: float) -> str:
    """Get poverty status description"""
    if ratio < 1.0:
        return "Di Bawah Garis Kemiskinan"
    elif ratio < 1.3:
        return "Sedikit Di Atas Garis Kemiskinan"
    elif ratio < 2.0:
        return "Di Atas Garis Kemiskinan"
    else:
        return "Jauh Di Atas Garis Kemiskinan"


def add_cluster_interpretations(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add interpretation labels to all clusters
    
    Args:
        clusters: List of cluster dictionaries
    
    Returns:
        Clusters with added interpretation
    """
    if not clusters:
        return clusters
    
    # Extract all centroids for comparison (skip None centroids for noise clusters)
    all_centroids = [cluster.get('centroid', {}) for cluster in clusters 
                     if cluster.get('centroid') is not None]
    
    # Skip interpretation if no valid centroids
    if not all_centroids:
        return clusters
    
    # Add interpretation to each cluster
    for cluster in clusters:
        centroid = cluster.get('centroid')
        # Only interpret clusters with valid centroid (skip noise clusters)
        if centroid is not None and centroid:
            interpretation = interpret_cluster_label(centroid, all_centroids)
            cluster['interpretation'] = interpretation
    
    return clusters


def get_cluster_summary_stats(clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get summary statistics of cluster interpretations
    
    Args:
        clusters: List of clusters with interpretations
    
    Returns:
        Summary statistics
    """
    if not clusters:
        return {}
    
    categories = {}
    total_regions = sum(cluster.get('size', 0) for cluster in clusters)
    
    for cluster in clusters:
        interpretation = cluster.get('interpretation', {})
        category = interpretation.get('category', 'unknown')
        size = cluster.get('size', 0)
        
        if category not in categories:
            categories[category] = {
                'count': 0,
                'regions': 0,
                'percentage': 0
            }
        
        categories[category]['count'] += 1
        categories[category]['regions'] += size
    
    # Calculate percentages
    for category in categories:
        if total_regions > 0:
            categories[category]['percentage'] = (categories[category]['regions'] / total_regions) * 100
    
    return {
        'total_clusters': len(clusters),
        'total_regions': total_regions,
        'categories': categories
    }
