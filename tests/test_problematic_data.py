#!/usr/bin/env python3
"""
Test clustering with problematic data scenarios
"""

import pandas as pd
import numpy as np
from clustering.algorithms import ClusteringAlgorithms

def test_low_variance_data():
    """Test with data that has very low variance"""
    print("=== Testing Low Variance Data ===")
    
    # Create data with very low variance
    np.random.seed(42)
    data = {
        'kabupaten_kota': [f'Kota_{i}' for i in range(20)],
        'tahun': [2023] * 20,
        'ipm': [75.0] * 20,  # All same value
        'garis_kemiskinan': np.random.normal(500000, 1000, 20),  # Very low variance
        'pengeluaran_per_kapita': np.random.normal(8000000, 5000, 20)  # Very low variance
    }
    
    df = pd.DataFrame(data)
    print(f"Data shape: {df.shape}")
    print("Data variance:")
    print(df[['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita']].var())
    
    clustering = ClusteringAlgorithms()
    
    try:
        results = clustering.fuzzy_c_means(
            df, 
            features=['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita'],
            n_clusters=3
        )
        
        print(f"Clustering results:")
        for cluster in results['clusters']:
            print(f"Cluster {cluster['id']}: {cluster['size']} members")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_duplicate_data():
    """Test with duplicate data points"""
    print("\n=== Testing Duplicate Data ===")
    
    # Create data with many duplicates
    data = {
        'kabupaten_kota': [f'Kota_{i}' for i in range(15)],
        'tahun': [2023] * 15,
        'ipm': [75.0, 75.0, 75.0, 80.0, 80.0, 80.0, 70.0, 70.0, 70.0, 75.0, 75.0, 80.0, 80.0, 70.0, 70.0],
        'garis_kemiskinan': [500000] * 15,  # All same
        'pengeluaran_per_kapita': [8000000, 8000000, 8000000, 9000000, 9000000, 9000000, 7000000, 7000000, 7000000, 8000000, 8000000, 9000000, 9000000, 7000000, 7000000]
    }
    
    df = pd.DataFrame(data)
    print(f"Data shape: {df.shape}")
    print("Unique combinations:")
    print(df[['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita']].drop_duplicates())
    
    clustering = ClusteringAlgorithms()
    
    try:
        results = clustering.fuzzy_c_means(
            df, 
            features=['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita'],
            n_clusters=3
        )
        
        print(f"Clustering results:")
        for cluster in results['clusters']:
            print(f"Cluster {cluster['id']}: {cluster['size']} members")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_small_dataset():
    """Test with very small dataset"""
    print("\n=== Testing Small Dataset ===")
    
    # Create very small dataset
    data = {
        'kabupaten_kota': ['Kota_A', 'Kota_B', 'Kota_C'],
        'tahun': [2023, 2023, 2023],
        'ipm': [75.0, 80.0, 70.0],
        'garis_kemiskinan': [500000, 400000, 600000],
        'pengeluaran_per_kapita': [8000000, 9000000, 7000000]
    }
    
    df = pd.DataFrame(data)
    print(f"Data shape: {df.shape}")
    
    clustering = ClusteringAlgorithms()
    
    try:
        results = clustering.fuzzy_c_means(
            df, 
            features=['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita'],
            n_clusters=3  # Same number as data points
        )
        
        print(f"Clustering results:")
        for cluster in results['clusters']:
            print(f"Cluster {cluster['id']}: {cluster['size']} members")
            
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_year_vs_multi_year():
    """Test single year vs multi-year data"""
    print("\n=== Testing Single Year vs Multi-Year ===")
    
    # Create multi-year data
    np.random.seed(42)
    years = [2020, 2021, 2022, 2023]
    cities = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang']
    
    data = []
    for year in years:
        for city in cities:
            data.append({
                'kabupaten_kota': city,
                'tahun': year,
                'ipm': np.random.normal(75, 5),
                'garis_kemiskinan': np.random.normal(500000, 100000),
                'pengeluaran_per_kapita': np.random.normal(8000000, 1000000)
            })
    
    df = pd.DataFrame(data)
    print(f"Multi-year data shape: {df.shape}")
    print(f"Years: {sorted(df['tahun'].unique())}")
    
    clustering = ClusteringAlgorithms()
    
    # Test 1: All years together
    print("\n--- All years together ---")
    try:
        results = clustering.fuzzy_c_means(
            df, 
            features=['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita'],
            n_clusters=3
        )
        
        print(f"All years clustering results:")
        for cluster in results['clusters']:
            print(f"Cluster {cluster['id']}: {cluster['size']} members")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Single year only
    print("\n--- Single year (2023) only ---")
    try:
        results = clustering.fuzzy_c_means(
            df, 
            features=['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita'],
            n_clusters=3,
            selected_year='2023'
        )
        
        print(f"Single year clustering results:")
        for cluster in results['clusters']:
            print(f"Cluster {cluster['id']}: {cluster['size']} members")
    except Exception as e:
        print(f"Error: {e}")
    
    return True

if __name__ == "__main__":
    print("Testing problematic data scenarios...")
    
    test_low_variance_data()
    test_duplicate_data()
    test_small_dataset()
    test_single_year_vs_multi_year()
    
    print("\n=== All tests completed ===")