#!/usr/bin/env python3
"""
Debug script to test clustering algorithms and identify why all data goes to cluster 0
"""

import pandas as pd
import numpy as np
from clustering.algorithms import ClusteringAlgorithms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_test_data():
    """Create test data with clear clusters"""
    np.random.seed(42)
    
    # Create 3 distinct groups
    group1 = {
        'kabupaten_kota': [f'Kota_A_{i}' for i in range(10)],
        'tahun': [2023] * 10,
        'ipm': np.random.normal(85, 2, 10),  # High IPM
        'garis_kemiskinan': np.random.normal(400000, 50000, 10),  # Low poverty line
        'pengeluaran_per_kapita': np.random.normal(15000000, 1000000, 10)  # High expenditure
    }
    
    group2 = {
        'kabupaten_kota': [f'Kota_B_{i}' for i in range(10)],
        'tahun': [2023] * 10,
        'ipm': np.random.normal(75, 2, 10),  # Medium IPM
        'garis_kemiskinan': np.random.normal(500000, 50000, 10),  # Medium poverty line
        'pengeluaran_per_kapita': np.random.normal(10000000, 1000000, 10)  # Medium expenditure
    }
    
    group3 = {
        'kabupaten_kota': [f'Kota_C_{i}' for i in range(10)],
        'tahun': [2023] * 10,
        'ipm': np.random.normal(65, 2, 10),  # Low IPM
        'garis_kemiskinan': np.random.normal(600000, 50000, 10),  # High poverty line
        'pengeluaran_per_kapita': np.random.normal(7000000, 1000000, 10)  # Low expenditure
    }
    
    # Combine all groups
    df1 = pd.DataFrame(group1)
    df2 = pd.DataFrame(group2)
    df3 = pd.DataFrame(group3)
    
    df = pd.concat([df1, df2, df3], ignore_index=True)
    return df

def debug_fcm_clustering():
    """Debug FCM clustering step by step"""
    print("=== FCM Clustering Debug ===")
    
    # Create test data
    df = create_test_data()
    print(f"Created test data with {len(df)} rows")
    print("\nData summary:")
    print(df[['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita']].describe())
    
    # Initialize clustering
    clustering = ClusteringAlgorithms()
    features = ['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita']
    
    # Preprocess data
    scaled_data, df_clean = clustering.preprocess_data(df, features)
    print(f"\nScaled data shape: {scaled_data.shape}")
    print("Scaled data summary:")
    print(f"Min: {scaled_data.min(axis=0)}")
    print(f"Max: {scaled_data.max(axis=0)}")
    print(f"Mean: {scaled_data.mean(axis=0)}")
    print(f"Std: {scaled_data.std(axis=0)}")
    
    # Test FCM clustering
    import skfuzzy as fuzz
    
    # Transpose data for skfuzzy
    data_T = scaled_data.T
    print(f"\nTransposed data shape: {data_T.shape}")
    
    # Perform FCM clustering
    n_clusters = 3
    m = 2.0
    error = 1e-5
    max_iter = 300
    
    print(f"\nRunning FCM with {n_clusters} clusters...")
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_T, n_clusters, m, error=error, maxiter=max_iter
        )
        
        print(f"FCM completed in {p} iterations")
        print(f"Partition coefficient: {fpc}")
        
        # Get cluster assignments
        cluster_labels = np.argmax(u, axis=0)
        print(f"\nCluster assignments: {cluster_labels}")
        print(f"Unique clusters: {np.unique(cluster_labels)}")
        print(f"Cluster counts: {np.bincount(cluster_labels)}")
        
        # Check membership matrix
        print(f"\nMembership matrix shape: {u.shape}")
        print("Membership matrix (first 10 samples):")
        for i in range(min(10, u.shape[1])):
            print(f"Sample {i}: {u[:, i]}")
        
        # Check centroids
        print(f"\nCentroids shape: {cntr.shape}")
        print("Centroids (scaled):")
        for i, centroid in enumerate(cntr):
            print(f"Cluster {i}: {centroid}")
        
        # Transform centroids back to original scale
        print("\nCentroids (original scale):")
        original_centroids = clustering.scaler.inverse_transform(cntr)
        for i, centroid in enumerate(original_centroids):
            print(f"Cluster {i}: IPM={centroid[0]:.2f}, GK={centroid[1]:.0f}, PP={centroid[2]:.0f}")
        
        return True
        
    except Exception as e:
        print(f"FCM clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_optics_clustering():
    """Debug OPTICS clustering step by step"""
    print("\n=== OPTICS Clustering Debug ===")
    
    # Create test data
    df = create_test_data()
    
    # Initialize clustering
    clustering = ClusteringAlgorithms()
    features = ['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita']
    
    # Preprocess data
    scaled_data, df_clean = clustering.preprocess_data(df, features)
    
    # Test OPTICS clustering
    from sklearn.cluster import OPTICS
    
    min_samples = 5
    xi = 0.05
    min_cluster_size = 0.05
    
    print(f"Running OPTICS with min_samples={min_samples}, xi={xi}, min_cluster_size={min_cluster_size}")
    
    try:
        optics = OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size
        )
        
        cluster_labels = optics.fit_predict(scaled_data)
        
        print(f"OPTICS cluster assignments: {cluster_labels}")
        print(f"Unique clusters: {np.unique(cluster_labels)}")
        print(f"Cluster counts: {np.bincount(cluster_labels + 1) if -1 in cluster_labels else np.bincount(cluster_labels)}")
        
        # Check for noise points
        noise_points = np.sum(cluster_labels == -1)
        print(f"Noise points: {noise_points}")
        
        return True
        
    except Exception as e:
        print(f"OPTICS clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data():
    """Test with sample Indonesian data if available"""
    print("\n=== Testing with Real Data ===")
    
    try:
        # Try to load sample data
        sample_files = [
            'sample_data_indonesia.csv',
            'example_dataset_indonesia.xlsx'
        ]
        
        df = None
        for file_path in sample_files:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                print(f"Loaded data from {file_path}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            print("No sample data files found")
            return False
        
        print(f"Data shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Check if data has the required columns
        required_cols = ['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
        
        # Test clustering with real data
        clustering = ClusteringAlgorithms()
        
        try:
            results = clustering.fuzzy_c_means(
                df, 
                features=required_cols,
                n_clusters=3,
                m=2.0,
                max_iter=100,
                error=0.001
            )
            
            print(f"\nClustering results:")
            print(f"Algorithm: {results['algorithm']}")
            print(f"Total regions: {results['summary']['total_regions']}")
            print(f"Number of clusters: {results['summary']['num_clusters']}")
            print(f"Iterations: {results['summary']['iterations']}")
            
            for cluster in results['clusters']:
                print(f"Cluster {cluster['id']}: {cluster['size']} members")
                if cluster['size'] > 0:
                    centroid = cluster['centroid']
                    print(f"  Centroid: IPM={centroid['ipm']:.2f}, GK={centroid['garis_kemiskinan']:.0f}, PP={centroid['pengeluaran_per_kapita']:.0f}")
            
            return True
            
        except Exception as e:
            print(f"Real data clustering failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"Error testing with real data: {e}")
        return False

if __name__ == "__main__":
    print("Starting clustering debug...")
    
    # Test FCM
    fcm_success = debug_fcm_clustering()
    
    # Test OPTICS
    optics_success = debug_optics_clustering()
    
    # Test with real data
    real_data_success = test_with_real_data()
    
    print(f"\n=== Debug Summary ===")
    print(f"FCM test: {'PASSED' if fcm_success else 'FAILED'}")
    print(f"OPTICS test: {'PASSED' if optics_success else 'FAILED'}")
    print(f"Real data test: {'PASSED' if real_data_success else 'FAILED'}")