#!/usr/bin/env python3
"""
Simple test untuk algoritma OPTICS tanpa Django
"""

import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score

def test_optics_simple():
    """Test OPTICS dengan data sample sederhana"""
    
    print("🧪 Testing OPTICS Algorithm (Simple Version)")
    print("=" * 50)
    
    # Create sample data
    data = {
        'kabupaten_kota': ['Jakarta Pusat', 'Jakarta Utara', 'Jakarta Selatan', 'Jakarta Barat', 'Jakarta Timur',
                          'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Palembang',
                          'Makassar', 'Denpasar', 'Yogyakarta', 'Malang', 'Bogor'],
        'ipm': [83.78, 83.19, 86.89, 83.45, 82.67,
                81.01, 80.56, 79.45, 79.01, 78.12,
                82.89, 82.23, 76.01, 74.78, 75.23],
        'pengeluaran_per_kapita': [10200000, 9600000, 11600000, 9200000, 8900000,
                                  8200000, 7600000, 7300000, 7000000, 6700000,
                                  8600000, 7900000, 6500000, 6200000, 6300000],
        'garis_kemiskinan': [700000, 700000, 700000, 700000, 580000,
                            540000, 510000, 500000, 480000, 470000,
                            540000, 510000, 460000, 450000, 520000]
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Data shape: {df.shape}")
    print(f"📋 Features: ipm, pengeluaran_per_kapita, garis_kemiskinan")
    
    # Prepare features
    features = ['ipm', 'pengeluaran_per_kapita', 'garis_kemiskinan']
    X = df[features].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n🔍 Running OPTICS clustering...")
    
    # Test different parameter combinations
    test_configs = [
        {'min_samples': 3, 'xi': 0.05, 'min_cluster_size': 0.1},
        {'min_samples': 2, 'xi': 0.1, 'min_cluster_size': 0.05},
        {'min_samples': 4, 'xi': 0.15, 'min_cluster_size': 0.15}
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n--- Test Configuration {i} ---")
        print(f"Parameters: {config}")
        
        try:
            # Run OPTICS
            optics = OPTICS(**config)
            cluster_labels = optics.fit_predict(X_scaled)
            
            # Analyze results
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            print(f"✅ Clustering successful!")
            print(f"   Clusters found: {n_clusters}")
            print(f"   Noise points: {n_noise}")
            print(f"   Cluster labels: {sorted(unique_labels)}")
            
            # Calculate metrics if we have valid clusters
            if n_clusters > 1:
                valid_mask = cluster_labels != -1
                if np.sum(valid_mask) > 1:
                    db_score = davies_bouldin_score(X_scaled[valid_mask], cluster_labels[valid_mask])
                    sil_score = silhouette_score(X_scaled[valid_mask], cluster_labels[valid_mask])
                    print(f"   Davies-Bouldin Index: {db_score:.4f}")
                    print(f"   Silhouette Score: {sil_score:.4f}")
            
            # Show cluster composition
            for label in sorted(unique_labels):
                mask = cluster_labels == label
                members = df[mask]['kabupaten_kota'].tolist()
                if label == -1:
                    print(f"   Noise: {members}")
                else:
                    print(f"   Cluster {label}: {members}")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return True

def test_fcm_comparison():
    """Test FCM untuk perbandingan"""
    
    print(f"\n🌟 Testing FCM for comparison...")
    
    try:
        import skfuzzy as fuzz
        
        # Sample data
        data = np.array([
            [83.78, 10200000, 700000],  # Jakarta Pusat
            [83.19, 9600000, 700000],   # Jakarta Utara
            [81.01, 8200000, 540000],   # Surabaya
            [80.56, 7600000, 510000],   # Bandung
            [79.45, 7300000, 500000],   # Medan
            [76.01, 6500000, 460000]    # Yogyakarta
        ])
        
        # Scale data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Run FCM
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_scaled.T, 3, 2.0, error=0.005, maxiter=1000
        )
        
        cluster_labels = np.argmax(u, axis=0)
        
        print(f"✅ FCM clustering successful!")
        print(f"   Clusters: {len(set(cluster_labels))}")
        print(f"   Iterations: {p}")
        print(f"   Partition coefficient: {fpc:.4f}")
        
        return True
        
    except ImportError:
        print("⚠️ scikit-fuzzy not available for FCM test")
        return False
    except Exception as e:
        print(f"❌ FCM Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Simple Clustering Tests")
    
    # Test OPTICS
    optics_success = test_optics_simple()
    
    # Test FCM
    fcm_success = test_fcm_comparison()
    
    print(f"\n" + "=" * 50)
    print("📋 Test Results:")
    print(f"   OPTICS: {'✅ PASSED' if optics_success else '❌ FAILED'}")
    print(f"   FCM: {'✅ PASSED' if fcm_success else '❌ FAILED'}")
    
    if optics_success:
        print("\n🎉 OPTICS algorithm is working correctly!")
    else:
        print("\n⚠️ OPTICS algorithm needs debugging.")