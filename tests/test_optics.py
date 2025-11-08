#!/usr/bin/env python3
"""
Test script untuk memastikan algoritma OPTICS berfungsi dengan benar
"""

import os
import sys
import django
import pandas as pd
import numpy as np

# Setup Django
sys.path.append('/workspace/backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from clustering.algorithms import get_clustering_results

def test_optics_algorithm():
    """Test OPTICS algorithm dengan data sample"""
    
    print("🧪 Testing OPTICS Algorithm...")
    
    # Create sample data in wide format
    sample_data = {
        'kabupaten/kota': ['Jakarta Pusat', 'Jakarta Utara', 'Surabaya', 'Bandung', 'Medan', 'Semarang'],
        'ipm_2020': [81.56, 81.05, 80.45, 79.67, 78.89, 78.12],
        'pengeluaran_2020': [9000000, 8400000, 7700000, 7000000, 6400000, 6100000],
        'garis_kemiskinan_2020': [620000, 620000, 500000, 460000, 420000, 420000],
        'ipm_2021': [82.12, 81.58, 81.01, 80.12, 79.45, 78.56],
        'pengeluaran_2021': [9300000, 8700000, 8000000, 7300000, 6700000, 6400000],
        'garis_kemiskinan_2021': [640000, 640000, 520000, 480000, 440000, 440000]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"📊 Sample data shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    try:
        # Test OPTICS clustering
        print("\n🔍 Running OPTICS clustering...")
        results = get_clustering_results(
            df, 
            algorithm='optics',
            features=['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita'],
            min_samples=2,  # Lower for small dataset
            xi=0.1,
            min_cluster_size=0.1,
            selected_year='2021'
        )
        
        print("✅ OPTICS clustering completed successfully!")
        print(f"📈 Algorithm: {results['algorithm']}")
        print(f"🎯 Total regions: {results['summary']['total_regions']}")
        print(f"🔢 Number of clusters: {results['summary']['num_clusters']}")
        print(f"🔇 Noise points: {results['summary']['noise_points']}")
        print(f"⏱️ Execution time: {results['summary']['execution_time']:.4f}s")
        
        # Print evaluation metrics
        print(f"\n📊 Evaluation Metrics:")
        print(f"   Davies-Bouldin Index: {results['evaluation']['davies_bouldin']}")
        print(f"   Silhouette Score: {results['evaluation']['silhouette_score']}")
        
        # Print cluster details
        print(f"\n🏷️ Cluster Details:")
        for i, cluster in enumerate(results['clusters']):
            cluster_id = cluster['id']
            size = cluster['size']
            print(f"   Cluster {cluster_id}: {size} members")
            
            if cluster['centroid']:
                centroid = cluster['centroid']
                print(f"      Centroid - IPM: {centroid['ipm']:.2f}, "
                      f"Garis Kemiskinan: {centroid['garis_kemiskinan']:,.0f}, "
                      f"Pengeluaran: {centroid['pengeluaran_per_kapita']:,.0f}")
            
            # Show first few members
            for j, member in enumerate(cluster['members'][:3]):
                print(f"      - {member['kabupaten_kota']} ({member['tahun']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during OPTICS clustering: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_fcm_algorithm():
    """Test FCM algorithm untuk perbandingan"""
    
    print("\n🧪 Testing FCM Algorithm for comparison...")
    
    # Create sample data
    sample_data = {
        'kabupaten/kota': ['Jakarta Pusat', 'Jakarta Utara', 'Surabaya', 'Bandung', 'Medan', 'Semarang'],
        'ipm_2021': [82.12, 81.58, 81.01, 80.12, 79.45, 78.56],
        'pengeluaran_2021': [9300000, 8700000, 8000000, 7300000, 6700000, 6400000],
        'garis_kemiskinan_2021': [640000, 640000, 520000, 480000, 440000, 440000]
    }
    
    df = pd.DataFrame(sample_data)
    
    try:
        # Test FCM clustering
        print("🌟 Running FCM clustering...")
        results = get_clustering_results(
            df, 
            algorithm='fcm',
            features=['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita'],
            n_clusters=3,
            m=2.0,
            selected_year='2021'
        )
        
        print("✅ FCM clustering completed successfully!")
        print(f"📈 Algorithm: {results['algorithm']}")
        print(f"🎯 Total regions: {results['summary']['total_regions']}")
        print(f"🔢 Number of clusters: {results['summary']['num_clusters']}")
        print(f"⏱️ Execution time: {results['summary']['execution_time']:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during FCM clustering: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Clustering Algorithm Tests")
    print("=" * 50)
    
    # Test both algorithms
    optics_success = test_optics_algorithm()
    fcm_success = test_fcm_algorithm()
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"   OPTICS: {'✅ PASSED' if optics_success else '❌ FAILED'}")
    print(f"   FCM: {'✅ PASSED' if fcm_success else '❌ FAILED'}")
    
    if optics_success and fcm_success:
        print("\n🎉 All tests passed! Both algorithms are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")