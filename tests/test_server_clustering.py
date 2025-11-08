#!/usr/bin/env python3
"""
Test server clustering end-to-end
"""

import requests
import pandas as pd
import io
import json
import time

def test_server_clustering():
    """Test clustering via server API"""
    
    # Create test data
    test_data = {
        'kabupaten_kota': ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Makassar', 'Palembang', 'Denpasar', 'Yogyakarta', 'Malang'],
        'tahun': [2023] * 10,
        'ipm': [85.0, 78.0, 76.0, 72.0, 75.0, 70.0, 73.0, 82.0, 80.0, 77.0],
        'garis_kemiskinan': [400000, 450000, 480000, 520000, 470000, 550000, 500000, 420000, 440000, 460000],
        'pengeluaran_per_kapita': [12000000, 8000000, 9000000, 7000000, 8500000, 6500000, 7500000, 11000000, 10000000, 8800000]
    }
    
    df = pd.DataFrame(test_data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    print("=== Testing Server Clustering ===")
    print(f"Test data shape: {df.shape}")
    print("Data preview:")
    print(df.head())
    
    # Test FCM clustering
    print("\n--- Testing FCM Clustering ---")
    
    files = {'file': ('test_data.csv', csv_content, 'text/csv')}
    data = {
        'algorithm': 'fcm',
        'num_clusters': 3,
        'fuzzy_coeff': 2.0,
        'max_iter': 100,
        'tolerance': 0.001
    }
    
    try:
        response = requests.post('http://localhost:8000/api/clustering/upload/', files=files, data=data, timeout=30)
        
        if response.status_code == 201:
            result = response.json()
            session_id = result.get('session_id')
            results = result.get('results')
            
            print(f"✅ FCM Clustering successful!")
            print(f"Session ID: {session_id}")
            print(f"Full response keys: {list(result.keys())}")
            if results:
                print(f"Results keys: {list(results.keys())}")
                
                # Check if it's per-year clustering
                if results.get('clustering_type') == 'per_year':
                    print("📊 Per-year clustering detected")
                    overall_summary = results.get('overall_summary', {})
                    print(f"Algorithm: {overall_summary.get('algorithm')}")
                    print(f"Years processed: {overall_summary.get('years_processed')}")
                    
                    results_per_year = results.get('results_per_year', {})
                    print(f"Results per year:")
                    
                    all_clusters_empty = True
                    for year, year_results in results_per_year.items():
                        print(f"  Year {year}:")
                        clusters = year_results.get('clusters', [])
                        if clusters:
                            all_clusters_empty = False
                            for cluster in clusters:
                                print(f"    Cluster {cluster.get('id')}: {cluster.get('size')} members")
                        else:
                            print(f"    No clusters found")
                    
                    if all_clusters_empty:
                        print("❌ PROBLEM: No clusters found in any year!")
                        return False
                    else:
                        print("✅ Data distributed across clusters")
                        return True
                        
                else:
                    # Single year clustering
                    print(f"Algorithm: {results.get('algorithm')}")
                    print(f"Total regions: {results.get('summary', {}).get('total_regions')}")
                    print(f"Number of clusters: {results.get('summary', {}).get('num_clusters')}")
                    
                    clusters = results.get('clusters', [])
                    print(f"Cluster distribution:")
                    for cluster in clusters:
                        print(f"  Cluster {cluster.get('id')}: {cluster.get('size')} members")
                        
            else:
                print("❌ No results in response!")
                print(f"Full response: {json.dumps(result, indent=2)}")
                return False
                
        else:
            print(f"❌ FCM Clustering failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure Django server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error during FCM test: {e}")
        return False

def test_optics_clustering():
    """Test OPTICS clustering"""
    
    # Create test data with clear clusters
    test_data = {
        'kabupaten_kota': ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Makassar', 'Palembang', 'Denpasar', 'Yogyakarta', 'Malang'],
        'tahun': [2023] * 10,
        'ipm': [85.0, 78.0, 76.0, 72.0, 75.0, 70.0, 73.0, 82.0, 80.0, 77.0],
        'garis_kemiskinan': [400000, 450000, 480000, 520000, 470000, 550000, 500000, 420000, 440000, 460000],
        'pengeluaran_per_kapita': [12000000, 8000000, 9000000, 7000000, 8500000, 6500000, 7500000, 11000000, 10000000, 8800000]
    }
    
    df = pd.DataFrame(test_data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    print("\n--- Testing OPTICS Clustering ---")
    
    files = {'file': ('test_data.csv', csv_content, 'text/csv')}
    data = {
        'algorithm': 'optics',
        'min_samples': 3,
        'xi': 0.05,
        'min_cluster_size': 0.1
    }
    
    try:
        response = requests.post('http://localhost:8000/api/clustering/upload/', files=files, data=data, timeout=30)
        
        if response.status_code == 201:
            result = response.json()
            results = result.get('results')
            
            print(f"✅ OPTICS Clustering successful!")
            print(f"Algorithm: {results.get('algorithm')}")
            print(f"Total regions: {results.get('summary', {}).get('total_regions')}")
            print(f"Number of clusters: {results.get('summary', {}).get('num_clusters')}")
            print(f"Noise points: {results.get('summary', {}).get('noise_points', 0)}")
            
            clusters = results.get('clusters', [])
            print(f"Cluster distribution:")
            for cluster in clusters:
                cluster_id = cluster.get('id')
                if cluster_id == 'noise':
                    print(f"  Noise: {cluster.get('size')} members")
                else:
                    print(f"  Cluster {cluster_id}: {cluster.get('size')} members")
                    
            return True
                
        else:
            print(f"❌ OPTICS Clustering failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during OPTICS test: {e}")
        return False

if __name__ == "__main__":
    print("Testing server clustering...")
    
    # Test FCM
    fcm_success = test_server_clustering()
    
    # Test OPTICS
    optics_success = test_optics_clustering()
    
    print(f"\n=== Test Summary ===")
    print(f"FCM test: {'PASSED' if fcm_success else 'FAILED'}")
    print(f"OPTICS test: {'PASSED' if optics_success else 'FAILED'}")
    
    if fcm_success and optics_success:
        print("🎉 All tests passed! Clustering is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for more details.")