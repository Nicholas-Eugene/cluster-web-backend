#!/usr/bin/env python3
"""
Test map functionality with server
"""

import requests
import pandas as pd
import io
import json

def test_map_markers():
    """Test map with known coordinates"""
    
    # Create test data with cities that definitely have coordinates
    test_data = {
        'kabupaten_kota': [
            'Jakarta Pusat', 'Jakarta Selatan', 'Surabaya', 
            'Bandung', 'Medan', 'Semarang',
            'Makassar', 'Palembang', 'Yogyakarta',
            'Denpasar', 'Malang', 'Bogor'
        ],
        'tahun': [2023] * 12,
        'ipm': [85, 83, 78, 76, 72, 75, 70, 73, 80, 82, 77, 79],
        'garis_kemiskinan': [400000, 420000, 480000, 500000, 520000, 470000, 550000, 510000, 440000, 410000, 460000, 430000],
        'pengeluaran_per_kapita': [15000000, 14000000, 9000000, 10000000, 7000000, 8500000, 6500000, 7500000, 11000000, 13000000, 8800000, 12000000]
    }
    
    df = pd.DataFrame(test_data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    print("=== Testing Map Markers ===")
    print(f"Test data cities: {df['kabupaten_kota'].tolist()}")
    
    # Upload to server
    files = {'file': ('test_map_data.csv', csv_content, 'text/csv')}
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
            
            print(f"✅ Clustering successful! Session ID: {session_id}")
            
            # Check per-year results
            if results.get('clustering_type') == 'per_year':
                results_per_year = results.get('results_per_year', {})
                for year, year_results in results_per_year.items():
                    print(f"\n📊 Year {year} results:")
                    clusters = year_results.get('clusters', [])
                    
                    total_members = 0
                    for cluster in clusters:
                        cluster_size = cluster.get('size', 0)
                        total_members += cluster_size
                        print(f"  Cluster {cluster.get('id')}: {cluster_size} members")
                        
                        # Show first few members with their coordinates
                        members = cluster.get('members', [])[:3]
                        for member in members:
                            city = member.get('kabupaten_kota')
                            lat = member.get('latitude', 'N/A')
                            lng = member.get('longitude', 'N/A')
                            print(f"    - {city}: [{lat}, {lng}]")
                    
                    print(f"  Total members: {total_members}")
            
            # Test geographical data endpoint
            print(f"\n--- Testing Geographical Data Endpoint ---")
            geo_response = requests.get(f'http://localhost:8000/api/clustering/geography/{session_id}/')
            
            if geo_response.status_code == 200:
                geo_data = geo_response.json()
                geographical_data = geo_data.get('geographical_data', [])
                
                print(f"✅ Geographical endpoint successful!")
                print(f"Total geographical points: {len(geographical_data)}")
                
                # Show coordinate data
                for i, point in enumerate(geographical_data[:5]):
                    city = point.get('kabupaten_kota')
                    lat = point.get('latitude')
                    lng = point.get('longitude')
                    cluster_id = point.get('cluster_id')
                    print(f"  {i+1}. {city}: [{lat}, {lng}] - Cluster {cluster_id}")
                
                # Check how many have valid coordinates
                valid_coords = sum(1 for p in geographical_data if p.get('latitude') and p.get('longitude'))
                print(f"Points with valid coordinates: {valid_coords}/{len(geographical_data)}")
                
                return True
            else:
                print(f"❌ Geographical endpoint failed: {geo_response.status_code}")
                return False
                
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_map_markers()
    if success:
        print("\n🎉 Map test completed successfully!")
        print("The map should now display markers for all clustered cities.")
    else:
        print("\n⚠️ Map test failed. Check server logs for details.")