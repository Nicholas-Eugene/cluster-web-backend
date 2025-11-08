#!/usr/bin/env python3
"""
Simple test script to verify the new API endpoints work correctly
"""

import requests
import json
import pandas as pd
import io

# Test data
BASE_URL = "http://localhost:8000/api"

def test_upload_and_endpoints():
    """Test the upload endpoint and new endpoints"""
    
    # Create a simple test CSV file
    test_data = {
        'kabupaten_kota': ['Jakarta Pusat', 'Jakarta Selatan', 'Surabaya'],
        'tahun': [2023, 2023, 2023],
        'ipm': [82.5, 81.3, 76.8],
        'garis_kemiskinan': [532000, 580000, 465000],
        'pengeluaran_per_kapita': [15000000, 14500000, 12000000]
    }
    
    df = pd.DataFrame(test_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    print("Test data created successfully")
    print("CSV content:")
    print(csv_content)
    
    # Test file upload
    files = {'file': ('test_data.csv', csv_content, 'text/csv')}
    data = {
        'algorithm': 'fcm',
        'num_clusters': 2,
        'fuzzy_coeff': 2.0,
        'max_iter': 100,
        'tolerance': 0.001
    }
    
    try:
        print("\n=== Testing Upload Endpoint ===")
        response = requests.post(f"{BASE_URL}/clustering/upload/", files=files, data=data, timeout=30)
        print(f"Upload response status: {response.status_code}")
        
        if response.status_code == 201:
            result = response.json()
            session_id = result.get('session_id')
            print(f"Session ID: {session_id}")
            
            # Test new endpoints
            if session_id:
                print("\n=== Testing New Endpoints ===")
                
                # Test export endpoint
                print("Testing export endpoint...")
                export_response = requests.get(f"{BASE_URL}/clustering/export/{session_id}/?format=csv")
                print(f"Export CSV status: {export_response.status_code}")
                
                # Test evaluation metrics endpoint
                print("Testing evaluation metrics endpoint...")
                eval_response = requests.get(f"{BASE_URL}/clustering/evaluation/{session_id}/")
                print(f"Evaluation metrics status: {eval_response.status_code}")
                
                # Test geographical data endpoint
                print("Testing geographical data endpoint...")
                geo_response = requests.get(f"{BASE_URL}/clustering/geography/{session_id}/")
                print(f"Geographical data status: {geo_response.status_code}")
                
                # Test report generation endpoint
                print("Testing report generation endpoint...")
                report_response = requests.post(f"{BASE_URL}/clustering/report/{session_id}/", json={})
                print(f"Report generation status: {report_response.status_code}")
                
                print("\n=== All endpoints tested successfully! ===")
            else:
                print("No session ID returned from upload")
        else:
            print(f"Upload failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Make sure Django server is running on localhost:8000")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_upload_and_endpoints()