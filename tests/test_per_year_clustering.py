#!/usr/bin/env python3
"""
Test clustering per tahun dengan data Excel
"""

import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import sys
import os

def test_per_year_clustering():
    """Test clustering per tahun dengan data sample"""
    
    print("🧪 Testing Per-Year Clustering")
    print("=" * 50)
    
    # Load sample Excel data
    excel_file = 'sample_data_indonesia.xlsx'
    
    if not os.path.exists(excel_file):
        print(f"❌ File {excel_file} tidak ditemukan")
        return False
    
    try:
        # Read Excel file
        print(f"📖 Reading Excel file: {excel_file}")
        df = pd.read_excel(excel_file, engine='openpyxl')
        
        print(f"✅ Excel file loaded successfully")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Extract years from columns
        years = set()
        for col in df.columns:
            if '_' in col and any(metric in col for metric in ['ipm_', 'pengeluaran_', 'garis_kemiskinan_']):
                try:
                    year = int(col.split('_')[-1])
                    if 2015 <= year <= 2025:
                        years.add(year)
                except ValueError:
                    continue
        
        years = sorted(years)
        print(f"📅 Years found: {years}")
        
        # Process each year separately
        results_per_year = {}
        
        for year in years:
            print(f"\n🗓️ Processing year {year}...")
            
            # Extract data for this year
            year_data = []
            
            for _, row in df.iterrows():
                kabupaten_kota = row.get('kabupaten/kota', '')
                
                ipm_col = f'ipm_{year}'
                pengeluaran_col = f'pengeluaran_{year}'
                garis_kemiskinan_col = f'garis_kemiskinan_{year}'
                
                if (ipm_col in df.columns and pengeluaran_col in df.columns and 
                    garis_kemiskinan_col in df.columns):
                    
                    ipm_val = row.get(ipm_col)
                    pengeluaran_val = row.get(pengeluaran_col)
                    garis_kemiskinan_val = row.get(garis_kemiskinan_col)
                    
                    if (pd.notna(ipm_val) and pd.notna(pengeluaran_val) and 
                        pd.notna(garis_kemiskinan_val)):
                        
                        year_data.append({
                            'kabupaten_kota': kabupaten_kota,
                            'tahun': year,
                            'ipm': float(ipm_val),
                            'pengeluaran_per_kapita': float(pengeluaran_val),
                            'garis_kemiskinan': float(garis_kemiskinan_val)
                        })
            
            if not year_data:
                print(f"❌ No valid data for year {year}")
                continue
            
            year_df = pd.DataFrame(year_data)
            print(f"📊 Year {year} data shape: {year_df.shape}")
            
            # Perform OPTICS clustering for this year
            try:
                features = ['ipm', 'pengeluaran_per_kapita', 'garis_kemiskinan']
                X = year_df[features].values
                
                # Scale data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Run OPTICS
                optics = OPTICS(min_samples=2, xi=0.1, min_cluster_size=0.1)
                cluster_labels = optics.fit_predict(X_scaled)
                
                # Analyze results
                unique_labels = set(cluster_labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                
                print(f"✅ Year {year}: {n_clusters} clusters, {n_noise} noise points")
                
                # Store results
                results_per_year[year] = {
                    'clusters': n_clusters,
                    'noise': n_noise,
                    'total_regions': len(year_data),
                    'cluster_labels': cluster_labels,
                    'data': year_df
                }
                
                # Show cluster composition
                for label in sorted(unique_labels):
                    mask = cluster_labels == label
                    members = year_df[mask]['kabupaten_kota'].tolist()
                    if label == -1:
                        print(f"   Noise: {members}")
                    else:
                        print(f"   Cluster {label}: {members}")
                
            except Exception as e:
                print(f"❌ Error clustering year {year}: {e}")
                results_per_year[year] = {'error': str(e)}
        
        # Summary
        successful_years = [year for year, result in results_per_year.items() if 'error' not in result]
        failed_years = [year for year, result in results_per_year.items() if 'error' in result]
        
        print(f"\n" + "=" * 50)
        print(f"📊 Per-Year Clustering Summary:")
        print(f"   Total years processed: {len(years)}")
        print(f"   Successful: {len(successful_years)}")
        print(f"   Failed: {len(failed_years)}")
        print(f"   Success rate: {len(successful_years)/len(years)*100:.1f}%")
        
        if successful_years:
            avg_clusters = sum(results_per_year[year]['clusters'] for year in successful_years) / len(successful_years)
            avg_noise = sum(results_per_year[year]['noise'] for year in successful_years) / len(successful_years)
            print(f"   Average clusters per year: {avg_clusters:.1f}")
            print(f"   Average noise points per year: {avg_noise:.1f}")
        
        return len(successful_years) > 0
        
    except Exception as e:
        print(f"❌ Error in per-year clustering test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fcm_per_year():
    """Test FCM per tahun"""
    
    print(f"\n🌟 Testing FCM Per-Year Clustering")
    print("=" * 50)
    
    try:
        import skfuzzy as fuzz
        
        # Sample data for multiple years
        excel_file = 'sample_data_indonesia.xlsx'
        df = pd.read_excel(excel_file, engine='openpyxl')
        
        # Test FCM for year 2024
        year = 2024
        year_data = []
        
        for _, row in df.iterrows():
            kabupaten_kota = row.get('kabupaten/kota', '')
            
            ipm_val = row.get(f'ipm_{year}')
            pengeluaran_val = row.get(f'pengeluaran_{year}')
            garis_kemiskinan_val = row.get(f'garis_kemiskinan_{year}')
            
            if (pd.notna(ipm_val) and pd.notna(pengeluaran_val) and 
                pd.notna(garis_kemiskinan_val)):
                
                year_data.append([float(ipm_val), float(pengeluaran_val), float(garis_kemiskinan_val)])
        
        if len(year_data) < 3:
            print(f"❌ Insufficient data for FCM clustering")
            return False
        
        # Convert to numpy array and scale
        X = np.array(year_data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run FCM
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_scaled.T, 3, 2.0, error=0.005, maxiter=300
        )
        
        cluster_labels = np.argmax(u, axis=0)
        n_clusters = len(set(cluster_labels))
        
        print(f"✅ FCM Year {year}: {n_clusters} clusters")
        print(f"   Iterations: {p}")
        print(f"   Partition coefficient: {fpc:.4f}")
        
        return True
        
    except ImportError:
        print("⚠️ scikit-fuzzy not available for FCM test")
        return False
    except Exception as e:
        print(f"❌ FCM per-year error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Per-Year Clustering Tests")
    
    # Test OPTICS per year
    optics_success = test_per_year_clustering()
    
    # Test FCM per year
    fcm_success = test_fcm_per_year()
    
    print(f"\n" + "=" * 50)
    print("📋 Final Test Results:")
    print(f"   OPTICS Per-Year: {'✅ PASSED' if optics_success else '❌ FAILED'}")
    print(f"   FCM Per-Year: {'✅ PASSED' if fcm_success else '❌ FAILED'}")
    
    if optics_success:
        print(f"\n🎉 Per-year clustering is working correctly!")
        print(f"📝 Key features:")
        print(f"   ✅ Separate clustering for each year")
        print(f"   ✅ Excel file support")
        print(f"   ✅ Wide format data processing")
        print(f"   ✅ OPTICS algorithm working")
        print(f"   ✅ Noise detection")
    else:
        print(f"\n⚠️ Per-year clustering needs debugging.")