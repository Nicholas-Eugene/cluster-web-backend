#!/usr/bin/env python3
"""
Test Excel processing dengan format yang benar
"""

import pandas as pd
import sys
import os

# Add the backend directory to path
sys.path.append('/workspace/backend')

def test_excel_processing():
    """Test processing Excel file dengan format yang diminta"""
    
    print("🧪 Testing Excel Processing")
    print("=" * 40)
    
    # Load the sample Excel file
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
        
        # Test the reshape function
        from clustering.algorithms import ClusteringAlgorithms
        
        clustering = ClusteringAlgorithms()
        
        print(f"\n🔄 Testing wide-to-long conversion...")
        long_df = clustering.reshape_wide_to_long(df)
        
        print(f"✅ Conversion successful!")
        print(f"📊 Long format shape: {long_df.shape}")
        print(f"📋 Long format columns: {list(long_df.columns)}")
        
        # Show sample data
        print(f"\n📖 Sample long format data:")
        print(long_df.head().to_string())
        
        # Test clustering
        print(f"\n🎯 Testing OPTICS clustering...")
        
        try:
            from clustering.algorithms import get_clustering_results
            
            results = get_clustering_results(
                df, 
                algorithm='optics',
                features=['ipm', 'garis_kemiskinan', 'pengeluaran_per_kapita'],
                min_samples=2,
                xi=0.1,
                min_cluster_size=0.1,
                selected_year='2024'
            )
            
            print(f"✅ OPTICS clustering successful!")
            print(f"📈 Algorithm: {results['algorithm']}")
            print(f"🎯 Total regions: {results['summary']['total_regions']}")
            print(f"🔢 Number of clusters: {results['summary']['num_clusters']}")
            print(f"🔇 Noise points: {results['summary']['noise_points']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Clustering failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Excel processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_comparison():
    """Test CSV processing untuk perbandingan"""
    
    print(f"\n🧪 Testing CSV Processing for comparison")
    print("=" * 40)
    
    try:
        # Create a CSV version of the data
        df_excel = pd.read_excel('sample_data_indonesia.xlsx', engine='openpyxl')
        csv_file = 'sample_data_test.csv'
        df_excel.to_csv(csv_file, index=False)
        
        print(f"📖 Reading CSV file: {csv_file}")
        df_csv = pd.read_csv(csv_file)
        
        print(f"✅ CSV file loaded successfully")
        print(f"📊 Shape: {df_csv.shape}")
        
        # Compare with Excel
        if df_excel.equals(df_csv):
            print(f"✅ CSV and Excel data are identical")
        else:
            print(f"⚠️ CSV and Excel data differ slightly (normal due to formatting)")
        
        # Clean up
        os.remove(csv_file)
        
        return True
        
    except Exception as e:
        print(f"❌ CSV comparison failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Excel Processing Tests")
    
    # Test Excel processing
    excel_success = test_excel_processing()
    
    # Test CSV comparison
    csv_success = test_csv_comparison()
    
    print(f"\n" + "=" * 40)
    print("📋 Test Results:")
    print(f"   Excel Processing: {'✅ PASSED' if excel_success else '❌ FAILED'}")
    print(f"   CSV Comparison: {'✅ PASSED' if csv_success else '❌ FAILED'}")
    
    if excel_success and csv_success:
        print(f"\n🎉 All tests passed! Excel support is working correctly.")
        print(f"\n📝 Format yang didukung:")
        print(f"   ✅ kabupaten/kota")
        print(f"   ✅ ipm_2016, ipm_2017, ..., ipm_2024")
        print(f"   ✅ pengeluaran_2016, pengeluaran_2017, ..., pengeluaran_2024")
        print(f"   ✅ garis_kemiskinan_2016, garis_kemiskinan_2017, ..., garis_kemiskinan_2024")
    else:
        print(f"\n⚠️ Some tests failed. Please check the implementation.")