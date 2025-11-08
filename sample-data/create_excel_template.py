#!/usr/bin/env python3
"""
Script untuk membuat template Excel yang bisa didownload
"""

import pandas as pd

def create_excel_template():
    """Buat template Excel kosong dengan struktur yang benar"""
    
    # Create template with proper structure
    template_data = {
        'kabupaten/kota': [
            'Jakarta Pusat', 'Jakarta Utara', 'Jakarta Selatan', 'Jakarta Barat', 'Jakarta Timur',
            'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Palembang',
            'Makassar', 'Denpasar', 'Yogyakarta', 'Malang', 'Bogor'
        ]
    }
    
    # Add IPM columns for each year
    for year in range(2016, 2025):
        template_data[f'ipm_{year}'] = [None] * 15
    
    # Add Pengeluaran columns for each year
    for year in range(2016, 2025):
        template_data[f'pengeluaran_{year}'] = [None] * 15
    
    # Add Garis Kemiskinan columns for each year
    for year in range(2016, 2025):
        template_data[f'garis_kemiskinan_{year}'] = [None] * 15
    
    # Create DataFrame
    df = pd.DataFrame(template_data)
    
    # Save to Excel with example data for first few cities
    example_data = {
        'kabupaten/kota': [
            'Jakarta Pusat', 'Jakarta Utara', 'Jakarta Selatan', 'Jakarta Barat', 'Jakarta Timur',
            'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Palembang',
            'Makassar', 'Denpasar', 'Yogyakarta', 'Malang', 'Bogor'
        ]
    }
    
    # Add example data for 3 cities, rest will be None
    num_cities = 15
    example_ipm = [79.32, 78.91, 81.65] + [None] * 12
    example_pengeluaran = [7800000, 7200000, 9200000] + [None] * 12
    example_garis_kemiskinan = [540000, 540000, 540000] + [None] * 12
    
    for year in range(2016, 2025):
        example_data[f'ipm_{year}'] = example_ipm.copy()
        example_data[f'pengeluaran_{year}'] = example_pengeluaran.copy()
        example_data[f'garis_kemiskinan_{year}'] = example_garis_kemiskinan.copy()
    
    # Create example DataFrame
    example_df = pd.DataFrame(example_data)
    
    # Save both template and example
    template_file = 'template_dataset_indonesia.xlsx'
    example_file = 'example_dataset_indonesia.xlsx'
    
    # Template (empty)
    df.to_excel(template_file, index=False, engine='openpyxl')
    
    # Example (with sample data)
    example_df.to_excel(example_file, index=False, engine='openpyxl')
    
    print(f"✅ Template Excel files created:")
    print(f"   📝 {template_file} - Template kosong")
    print(f"   📊 {example_file} - Contoh dengan data sample")
    print(f"📊 Structure: {df.shape}")
    print(f"📋 Total columns: {len(df.columns)}")
    print(f"📅 Years: 2016-2024 (9 tahun)")
    
    # Show column structure
    print(f"\n📋 Column structure:")
    print(f"   1. kabupaten/kota")
    print(f"   2-10. ipm_2016 to ipm_2024")
    print(f"   11-19. pengeluaran_2016 to pengeluaran_2024")
    print(f"   20-28. garis_kemiskinan_2016 to garis_kemiskinan_2024")
    
    return template_file, example_file

if __name__ == "__main__":
    try:
        template_file, example_file = create_excel_template()
        print(f"\n🎉 Excel templates created successfully!")
        print(f"\n📥 Files ready for download:")
        print(f"   - {template_file}")
        print(f"   - {example_file}")
    except Exception as e:
        print(f"❌ Error creating Excel templates: {e}")
        print("Make sure you have pandas and openpyxl installed")