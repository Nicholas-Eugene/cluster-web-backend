#!/usr/bin/env python3
"""
Script untuk membuat file Excel sample dengan format yang benar
"""

import pandas as pd

def create_sample_excel():
    """Buat file Excel sample dengan format yang diminta"""
    
    # Data sample sesuai format yang diminta
    data = {
        'kabupaten/kota': [
            'Jakarta Pusat', 'Jakarta Utara', 'Jakarta Selatan', 'Jakarta Barat', 'Jakarta Timur',
            'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Palembang'
        ],
        # IPM data per tahun
        'ipm_2016': [79.32, 78.91, 81.65, 79.88, 78.23, 77.45, 76.12, 75.89, 74.56, 73.78],
        'ipm_2017': [79.78, 79.45, 82.23, 80.34, 78.78, 77.89, 76.67, 76.34, 75.12, 74.23],
        'ipm_2018': [80.45, 79.98, 82.89, 80.78, 79.34, 78.34, 77.23, 76.78, 75.67, 74.78],
        'ipm_2019': [81.12, 80.52, 83.56, 81.23, 79.89, 78.78, 77.78, 77.23, 76.23, 75.34],
        'ipm_2020': [81.56, 81.05, 84.23, 81.67, 80.45, 79.23, 78.34, 77.67, 76.78, 75.89],
        'ipm_2021': [82.12, 81.58, 84.89, 82.12, 81.01, 79.67, 78.89, 78.12, 77.34, 76.45],
        'ipm_2022': [82.67, 82.12, 85.56, 82.56, 81.56, 80.12, 79.45, 78.56, 77.89, 77.01],
        'ipm_2023': [83.23, 82.65, 86.23, 83.01, 82.12, 80.56, 80.01, 79.01, 78.45, 77.56],
        'ipm_2024': [83.78, 83.19, 86.89, 83.45, 82.67, 81.01, 80.56, 79.45, 79.01, 78.12],
        
        # Pengeluaran per kapita per tahun
        'pengeluaran_2016': [7800000, 7200000, 9200000, 6800000, 6500000, 5800000, 5200000, 4900000, 4600000, 4300000],
        'pengeluaran_2017': [8100000, 7500000, 9500000, 7100000, 6800000, 6100000, 5500000, 5200000, 4900000, 4600000],
        'pengeluaran_2018': [8400000, 7800000, 9800000, 7400000, 7100000, 6400000, 5800000, 5500000, 5200000, 4900000],
        'pengeluaran_2019': [8700000, 8100000, 10100000, 7700000, 7400000, 6700000, 6100000, 5800000, 5500000, 5200000],
        'pengeluaran_2020': [9000000, 8400000, 10400000, 8000000, 7700000, 7000000, 6400000, 6100000, 5800000, 5500000],
        'pengeluaran_2021': [9300000, 8700000, 10700000, 8300000, 8000000, 7300000, 6700000, 6400000, 6100000, 5800000],
        'pengeluaran_2022': [9600000, 9000000, 11000000, 8600000, 8300000, 7600000, 7000000, 6700000, 6400000, 6100000],
        'pengeluaran_2023': [9900000, 9300000, 11300000, 8900000, 8600000, 7900000, 7300000, 7000000, 6700000, 6400000],
        'pengeluaran_2024': [10200000, 9600000, 11600000, 9200000, 8900000, 8200000, 7600000, 7300000, 7000000, 6700000],
        
        # Garis kemiskinan per tahun
        'garis_kemiskinan_2016': [540000, 540000, 540000, 540000, 420000, 380000, 350000, 340000, 320000, 310000],
        'garis_kemiskinan_2017': [560000, 560000, 560000, 560000, 440000, 400000, 370000, 360000, 340000, 330000],
        'garis_kemiskinan_2018': [580000, 580000, 580000, 580000, 460000, 420000, 390000, 380000, 360000, 350000],
        'garis_kemiskinan_2019': [600000, 600000, 600000, 600000, 480000, 440000, 410000, 400000, 380000, 370000],
        'garis_kemiskinan_2020': [620000, 620000, 620000, 620000, 500000, 460000, 430000, 420000, 400000, 390000],
        'garis_kemiskinan_2021': [640000, 640000, 640000, 640000, 520000, 480000, 450000, 440000, 420000, 410000],
        'garis_kemiskinan_2022': [660000, 660000, 660000, 660000, 540000, 500000, 470000, 460000, 440000, 430000],
        'garis_kemiskinan_2023': [680000, 680000, 680000, 680000, 560000, 520000, 490000, 480000, 460000, 450000],
        'garis_kemiskinan_2024': [700000, 700000, 700000, 700000, 580000, 540000, 510000, 500000, 480000, 470000]
    }
    
    # Buat DataFrame
    df = pd.DataFrame(data)
    
    # Simpan ke file Excel
    output_file = 'sample_data_indonesia.xlsx'
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"✅ File Excel sample berhasil dibuat: {output_file}")
    print(f"📊 Data shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"📅 Years: 2016-2024")
    print(f"🏙️ Cities: {len(df)} kabupaten/kota")
    
    # Tampilkan preview
    print("\n📖 Preview data:")
    print(df.head(3).to_string())
    
    return output_file

if __name__ == "__main__":
    try:
        create_sample_excel()
        print("\n🎉 Sample Excel file created successfully!")
    except Exception as e:
        print(f"❌ Error creating Excel file: {e}")
        print("Make sure you have pandas and openpyxl installed:")
        print("pip install pandas openpyxl")