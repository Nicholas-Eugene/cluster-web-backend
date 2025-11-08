#!/usr/bin/env python3
"""
Test script to check if city names in sample data match with coordinate database
"""

import pandas as pd
import json


def test_coordinate_matching():
    """Test coordinate matching with sample data"""

    # Load sample data
    try:
        df = pd.read_csv("sample_data_indonesia.csv")
        print(f"=== Coordinate Matching Test ===")
        print(f"Sample data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Get unique city names
        unique_cities = df["kabupaten_kota"].unique()
        print(f"\nUnique cities in dataset: {len(unique_cities)}")

        # Show first 20 city names
        print("\nFirst 20 city names:")
        for i, city in enumerate(unique_cities[:20]):
            print(f"{i+1:2d}. {city}")

        # JavaScript coordinate database (subset for testing)
        js_coords = {
            # Aceh
            "Aceh Barat": [4.4531, 96.1265],
            "Aceh Barat Daya": [3.8335, 96.8417],
            "Aceh Besar": [5.3686, 95.4678],
            "Aceh Jaya": [4.8028, 95.7369],
            "Aceh Selatan": [3.2081, 97.2608],
            "Aceh Singkil": [2.4988, 97.8398],
            "Aceh Tamiang": [4.2694, 98.0194],
            "Aceh Tengah": [4.5165, 96.8417],
            "Aceh Tenggara": [3.3769, 97.7124],
            "Aceh Timur": [4.6401, 97.6256],
            "Aceh Utara": [4.9961, 97.2144],
            "Bener Meriah": [4.7479, 96.8601],
            "Bireuen": [5.0872, 96.6559],
            "Gayo Lues": [3.9789, 97.3592],
            "Nagan Raya": [4.1557, 96.5369],
            "Pidie": [5.0244, 96.0968],
            "Pidie Jaya": [5.1278, 96.2238],
            "Simeulue": [2.6052, 96.0791],
            "Banda Aceh": [5.5501, 95.3187],
            "Langsa": [4.4735, 97.9702],
            "Lhokseumawe": [5.1764, 97.1436],
            "Sabang": [5.8824, 95.3245],
            "Subulussalam": [2.7667, 97.9167],
            # Sumatera Utara
            "Asahan": [2.7699, 99.5303],
            "Batu Bara": [3.2000, 99.4167],
            "Dairi": [2.8167, 98.2500],
            "Deli Serdang": [3.5333, 98.7167],
            "Humbang Hasundutan": [2.2612, 98.5137],
            "Karo": [3.1000, 98.3000],
            "Labuhanbatu": [2.1667, 100.1667],
            "Labuhanbatu Selatan": [2.0333, 100.1333],
            "Labuhanbatu Utara": [2.5167, 99.7500],
            "Langkat": [3.7333, 98.2167],
            "Mandailing Natal": [0.7833, 99.2500],
            "Nias": [1.0833, 97.7500],
            "Nias Barat": [1.0000, 97.4167],
            "Nias Selatan": [0.5500, 97.7833],
            "Nias Utara": [1.3333, 97.3333],
            "Padang Lawas": [1.4167, 99.8333],
            "Padang Lawas Utara": [1.6667, 99.6667],
            "Pakpak Bharat": [2.5667, 98.3000],
            "Samosir": [2.6167, 98.7167],
            "Serdang Bedagai": [3.3333, 99.0000],
            "Simalungun": [2.8667, 99.0000],
            "Tapanuli Selatan": [1.5833, 99.2500],
            "Tapanuli Tengah": [1.9000, 98.6667],
            "Tapanuli Utara": [2.0000, 99.0667],
            "Toba": [2.3667, 99.2167],
            "Binjai": [3.6000, 98.4833],
            "Gunungsitoli": [1.2833, 97.6167],
            "Medan": [3.5833, 98.6667],
            "Padangsidimpuan": [1.3833, 99.2667],
            "Pematangsiantar": [2.9667, 99.0667],
            "Sibolga": [1.7333, 98.7833],
            "Tanjungbalai": [2.9667, 99.8000],
            "Tebing Tinggi": [3.3333, 99.1667],
            # Sumatera Barat
            "Agam": [-0.3167, 100.0000],
            "Dharmasraya": [-1.0667, 101.3667],
            "Kepulauan Mentawai": [-2.0833, 99.5000],
            "Lima Puluh Kota": [0.1333, 100.5833],
            "Padang Pariaman": [-0.6000, 100.2500],
            "Pasaman": [0.7833, 100.0833],
            "Pasaman Barat": [0.3333, 99.5833],
            "Pesisir Selatan": [-1.6667, 100.8333],
            "Sijunjung": [-0.7167, 101.0000],
            "Solok": [-0.9500, 100.8333],
            "Solok Selatan": [-1.3167, 101.3500],
            "Tanah Datar": [-0.4167, 100.5667],
            "Bukittinggi": [-0.3000, 100.3667],
            "Padang": [-0.9500, 100.3500],
            "Padangpanjang": [-0.4667, 100.4167],
            "Pariaman": [-0.6167, 100.1167],
            "Payakumbuh": [-0.2167, 100.6333],
            "Sawahlunto": [-0.6833, 100.7833],
            "Solok": [-0.7833, 100.6500],
            # Riau
            "Bengkalis": [1.4833, 101.6667],
            "Indragiri Hilir": [-0.3333, 103.1667],
            "Indragiri Hulu": [-0.5333, 102.3333],
            "Kampar": [0.3333, 101.0000],
            "Kepulauan Meranti": [0.9667, 102.6667],
            "Kuantan Singingi": [-0.5167, 101.5000],
            "Pelalawan": [0.3500, 102.1667],
            "Rokan Hilir": [2.1667, 100.8333],
            "Rokan Hulu": [0.9167, 100.5000],
            "Siak": [0.8333, 101.6667],
            "Dumai": [1.6667, 101.4500],
            "Pekanbaru": [0.5333, 101.4500],
            # Jambi
            "Batanghari": [-1.7167, 103.1000],
            "Bungo": [-1.5167, 101.9667],
            "Kerinci": [-2.0833, 101.4833],
            "Merangin": [-2.1167, 102.1333],
            "Muaro Jambi": [-1.5833, 103.5833],
            "Sarolangun": [-2.2833, 102.6333],
            "Tanjung Jabung Barat": [-1.0000, 103.3333],
            "Tanjung Jabung Timur": [-1.1333, 103.9167],
            "Tebo": [-1.4167, 102.3167],
            "Jambi": [-1.6000, 103.6167],
            "Sungai Penuh": [-2.0667, 101.3833],
            # Sumatera Selatan
            "Banyuasin": [-2.8833, 104.3833],
            "Empat Lawang": [-3.8333, 103.0833],
            "Lahat": [-3.8000, 103.5333],
            "Muara Enim": [-3.6500, 103.7833],
            "Musi Banyuasin": [-2.5000, 103.7500],
            "Musi Rawas": [-3.1667, 103.1667],
            "Musi Rawas Utara": [-2.6000, 103.3000],
            "Ogan Ilir": [-3.2500, 104.6500],
            "Ogan Komering Ilir": [-3.3833, 105.3667],
            "Ogan Komering Ulu": [-4.1333, 104.0333],
            "Ogan Komering Ulu Selatan": [-4.6333, 104.1333],
            "Ogan Komering Ulu Timur": [-3.8667, 104.7500],
            "Penukal Abab Lematang Ilir": [-3.2513, 104.0906],
            "Lubuklinggau": [-3.3000, 102.8667],
            "Pagar Alam": [-4.1333, 103.2500],
            "Palembang": [-2.9833, 104.7667],
            "Prabumulih": [-3.4333, 104.2333],
            # Bengkulu
            "Bengkulu Selatan": [-4.3333, 103.0000],
            "Bengkulu Tengah": [-3.5333, 102.4333],
            "Bengkulu Utara": [-3.2000, 102.2167],
            "Kaur": [-4.7833, 103.4167],
            "Kepahiang": [-3.6167, 102.5833],
            "Lebong": [-3.1500, 102.3333],
            "Mukomuko": [-2.5500, 101.5000],
            "Rejang Lebong": [-3.4667, 102.5333],
            "Seluma": [-4.0000, 102.5000],
            "Bengkulu": [-3.8000, 102.2667],
            # Lampung
            "Lampung Barat": [-5.0833, 104.1667],
            "Lampung Selatan": [-5.5667, 105.5000],
            "Lampung Tengah": [-4.8667, 105.2667],
            "Lampung Timur": [-5.1167, 105.7333],
            "Lampung Utara": [-4.8333, 104.7500],
            "Mesuji": [-4.0500, 105.4167],
            "Pesawaran": [-5.4500, 105.1833],
            "Pesisir Barat": [-5.1833, 103.9500],
            "Pringsewu": [-5.3500, 104.9667],
            "Tanggamus": [-5.4167, 104.6667],
            "Tulang Bawang": [-4.4167, 105.5833],
            "Tulang Bawang Barat": [-4.4000, 105.0833],
            "Way Kanan": [-4.5000, 104.5333],
            "Bandar Lampung": [-5.4333, 105.2667],
            "Metro": [-5.1167, 105.3000],
            # Kepulauan Bangka Belitung
            "Bangka": [-1.9167, 105.9000],
            "Bangka Barat": [-2.0833, 105.4500],
            "Bangka Selatan": [-2.7500, 106.3333],
            "Bangka Tengah": [-2.3333, 106.1667],
            "Belitung": [-2.8333, 107.6667],
            "Belitung Timur": [-2.8333, 108.1667],
            "Pangkal Pinang": [-2.1333, 106.1167],
            # Kepulauan Riau
            "Bintan": [1.1333, 104.4500],
            "Karimun": [0.8333, 103.4167],
            "Kepulauan Anambas": [3.2000, 106.0000],
            "Lingga": [-0.2000, 104.5833],
            "Natuna": [4.0000, 108.2500],
            "Batam": [1.0500, 104.0000],
            "Tanjung Pinang": [0.9167, 104.4500],
            # DKI Jakarta
            "Administrasi Kepulauan Seribu": [-5.6167, 106.5833],
            "Administrasi Jakarta Barat": [-6.1667, 106.7833],
            "Administrasi Jakarta Pusat": [-6.1833, 106.8333],
            "Administrasi Jakarta Selatan": [-6.2667, 106.8167],
            "Administrasi Jakarta Timur": [-6.2333, 106.8833],
            "Administrasi Jakarta Utara": [-6.1333, 106.8500],
            # Jawa Barat
            "Bandung": [-7.0177, 107.6167],
            "Bandung Barat": [-6.8667, 107.4167],
            "Bekasi": [-6.2383, 107.1347],
            "Bogor": [-6.5950, 106.7892],
            "Ciamis": [-7.3264, 108.3537],
            "Cianjur": [-6.8193, 107.1397],
            "Cirebon": [-6.7063, 108.5617],
            "Garut": [-7.2167, 107.9066],
            "Indramayu": [-6.3253, 108.3237],
            "Karawang": [-6.3050, 107.3496],
            "Kuningan": [-6.9789, 108.4855],
            "Majalengka": [-6.8394, 108.2254],
            "Pangandaran": [-7.6907, 108.6543],
            "Purwakarta": [-6.5589, 107.4429],
            "Subang": [-6.5765, 107.7618],
            "Sukabumi": [-6.9238, 106.9280],
            "Sumedang": [-6.8374, 107.9168],
            "Tasikmalaya": [-7.3949, 108.2087],
            "Bandung": [-6.9175, 107.6191],
            "Banjar": [-7.3706, 108.5346],
            "Bekasi": [-6.2409, 106.9921],
            "Bogor": [-6.5944, 106.7896],
            "Cimahi": [-6.8723, 107.5427],
            "Cirebon": [-6.7263, 108.5522],
            "Depok": [-6.4025, 106.7942],
            "Sukabumi": [-6.9285, 106.9287],
            "Tasikmalaya": [-7.3274, 108.2206],
            # Jawa Tengah
            "Banjarnegara": [-7.4000, 109.7000],
            "Banyumas": [-7.4167, 109.2333],
            "Batang": [-7.0833, 109.8333],
            "Blora": [-7.0000, 111.4167],
            "Boyolali": [-7.5167, 110.6000],
            "Brebes": [-7.0333, 108.8500],
            "Cilacap": [-7.6333, 109.0167],
            "Demak": [-6.9000, 110.6333],
            "Grobogan": [-7.0833, 110.9167],
            "Jepara": [-6.5833, 110.6667],
            "Karanganyar": [-7.6000, 110.9500],
            "Kebumen": [-7.6667, 109.6667],
            "Kendal": [-7.0333, 110.1833],
            "Klaten": [-7.7000, 110.6000],
            "Kudus": [-6.8000, 110.8333],
            "Magelang": [-7.4667, 110.2167],
            "Pati": [-6.7500, 111.0333],
            "Pekalongan": [-7.0833, 109.6667],
            "Pemalang": [-6.9167, 109.3833],
            "Purbalingga": [-7.3833, 109.3667],
            "Purworejo": [-7.7167, 109.9167],
            "Rembang": [-6.7167, 111.3333],
            "Semarang": [-7.2167, 110.4333],
            "Sragen": [-7.4167, 111.0167],
            "Sukoharjo": [-7.6833, 110.8333],
            "Tegal": [-7.0333, 109.1333],
            "Temanggung": [-7.3167, 110.1833],
            "Wonogiri": [-7.9167, 110.9167],
            "Wonosobo": [-7.3667, 109.9000],
            "Magelang": [-7.4833, 110.2167],
            "Pekalongan": [-6.8833, 109.6667],
            "Salatiga": [-7.3333, 110.5000],
            "Semarang": [-7.0000, 110.4167],
            "Surakarta": [-7.5667, 110.8167],
            "Tegal": [-6.8667, 109.1333],
            # DI Yogyakarta
            "Bantul": [-7.8833, 110.3333],
            "Gunung Kidul": [-7.9667, 110.6000],
            "Kulon Progo": [-7.8500, 110.1667],
            "Sleman": [-7.7833, 110.3667],
            "Yogyakarta": [-7.8000, 110.3667],
            # Jawa Timur
            "Bangkalan": [-7.0500, 112.9333],
            "Banyuwangi": [-8.2167, 114.3667],
            "Blitar": [-8.1000, 112.1667],
            "Bojonegoro": [-7.1500, 111.8833],
            "Bondowoso": [-7.9167, 113.8167],
            "Gresik": [-7.1500, 112.6500],
            "Jember": [-8.1667, 113.7000],
            "Jombang": [-7.5500, 112.2333],
            "Kediri": [-7.8333, 112.0167],
            "Lamongan": [-7.1167, 112.4167],
            "Lumajang": [-8.1333, 113.2167],
            "Madiun": [-7.6333, 111.6500],
            "Magetan": [-7.6500, 111.3500],
            "Malang": [-8.1667, 112.6167],
            "Mojokerto": [-7.4667, 112.4333],
            "Nganjuk": [-7.6000, 111.9000],
            "Ngawi": [-7.4000, 111.4500],
            "Pacitan": [-8.2000, 111.1000],
            "Pamekasan": [-7.1500, 113.4667],
            "Pasuruan": [-7.6500, 112.9000],
            "Ponorogo": [-7.8667, 111.4667],
            "Probolinggo": [-7.7500, 113.2167],
            "Sampang": [-7.1833, 113.2500],
            "Sidoarjo": [-7.4500, 112.7167],
            "Situbondo": [-7.7000, 114.0000],
            "Sumenep": [-7.0167, 113.8500],
            "Trenggalek": [-8.0667, 111.7167],
            "Tuban": [-6.9000, 112.0500],
            "Tulungagung": [-8.0667, 111.9000],
            "Batu": [-7.8833, 112.5333],
            "Blitar": [-8.1000, 112.1667],
            "Kediri": [-7.8167, 112.0167],
            "Madiun": [-7.6333, 111.5167],
            "Malang": [-7.9833, 112.6167],
            "Mojokerto": [-7.4667, 112.4333],
            "Pasuruan": [-7.6500, 112.9000],
            "Probolinggo": [-7.7500, 113.2167],
            "Surabaya": [-7.2500, 112.7500],
            # Banten
            "Lebak": [-6.5500, 106.2500],
            "Pandeglang": [-6.6667, 105.8333],
            "Serang": [-6.1167, 106.1500],
            "Tangerang": [-6.1667, 106.6333],
            "Cilegon": [-6.0167, 106.0500],
            "Serang": [-6.1167, 106.1500],
            "Tangerang": [-6.1667, 106.6333],
            "Tangerang Selatan": [-6.2833, 106.7167],
            # Bali
            "Badung": [-8.5333, 115.2167],
            "Bangli": [-8.2833, 115.3500],
            "Buleleng": [-8.1167, 115.0833],
            "Gianyar": [-8.5333, 115.2667],
            "Jembrana": [-8.3500, 114.6500],
            "Karangasem": [-8.3333, 115.5833],
            "Klungkung": [-8.5333, 115.4000],
            "Tabanan": [-8.5333, 115.1167],
            "Denpasar": [-8.6500, 115.2167],
            # Nusa Tenggara Barat
            "Bima": [-8.5000, 118.5000],
            "Dompu": [-8.5333, 118.4667],
            "Lombok Barat": [-8.6500, 116.1167],
            "Lombok Tengah": [-8.7000, 116.2833],
            "Lombok Timur": [-8.5167, 116.4167],
            "Lombok Utara": [-8.3500, 116.2500],
            "Sumbawa": [-8.7500, 117.4167],
            "Sumbawa Barat": [-8.7167, 116.8500],
            "Bima": [-8.4667, 118.7333],
            "Mataram": [-8.5833, 116.1167],
            # Nusa Tenggara Timur
            "Alor": [-8.2500, 124.5000],
            "Belu": [-9.4167, 124.9167],
            "Ende": [-8.8333, 121.6667],
            "Flores Timur": [-8.3333, 122.9667],
            "Kupang": [-10.1667, 123.5833],
            "Lembata": [-8.3333, 123.5000],
            "Malaka": [-9.5667, 124.9000],
            "Manggarai": [-8.6333, 120.4667],
            "Manggarai Barat": [-8.6500, 119.8833],
            "Manggarai Timur": [-8.5833, 120.5833],
            "Nagekeo": [-8.8333, 121.2167],
            "Ngada": [-8.8000, 121.0000],
            "Rote Ndao": [-10.7333, 123.1333],
            "Sabu Raijua": [-10.5667, 121.8333],
            "Sikka": [-8.6667, 122.3333],
            "Sumba Barat": [-9.6500, 119.3333],
            "Sumba Barat Daya": [-9.6000, 119.1667],
            "Sumba Tengah": [-9.6667, 119.6667],
            "Sumba Timur": [-9.8333, 120.3333],
            "Timor Tengah Selatan": [-9.8333, 124.3333],
            "Timor Tengah Utara": [-9.4167, 124.5000],
            "Kupang": [-10.1667, 123.5833],
            # Kalimantan Barat
            "Bengkayang": [1.0333, 109.9667],
            "Kapuas Hulu": [0.8333, 112.9333],
            "Kayong Utara": [-1.2167, 110.1667],
            "Ketapang": [-1.8500, 109.9667],
            "Kubu Raya": [-0.0333, 109.3333],
            "Landak": [0.5000, 109.8333],
            "Melawi": [-0.7833, 111.7500],
            "Mempawah": [0.4500, 108.9667],
            "Sambas": [1.3500, 109.3000],
            "Sanggau": [0.1333, 110.5833],
            "Sekadau": [-0.0500, 110.9500],
            "Sintang": [0.0667, 111.4833],
            "Pontianak": [-0.0167, 109.3333],
            "Singkawang": [0.9167, 108.9833],
            # Kalimantan Tengah
            "Barito Selatan": [-1.9167, 114.7000],
            "Barito Timur": [-1.9333, 115.1000],
            "Barito Utara": [-0.9667, 114.9000],
            "Gunung Mas": [-0.9500, 113.5833],
            "Kapuas": [-2.0000, 114.3833],
            "Katingan": [-2.0667, 113.3833],
            "Kotawaringin Barat": [-2.1000, 111.7500],
            "Kotawaringin Timur": [-1.8333, 112.7500],
            "Lamandau": [-2.0000, 111.4500],
            "Murung Raya": [-0.0333, 114.3333],
            "Pulang Pisau": [-2.7500, 114.2500],
            "Sukamara": [-2.6333, 111.2333],
            "Seruyan": [-2.3333, 112.2500],
            "Palangka Raya": [-2.2167, 113.9167],
            # Kalimantan Selatan
            "Balangan": [-2.3333, 115.5833],
            "Banjar": [-3.3000, 114.8333],
            "Barito Kuala": [-3.1000, 114.5000],
            "Hulu Sungai Selatan": [-2.7667, 115.1667],
            "Hulu Sungai Tengah": [-2.5833, 115.2500],
            "Hulu Sungai Utara": [-2.4333, 115.1500],
            "Kotabaru": [-3.2500, 116.2167],
            "Tabalong": [-1.8333, 115.4167],
            "Tanah Bumbu": [-3.4500, 115.6667],
            "Tanah Laut": [-3.8000, 114.7667],
            "Tapin": [-2.8333, 115.0000],
            "Banjarbaru": [-3.4500, 114.8333],
            "Banjarmasin": [-3.3167, 114.5833],
            # Kalimantan Timur
            "Berau": [2.1667, 117.5000],
            "Kutai Barat": [-0.2500, 115.0000],
            "Kutai Kartanegara": [-0.4333, 116.9833],
            "Kutai Timur": [1.1667, 117.5000],
            "Mahakam Ulu": [0.5333, 115.5000],
            "Paser": [-1.8333, 116.1667],
            "Penajam Paser Utara": [-1.2500, 116.7000],
            "Balikpapan": [-1.2500, 116.8333],
            "Bontang": [0.1333, 117.5000],
            "Samarinda": [-0.5000, 117.1500],
            "Nusantara": [-0.9796, 116.7028],  # Ibu Kota Nusantara
            # Kalimantan Utara
            "Bulungan": [2.8333, 117.3333],
            "Malinau": [2.8333, 116.1333],
            "Nunukan": [4.1333, 117.6667],
            "Tana Tidung": [3.5000, 117.0000],
            "Tarakan": [3.3000, 117.6333],
            # Sulawesi Utara
            "Bolaang Mongondow": [0.7500, 124.0000],
            "Bolaang Mongondow Selatan": [0.4167, 123.6667],
            "Bolaang Mongondow Timur": [0.7500, 124.5833],
            "Bolaang Mongondow Utara": [0.8167, 123.3333],
            "Kepulauan Sangihe": [3.5833, 125.5000],
            "Kepulauan Siau Tagulandang Biaro": [2.0833, 125.4000],
            "Kepulauan Talaud": [4.2500, 126.7500],
            "Minahasa": [1.2500, 124.8333],
            "Minahasa Selatan": [1.1667, 124.5833],
            "Minahasa Tenggara": [1.0833, 124.8333],
            "Minahasa Utara": [1.4167, 125.0000],
            "Bitung": [1.4500, 125.1833],
            "Kotamobagu": [0.7333, 124.3167],
            "Manado": [1.4833, 124.8500],
            "Tomohon": [1.3333, 124.8333],
            # Sulawesi Tengah
            "Banggai": [-1.4167, 122.5833],
            "Banggai Kepulauan": [-1.6667, 123.5000],
            "Banggai Laut": [-1.8333, 123.5000],
            "Buol": [1.1667, 121.4167],
            "Donggala": [-0.7000, 119.7500],
            "Morowali": [-2.8333, 122.1667],
            "Morowali Utara": [-1.9167, 121.3333],
            "Parigi Moutong": [-1.1667, 120.1667],
            "Poso": [-1.3833, 120.7500],
            "Sigi": [-1.4000, 120.0000],
            "Tojo Una-Una": [-1.0000, 121.5000],
            "Tolitoli": [1.0000, 120.8167],
            "Palu": [-0.8833, 119.8667],
            # Sulawesi Selatan
            "Bantaeng": [-5.5333, 120.0000],
            "Barru": [-4.4167, 119.6667],
            "Bone": [-4.5333, 120.3333],
            "Bulukumba": [-5.4500, 120.1667],
            "Enrekang": [-3.5667, 119.7667],
            "Gowa": [-5.3333, 119.6667],
            "Jeneponto": [-5.6667, 119.7500],
            "Kepulauan Selayar": [-6.3333, 120.5000],
            "Luwu": [-3.3667, 120.1667],
            "Luwu Timur": [-2.5833, 121.1667],
            "Luwu Utara": [-2.5000, 120.1667],
            "Maros": [-5.0000, 119.5667],
            "Pangkajene dan Kepulauan": [-4.8333, 119.5500],
            "Pinrang": [-3.8000, 119.6500],
            "Sidenreng Rappang": [-3.9167, 119.8333],
            "Sinjai": [-5.2000, 120.2500],
            "Soppeng": [-4.3500, 119.8833],
            "Takalar": [-5.4167, 119.4500],
            "Tana Toraja": [-3.0500, 119.7500],
            "Toraja Utara": [-2.9667, 119.8833],
            "Wajo": [-4.1167, 120.0333],
            "Makassar": [-5.1333, 119.4167],
            "Palopo": [-3.0000, 120.2000],
            "Parepare": [-4.0167, 119.6333],
            # Sulawesi Tenggara
            "Bombana": [-4.6000, 121.8333],
            "Buton": [-5.4500, 122.9167],
            "Buton Selatan": [-5.6333, 122.6667],
            "Buton Tengah": [-5.3833, 122.4167],
            "Buton Utara": [-4.7500, 123.0833],
            "Kolaka": [-4.0833, 121.5833],
            "Kolaka Timur": [-4.0167, 121.8833],
            "Kolaka Utara": [-3.3667, 121.1667],
            "Konawe": [-3.9167, 122.0833],
            "Konawe Kepulauan": [-4.1667, 123.0833],
            "Konawe Selatan": [-4.2500, 122.4167],
            "Konawe Utara": [-3.4167, 122.0000],
            "Muna": [-4.8333, 122.6667],
            "Muna Barat": [-4.8667, 122.4833],
            "Wakatobi": [-5.3167, 123.5833],
            "Bau-Bau": [-5.4667, 122.6000],
            "Kendari": [-3.9833, 122.5167],
            # Gorontalo
            "Boalemo": [0.7500, 122.3333],
            "Bone Bolango": [0.5333, 123.2500],
            "Gorontalo": [0.7000, 122.8333],
            "Gorontalo Utara": [0.8167, 122.3167],
            "Pohuwato": [0.7000, 121.6667],
            "Gorontalo": [0.5333, 123.0667],
            # Sulawesi Barat
            "Majene": [-3.5333, 118.9667],
            "Mamasa": [-2.9500, 119.3833],
            "Mamuju": [-2.6667, 118.8833],
            "Mamuju Tengah": [-2.3333, 119.3333],
            "Pasangkayu": [-1.3167, 119.3667],
            "Polewali Mandar": [-3.4333, 119.3333],
            # Maluku
            "Buru": [-3.3500, 126.6667],
            "Buru Selatan": [-3.6667, 126.3333],
            "Kepulauan Aru": [-6.1667, 134.5000],
            "Kepulauan Tanimbar": [-7.5000, 131.5000],
            "Maluku Barat Daya": [-7.8333, 127.8333],
            "Maluku Tengah": [-3.0000, 129.0000],
            "Maluku Tenggara": [-5.6667, 132.7500],
            "Seram Bagian Barat": [-3.3333, 128.3333],
            "Seram Bagian Timur": [-3.6667, 130.5000],
            "Ambon": [-3.7000, 128.1667],
            "Tual": [-5.6333, 132.7500],
            # Maluku Utara
            "Halmahera Barat": [1.5000, 127.5000],
            "Halmahera Tengah": [0.5000, 128.5000],
            "Halmahera Timur": [1.3333, 128.3333],
            "Halmahera Selatan": [-0.5000, 127.5000],
            "Halmahera Utara": [1.8333, 128.0000],
            "Kepulauan Sula": [-2.0000, 125.0000],
            "Pulau Morotai": [2.2500, 128.4167],
            "Pulau Taliabu": [-1.8333, 124.6667],
            "Ternate": [0.7833, 127.3833],
            "Tidore Kepulauan": [0.6833, 127.4167],
            # Papua Barat
            "Fakfak": [-2.9167, 132.3000],
            "Kaimana": [-3.6667, 133.7500],
            "Manokwari": [-0.8667, 134.0833],
            "Manokwari Selatan": [-1.3333, 134.0000],
            "Pegunungan Arfak": [-1.5000, 133.5000],
            "Teluk Bintuni": [-2.1667, 133.3333],
            "Teluk Wondama": [-2.6667, 134.5000],
            # Papua Barat Daya
            "Maybrat": [-1.3333, 132.4167],
            "Raja Ampat": [-0.5000, 130.5000],
            "Sorong": [-1.0000, 131.2500],
            "Sorong Selatan": [-1.5000, 132.2500],
            "Tambrauw": [-0.8667, 132.6667],
            "Sorong": [-0.8833, 131.2500],
            # Papua
            "Biak Numfor": [-1.0000, 136.0000],
            "Jayapura": [-2.5333, 140.7167],
            "Keerom": [-3.3500, 140.6667],
            "Kepulauan Yapen": [-1.7500, 136.2500],
            "Mamberamo Raya": [-2.3333, 138.0000],
            "Sarmi": [-1.8333, 138.7500],
            "Supiori": [-0.7500, 135.5000],
            "Waropen": [-2.7500, 136.5833],
            "Jayapura": [-2.5333, 140.7167],
            # Papua Tengah
            "Deiyai": [-4.0833, 136.3333],
            "Dogiyai": [-4.0500, 135.4333],
            "Intan Jaya": [-3.7500, 136.6667],
            "Mimika": [-4.6333, 136.4167],
            "Nabire": [-3.3667, 135.5000],
            "Paniai": [-3.8333, 136.0000],
            "Puncak": [-4.0000, 137.4833],
            "Puncak Jaya": [-3.7500, 138.0000],
            # Papua Pegunungan
            "Jayawijaya": [-4.0833, 138.9167],
            "Lanny Jaya": [-3.8333, 138.4167],
            "Mamberamo Tengah": [-3.0000, 138.5000],
            "Nduga": [-4.4500, 138.0833],
            "Pegunungan Bintang": [-4.5000, 140.5000],
            "Tolikara": [-3.5000, 138.5000],
            "Yahukimo": [-4.5000, 139.5000],
            "Yalimo": [-3.8333, 139.5000],
            # Papua Selatan
            "Asmat": [-5.3333, 138.3333],
            "Boven Digoel": [-6.0833, 140.3333],
            "Mappi": [-6.5000, 139.3333],
            "Merauke": [-8.4833, 140.3333],
        }

        # Test matching
        matched = 0
        unmatched = []

        for city in unique_cities:
            # Normalize city name (similar to frontend logic)
            normalized = city.strip()
            for prefix in ["Kab. ", "Kabupaten ", "Kota ", "Administrasi ", "DKI "]:
                normalized = normalized.replace(prefix, "")
            normalized = normalized.strip()

            if normalized in js_coords:
                matched += 1
                print(f"✅ {city} -> {normalized} -> {js_coords[normalized]}")
            else:
                # Try fuzzy matching
                fuzzy_match = None
                for key in js_coords.keys():
                    if (
                        key.lower() in normalized.lower()
                        or normalized.lower() in key.lower()
                    ):
                        fuzzy_match = key
                        break

                if fuzzy_match:
                    matched += 1
                    print(
                        f"🔍 {city} -> {fuzzy_match} (fuzzy) -> {js_coords[fuzzy_match]}"
                    )
                else:
                    unmatched.append(city)

        print(f"\n=== Matching Results ===")
        print(f"Total cities: {len(unique_cities)}")
        print(f"Matched: {matched}")
        print(f"Unmatched: {len(unmatched)}")
        print(f"Match rate: {(matched/len(unique_cities)*100):.1f}%")

        if unmatched:
            print(f"\nUnmatched cities (first 10):")
            for city in unmatched[:10]:
                print(f"❌ {city}")

        return matched, len(unmatched)

    except Exception as e:
        print(f"Error: {e}")
        return 0, 0


def create_test_clustering_data():
    """Create test data with known city names"""

    test_data = {
        "kabupaten_kota": [
            "Jakarta Pusat",
            "Jakarta Selatan",
            "Jakarta Barat",
            "Surabaya",
            "Bandung",
            "Semarang",
            "Medan",
            "Makassar",
            "Palembang",
            "Yogyakarta",
            "Malang",
            "Bogor",
        ],
        "tahun": [2023] * 12,
        "ipm": [85, 83, 82, 78, 76, 75, 72, 70, 73, 80, 77, 79],
        "garis_kemiskinan": [
            400000,
            420000,
            450000,
            480000,
            500000,
            470000,
            520000,
            550000,
            510000,
            440000,
            460000,
            430000,
        ],
        "pengeluaran_per_kapita": [
            15000000,
            14000000,
            13000000,
            9000000,
            10000000,
            8500000,
            7000000,
            6500000,
            7500000,
            11000000,
            8800000,
            12000000,
        ],
    }

    df = pd.DataFrame(test_data)

    # Save as CSV for testing
    df.to_csv("test_map_data.csv", index=False)
    print("✅ Created test_map_data.csv with known city coordinates")

    return df


if __name__ == "__main__":
    print("Testing coordinate matching...")

    # Test with sample data
    matched, unmatched = test_coordinate_matching()

    # Create test data
    test_df = create_test_clustering_data()
    print(f"\nTest data created with {len(test_df)} cities")
    print("Cities in test data:")
    for city in test_df["kabupaten_kota"]:
        print(f"  - {city}")
