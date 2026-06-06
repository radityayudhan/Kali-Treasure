import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# ==========================================
# 2. FUNGSI UTILITAS (GEOJSON & ON-THE-FLY ETL)
# ==========================================
@st.cache_data
def load_and_filter_geojson(filepath, level='provinsi'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
            
        daftar_kalimantan = [
            "Kalimantan Barat", "Kalimantan Tengah", 
            "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara"
        ]
        
        fitur_tersaring = []
        for feature in geojson_data['features']:
            if level == 'provinsi':
                if feature['properties'].get('PROVINSI') in daftar_kalimantan:
                    fitur_tersaring.append(feature)
            elif level == 'kabkot':
                if feature['properties'].get('WADMPR') in daftar_kalimantan:
                    fitur_tersaring.append(feature)
                    
        return {"type": "FeatureCollection", "features": fitur_tersaring}
    except FileNotFoundError:
        st.error(f"File GeoJSON tidak ditemukan di: {filepath}")
        return None

@st.cache_data
def load_and_process_raw_data():
    file_kur = os.path.join('data', 'Realisasi_KUR_Kalimantan.csv')
    file_umi = os.path.join('data', 'Realisasi_UMI_Kalimantan.csv')
    
    if not os.path.exists(file_kur) or not os.path.exists(file_umi):
        st.error("File mentah Realisasi KUR/UMi tidak ditemukan di folder 'data'.")
        return pd.DataFrame()

    df_kur = pd.read_csv(file_kur, sep=';', low_memory=False)
    df_umi = pd.read_csv(file_umi, sep=';', low_memory=False)

    df_kur['PROGRAM'] = 'KUR'
    df_umi['PROGRAM'] = 'UMI'

    if 'NAMA_WILAYAH' in df_kur.columns:
        df_kur.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT', 'KODE_WILAYAH': 'KODE_KABKOT'}, inplace=True)
    if 'NAMA_WILAYAH' in df_umi.columns:
        df_umi.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT', 'KODE_WILAYAH': 'KODE_KABKOT'}, inplace=True)

    df_gabungan = pd.concat([df_kur, df_umi], ignore_index=True)
    df_gabungan['TAHUN'] = pd.to_numeric(df_gabungan['TAHUN'], errors='coerce').fillna(2023).astype(int)

    if 'NAMA_JNS_KELAMIN' in df_gabungan.columns:
        df_gabungan['DEBITUR_PEREMPUAN'] = np.where(
            df_gabungan['NAMA_JNS_KELAMIN'].astype(str).str.upper().str.startswith('P'),
            df_gabungan['SUM_JML_DEBITUR'], 0
        )
    else:
        df_gabungan['DEBITUR_PEREMPUAN'] = 0

    if 'NAMA_PENDIDIKAN' in df_gabungan.columns:
        kondisi_dasar = df_gabungan['NAMA_PENDIDIKAN'].astype(str).str.upper().str.contains('SD|SMP|DASAR|PERTAMA|SLTP|MI|MTS', regex=True)
        df_gabungan['PENDIDIKAN_SD_SMP'] = np.where(
            kondisi_dasar, df_gabungan['SUM_JML_DEBITUR'], 0
        )
    else:
        df_gabungan['PENDIDIKAN_SD_SMP'] = 0
    
    # Mengambil sektor dari data KUR
    sektor_kur = df_gabungan['NAMA_SEKTOR'] if 'NAMA_SEKTOR' in df_gabungan.columns else pd.Series([np.nan]*len(df_gabungan))
    df_gabungan['SEKTOR_USAHA'] = sektor_kur
    
    # Asumsi Lapangan: Seluruh program UMi dialokasikan ke Sektor Perdagangan Besar dan Eceran
    df_gabungan['SEKTOR_USAHA'] = np.where(
        df_gabungan['PROGRAM'] == 'UMI', 
        'PERDAGANGAN BESAR DAN ECERAN', 
        df_gabungan['SEKTOR_USAHA']
    )
    
    # Bersihkan teks Sektor agar seragam (Title Case)
    df_gabungan['SEKTOR_USAHA'] = df_gabungan['SEKTOR_USAHA'].astype(str).str.title().str.strip()
    df_gabungan['SEKTOR_USAHA'] = df_gabungan['SEKTOR_USAHA'].replace({'Nan': 'Sektor Lainnya', 'Tidak Diketahui': 'Sektor Lainnya'})
    df_gabungan['NAMA_PROVINSI'] = df_gabungan['NAMA_PROVINSI'].astype(str).str.strip().str.upper()
    df_gabungan['NAMA_KABKOT'] = df_gabungan['NAMA_KABKOT'].astype(str).str.strip().str.upper()

    agregat_spasial = df_gabungan.groupby(['TAHUN', 'PROGRAM', 'NAMA_PROVINSI', 'NAMA_KABKOT', 'SEKTOR_USAHA']).agg(
        TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum'),
        TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum'),
        DEBITUR_PEREMPUAN=('DEBITUR_PEREMPUAN', 'sum'),
        PENDIDIKAN_SD_SMP=('PENDIDIKAN_SD_SMP', 'sum')
    ).reset_index()

    return agregat_spasial

@st.cache_data
def load_ews_skema_data():
    """Mesin ETL Khusus untuk Analisis Efisiensi Skema (EWS) - Fokus pada irisan waktu <= 2021"""
    file_kur = os.path.join('data', 'Realisasi_KUR_Kalimantan.csv')
    file_sub = os.path.join('data', 'Subsidi_KUR_Kalimantan_Clean.csv')
    
    if not os.path.exists(file_kur) or not os.path.exists(file_sub):
        return pd.DataFrame()

    df_kur = pd.read_csv(file_kur, sep=';', low_memory=False)
    df_sub = pd.read_csv(file_sub, sep=';', low_memory=False)

    # 1. Penyelarasan Nomenklatur Wilayah dan Skema
    if 'NAMA_WILAYAH' in df_kur.columns: 
        df_kur.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT'}, inplace=True)
    if 'NAMA_WILAYAH' in df_sub.columns: 
        df_sub.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT'}, inplace=True)

    df_kur['NAMA_KABKOT'] = df_kur['NAMA_KABKOT'].astype(str).str.strip().str.upper()
    df_sub['NAMA_KABKOT'] = df_sub['NAMA_KABKOT'].astype(str).str.strip().str.upper()

    # --- BARIS PERBAIKAN: PENYELARASAN NAMA SKEMA ---
    df_kur['NAMA_SKEMA'] = df_kur['NAMA_SKEMA'].astype(str).str.strip().str.upper()
    
    # Memotong string " - KMK" atau " - KI" pada data subsidi agar cocok dengan realisasi
    df_sub['NAMA_SKEMA'] = df_sub['NAMA_SKEMA'].astype(str).str.upper()
    df_sub['NAMA_SKEMA'] = df_sub['NAMA_SKEMA'].str.split('-').str[0].str.strip()
    # ------------------------------------------------

    df_kur['TAHUN'] = pd.to_numeric(df_kur['TAHUN'], errors='coerce').fillna(0)
    df_sub['TAHUN'] = pd.to_numeric(df_sub['TAHUN'], errors='coerce').fillna(0)

    # 2. FILTER TIME HORIZON: Mengambil irisan data yang valid (Misal: 2016 - 2021)
    df_kur_lap = df_kur[(df_kur['TAHUN'] >= 2016) & (df_kur['TAHUN'] <= 2021)]
    df_sub_lap = df_sub[(df_sub['TAHUN'] >= 2016) & (df_sub['TAHUN'] <= 2021)]

    # 3. Agregasi berdasar Wilayah dan SKEMA (Bukan Bank)
    agg_kur = df_kur_lap.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT', 'NAMA_SKEMA']).agg(
        TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum'),
        TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum')
    ).reset_index()

    agg_sub = df_sub_lap.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT', 'NAMA_SKEMA']).agg(
        TOTAL_SUBSIDI=('SUM_NILAI_SUBSIDI', 'sum')
    ).reset_index()

    # 4. INNER JOIN: Menggabungkan Realisasi dan Subsidi
    df_ews = pd.merge(agg_kur, agg_sub, on=['NAMA_PROVINSI', 'NAMA_KABKOT', 'NAMA_SKEMA'], how='inner')

    # 5. KALKULASI RASIO INEFISIENSI
    df_ews['SUBSIDI_PER_DEBITUR'] = df_ews['TOTAL_SUBSIDI'] / df_ews['TOTAL_DEBITUR']
    df_ews['PENYALURAN_PER_DEBITUR'] = df_ews['TOTAL_PENYALURAN'] / df_ews['TOTAL_DEBITUR']

    # Buang outlier (skema yang punya kurang dari 10 debitur historis di wilayah tersebut)
    df_ews = df_ews[df_ews['TOTAL_DEBITUR'] > 10].reset_index(drop=True)
    
    return df_ews