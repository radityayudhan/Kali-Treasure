import streamlit as st
import pandas as pd
import numpy as np
import json
import os

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
    file_penduduk = os.path.join('data', 'Penduduk_Kalimantan.csv')
    
    if not os.path.exists(file_kur) or not os.path.exists(file_umi):
        st.error("File mentah Realisasi KUR/UMi tidak ditemukan di folder 'data'.")
        return pd.DataFrame()

    df_kur_raw = pd.read_csv(file_kur, sep=';', low_memory=False)
    df_umi_raw = pd.read_csv(file_umi, sep=';', low_memory=False)

    #pisahkan data UMi dari data KUR
    kondisi_umi_di_kur = False
    if 'NAMA_SKEMA' in df_kur_raw.columns:
        kondisi_umi_di_kur = df_kur_raw['NAMA_SKEMA'].astype(str).str.upper() == 'UMI'

    df_kur = df_kur_raw[~kondisi_umi_di_kur].copy()
    df_umi_dari_kur = df_kur_raw[kondisi_umi_di_kur].copy()

    df_kur['PROGRAM'] = 'KUR'
    df_umi_dari_kur['PROGRAM'] = 'UMI'
    df_umi_raw['PROGRAM'] = 'UMI'

    for df in [df_kur, df_umi_dari_kur, df_umi_raw]:
        if 'NAMA_WILAYAH' in df.columns:
            df.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT', 'KODE_WILAYAH': 'KODE_KABKOT'}, inplace=True)
            
        # Ekstrak TAHUN dan BULAN untuk Filtering Splicing
        df['TAHUN'] = pd.to_numeric(df['TAHUN'], errors='coerce').fillna(2023).astype(int)
        df['BULAN'] = pd.to_numeric(df.get('BULAN', 1), errors='coerce').fillna(1).astype(int)
        df['TAHUN_BULAN'] = df['TAHUN'] * 100 + df['BULAN']

    batas_waktu_splicing = 202404
    
    df_umi_part1 = df_umi_dari_kur[df_umi_dari_kur['TAHUN_BULAN'] <= batas_waktu_splicing].copy()
    df_umi_part2 = df_umi_raw[df_umi_raw['TAHUN_BULAN'] > batas_waktu_splicing].copy()

    df_gabungan = pd.concat([df_kur, df_umi_part1, df_umi_part2], ignore_index=True)

    if 'NAMA_JNS_KELAMIN' in df_gabungan.columns:
        df_gabungan['DEBITUR_PEREMPUAN'] = np.where(
            df_gabungan['NAMA_JNS_KELAMIN'].astype(str).str.upper().str.startswith('P'),
            df_gabungan['SUM_JML_DEBITUR'], 0
        )
    else:
        df_gabungan['DEBITUR_PEREMPUAN'] = 0

    if 'NAMA_PENDIDIKAN' in df_gabungan.columns:
        kondisi_dasar = df_gabungan['NAMA_PENDIDIKAN'].astype(str).str.upper().str.contains('SD|SMP|DASAR|PERTAMA|SLTP|MI|MTS', regex=True)
        df_gabungan['PENDIDIKAN_SD_SMP'] = np.where(kondisi_dasar, df_gabungan['SUM_JML_DEBITUR'], 0)
    else:
        df_gabungan['PENDIDIKAN_SD_SMP'] = 0

    if 'NAMA_SEKTOR' not in df_gabungan.columns:
        df_gabungan['NAMA_SEKTOR'] = np.nan
        
    df_gabungan['SEKTOR_USAHA'] = df_gabungan['NAMA_SEKTOR']

    df_gabungan['SEKTOR_USAHA'] = np.where(
        (df_gabungan['PROGRAM'] == 'UMI') & (df_gabungan['SEKTOR_USAHA'].isna() | (df_gabungan['SEKTOR_USAHA'] == '')), 
        'PERDAGANGAN BESAR DAN ECERAN', 
        df_gabungan['SEKTOR_USAHA']
    )
    
    df_gabungan['SEKTOR_USAHA'] = df_gabungan['SEKTOR_USAHA'].astype(str).str.title().str.strip()
    df_gabungan['SEKTOR_USAHA'] = df_gabungan['SEKTOR_USAHA'].replace({'Nan': 'Sektor Lainnya', 'Tidak Diketahui': 'Sektor Lainnya'})
    
    df_gabungan['NAMA_PROVINSI'] = df_gabungan['NAMA_PROVINSI'].astype(str).str.strip().str.upper()
    df_gabungan['NAMA_KABKOT'] = df_gabungan['NAMA_KABKOT'].astype(str).str.strip().str.upper()

    if 'NAMA_JNS_KELAMIN' not in df_gabungan.columns: df_gabungan['NAMA_JNS_KELAMIN'] = 'NV'
    if 'NAMA_MARITAL_STS' not in df_gabungan.columns: df_gabungan['NAMA_MARITAL_STS'] = 'NV'
    
    df_gabungan['NAMA_JNS_KELAMIN'] = df_gabungan['NAMA_JNS_KELAMIN'].astype(str).str.upper().str.strip()
    df_gabungan['NAMA_MARITAL_STS'] = df_gabungan['NAMA_MARITAL_STS'].astype(str).str.upper().str.strip()

    agregat_spasial = df_gabungan.groupby([
        'TAHUN', 'PROGRAM', 'NAMA_PROVINSI', 'NAMA_KABKOT', 'SEKTOR_USAHA', 
        'NAMA_JNS_KELAMIN', 'NAMA_MARITAL_STS' 
    ]).agg(
        TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum'),
        TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum'),
        TOTAL_OUTSTANDING=('SUM_JML_OUTSTANDING', 'sum'),
        DEBITUR_PEREMPUAN=('DEBITUR_PEREMPUAN', 'sum'),
        PENDIDIKAN_SD_SMP=('PENDIDIKAN_SD_SMP', 'sum')
    ).reset_index()

    if os.path.exists(file_penduduk):
        df_pop = pd.read_csv(file_penduduk, sep=';') 
        df_pop['NAMA_KABKOT'] = df_pop['NAMA_KABKOT'].astype(str).str.upper().str.strip()
        df_pop['JUMLAH_PENDUDUK'] = df_pop['JUMLAH_PENDUDUK'].astype(str).str.replace(',', '.').astype(float)
        df_pop['JUMLAH_PENDUDUK'] = df_pop['JUMLAH_PENDUDUK'] * 1000 
        df_pop['JUMLAH_PENDUDUK'] = df_pop['JUMLAH_PENDUDUK'] * 0.70 
        
        agregat_spasial = pd.merge(agregat_spasial, df_pop, on=['TAHUN', 'NAMA_KABKOT'], how='left')
        agregat_spasial['JUMLAH_PENDUDUK'] = agregat_spasial.groupby('NAMA_KABKOT')['JUMLAH_PENDUDUK'].transform(lambda x: x.ffill().bfill())
    else:
        agregat_spasial['JUMLAH_PENDUDUK'] = 0

    return agregat_spasial

@st.cache_data
def load_ews_skema_data():
    file_kur = os.path.join('data', 'Realisasi_KUR_Kalimantan.csv')
    file_sub = os.path.join('data', 'Subsidi_KUR_Kalimantan_Clean.csv')
    
    if not os.path.exists(file_kur) or not os.path.exists(file_sub):
        return pd.DataFrame()

    df_kur = pd.read_csv(file_kur, sep=';', low_memory=False)
    df_sub = pd.read_csv(file_sub, sep=';', low_memory=False)

    # Buang UMi (Fokus murni efisiensi subsidi KUR)
    if 'NAMA_SKEMA' in df_kur.columns:
        df_kur = df_kur[df_kur['NAMA_SKEMA'].astype(str).str.upper() != 'UMI']
    
    if 'NAMA_WILAYAH' in df_kur.columns: df_kur.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT'}, inplace=True)
    if 'NAMA_WILAYAH' in df_sub.columns: df_sub.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT'}, inplace=True)

    df_kur['NAMA_PROVINSI'] = df_kur['NAMA_PROVINSI'].astype(str).str.strip().str.upper()
    df_sub['NAMA_PROVINSI'] = df_sub['NAMA_PROVINSI'].astype(str).str.strip().str.upper()

    df_kur['NAMA_KABKOT'] = df_kur['NAMA_KABKOT'].astype(str).str.strip().str.upper()
    df_sub['NAMA_KABKOT'] = df_sub['NAMA_KABKOT'].astype(str).str.strip().str.upper()

    df_kur['NAMA_SKEMA'] = df_kur['NAMA_SKEMA'].astype(str).str.strip().str.upper()
    df_sub['NAMA_SKEMA'] = df_sub['NAMA_SKEMA'].astype(str).str.upper().str.split('-').str[0].str.strip()

    df_kur['NAMA_SKEMA'] = df_kur['NAMA_SKEMA'].replace({'SUPERMI': 'SUPER MIKRO'})
    df_sub['NAMA_SKEMA'] = df_sub['NAMA_SKEMA'].replace({'KECIL KHUSUS': 'KECIL'})

    df_kur['TAHUN'] = pd.to_numeric(df_kur['TAHUN'], errors='coerce').fillna(0)
    df_sub['TAHUN'] = pd.to_numeric(df_sub['TAHUN'], errors='coerce').fillna(0)

    df_kur_lap = df_kur[(df_kur['TAHUN'] >= 2016) & (df_kur['TAHUN'] <= 2021)]
    df_sub_lap = df_sub[(df_sub['TAHUN'] >= 2016) & (df_sub['TAHUN'] <= 2021)]

    agg_kur = df_kur_lap.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT', 'NAMA_SKEMA']).agg(
        TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum'),
        TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum')
    ).reset_index()

    agg_sub = df_sub_lap.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT', 'NAMA_SKEMA']).agg(
        TOTAL_SUBSIDI=('SUM_NILAI_SUBSIDI', 'sum')
    ).reset_index()

    df_ews = pd.merge(agg_kur, agg_sub, on=['NAMA_PROVINSI', 'NAMA_KABKOT', 'NAMA_SKEMA'], how='inner')
    df_ews['SUBSIDI_PER_DEBITUR'] = df_ews['TOTAL_SUBSIDI'] / df_ews['TOTAL_DEBITUR']
    df_ews['PENYALURAN_PER_DEBITUR'] = df_ews['TOTAL_PENYALURAN'] / df_ews['TOTAL_DEBITUR']
    df_ews = df_ews[df_ews['TOTAL_DEBITUR'] > 10].reset_index(drop=True)
    
    return df_ews

@st.cache_data
def load_leverage_trend_data():
    file_kur = os.path.join('data', 'Realisasi_KUR_Kalimantan.csv')
    file_sub = os.path.join('data', 'Subsidi_KUR_Kalimantan_Clean.csv')
    
    if not os.path.exists(file_kur) or not os.path.exists(file_sub):
        return pd.DataFrame()

    df_kur = pd.read_csv(file_kur, sep=';', low_memory=False)
    df_sub = pd.read_csv(file_sub, sep=';', low_memory=False)

    # Buang UMi (Fokus murni efisiensi subsidi KUR)
    if 'NAMA_SKEMA' in df_kur.columns:
        df_kur = df_kur[df_kur['NAMA_SKEMA'].astype(str).str.upper() != 'UMI']
    

    df_kur['TAHUN'] = pd.to_numeric(df_kur['TAHUN'], errors='coerce').fillna(0).astype(int)
    df_sub['TAHUN'] = pd.to_numeric(df_sub['TAHUN'], errors='coerce').fillna(0).astype(int)
    
    if 'NAMA_WILAYAH' in df_kur.columns: df_kur.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT'}, inplace=True)
    if 'NAMA_WILAYAH' in df_sub.columns: df_sub.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT'}, inplace=True)

    df_kur['NAMA_PROVINSI'] = df_kur['NAMA_PROVINSI'].astype(str).str.strip().str.upper()
    df_kur['NAMA_KABKOT'] = df_kur['NAMA_KABKOT'].astype(str).str.strip().str.upper()
    df_sub['NAMA_PROVINSI'] = df_sub['NAMA_PROVINSI'].astype(str).str.strip().str.upper()
    df_sub['NAMA_KABKOT'] = df_sub['NAMA_KABKOT'].astype(str).str.strip().str.upper()

    df_kur_lap = df_kur[(df_kur['TAHUN'] >= 2017) & (df_kur['TAHUN'] <= 2021)]
    df_sub_lap = df_sub[(df_sub['TAHUN'] >= 2017) & (df_sub['TAHUN'] <= 2021)]

    agg_kur = df_kur_lap.groupby(['TAHUN', 'NAMA_PROVINSI', 'NAMA_KABKOT']).agg(
        TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum')
    ).reset_index()

    agg_sub = df_sub_lap.groupby(['TAHUN', 'NAMA_PROVINSI', 'NAMA_KABKOT']).agg(
        TOTAL_SUBSIDI=('SUM_NILAI_SUBSIDI', 'sum')
    ).reset_index()

    df_trend = pd.merge(agg_kur, agg_sub, on=['TAHUN', 'NAMA_PROVINSI', 'NAMA_KABKOT'], how='inner')
    
    return df_trend

@st.cache_data
def load_graduasi_trend_data():
    file_kur = os.path.join('data', 'Realisasi_KUR_Kalimantan.csv')
    file_umi = os.path.join('data', 'Realisasi_UMI_Kalimantan.csv')
    
    if not os.path.exists(file_kur) or not os.path.exists(file_umi):
        return pd.DataFrame()

    df_kur_raw = pd.read_csv(file_kur, sep=';', low_memory=False)
    df_umi_raw = pd.read_csv(file_umi, sep=';', low_memory=False)

    kondisi_umi_di_kur = False
    if 'NAMA_SKEMA' in df_kur_raw.columns:
        kondisi_umi_di_kur = df_kur_raw['NAMA_SKEMA'].astype(str).str.upper() == 'UMI'

    df_kur = df_kur_raw[~kondisi_umi_di_kur].copy()
    df_umi_dari_kur = df_kur_raw[kondisi_umi_di_kur].copy()

    if 'NAMA_SKEMA' in df_kur.columns:
        df_kur = df_kur[~df_kur['NAMA_SKEMA'].isin(['SUPERMI', 'SUPER MIKRO'])]
    
    df_kur['PROGRAM'] = 'KUR'
    df_umi_dari_kur['PROGRAM'] = 'UMI'
    df_umi_raw['PROGRAM'] = 'UMI'

    for df in [df_kur, df_umi_dari_kur, df_umi_raw]:
        if 'NAMA_WILAYAH' in df.columns: 
            df.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT'}, inplace=True)
        df['TAHUN'] = pd.to_numeric(df['TAHUN'], errors='coerce').fillna(2023).astype(int)
        df['BULAN'] = pd.to_numeric(df.get('BULAN', 1), errors='coerce').fillna(1).astype(int)
        df['TAHUN_BULAN'] = df['TAHUN'] * 100 + df['BULAN']

    batas_waktu_splicing = 202404
    df_umi_part1 = df_umi_dari_kur[df_umi_dari_kur['TAHUN_BULAN'] <= batas_waktu_splicing].copy()
    df_umi_part2 = df_umi_raw[df_umi_raw['TAHUN_BULAN'] > batas_waktu_splicing].copy()

    df_gabung = pd.concat([df_kur, df_umi_part1, df_umi_part2], ignore_index=True)

    df_gabung['NAMA_PROVINSI'] = df_gabung['NAMA_PROVINSI'].astype(str).str.strip().str.upper()
    df_gabung['NAMA_KABKOT'] = df_gabung['NAMA_KABKOT'].astype(str).str.strip().str.upper()

    df_gabung = df_gabung[(df_gabung['TAHUN'] >= 2017) & (df_gabung['TAHUN'] <= 2023)]

    agg_graduasi = df_gabung.groupby(['TAHUN', 'PROGRAM', 'NAMA_PROVINSI', 'NAMA_KABKOT']).agg(
        TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum')
    ).reset_index()

    df_pivot = agg_graduasi.pivot_table(
        index=['TAHUN', 'NAMA_PROVINSI', 'NAMA_KABKOT'], 
        columns='PROGRAM', 
        values='TOTAL_DEBITUR', 
        fill_value=0
    ).reset_index()

    if 'KUR' not in df_pivot.columns: df_pivot['KUR'] = 0
    if 'UMI' not in df_pivot.columns: df_pivot['UMI'] = 0

    df_pivot.rename(columns={'KUR': 'DEBITUR_KUR', 'UMI': 'DEBITUR_UMI'}, inplace=True)
    
    return df_pivot