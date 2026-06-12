import pandas as pd
import os
import numpy as np

print("=== MEMULAI EDA TAB 4: WHAT-IF POLICY SIMULATOR (BASELINE MULTIPLIERS) ===")

folder_data = 'data'
file_kur = os.path.join(folder_data, 'Realisasi_KUR_Kalimantan.csv')
file_umi = os.path.join(folder_data, 'Realisasi_UMI_Kalimantan.csv')

out_baseline_sektoral = os.path.join(folder_data, 'baseline_sektoral_tab4.csv')
out_baseline_regional = os.path.join(folder_data, 'baseline_regional_tab4.csv')

df_kur = pd.read_csv(file_kur, sep=';', low_memory=False)
df_umi = pd.read_csv(file_umi, sep=';', low_memory=False)

def selaraskan_kolom(df):
    mapper = {}
    for col in df.columns:
        col_up = col.upper()
        if col_up in ['NAMA_SEKTOR', 'SEKTOR_USAHA']: mapper[col] = 'NAMA_SEKTOR'
        elif col_up in ['NAMA_KABKOT', 'NAMA_WILAYAH']: mapper[col] = 'NAMA_KABKOT'
        elif col_up in ['NAMA_PROVINSI', 'PROVINSI']: mapper[col] = 'NAMA_PROVINSI'
    if mapper: df.rename(columns=mapper, inplace=True)
    return df

df_kur = selaraskan_kolom(df_kur)
df_umi = selaraskan_kolom(df_umi)

df_kur['PROGRAM'] = 'KUR'
df_umi['PROGRAM'] = 'UMI'

# Menggabungkan data (Hanya mengambil kolom yang penting untuk simulator)
kolom_esensial = ['NAMA_PROVINSI', 'NAMA_KABKOT', 'NAMA_SEKTOR', 'SUM_JML_PENYALURAN', 'SUM_JML_DEBITUR', 'PROGRAM']
for col in kolom_esensial:
    if col not in df_kur.columns: df_kur[col] = 'TIDAK DIKETAHUI'
    if col not in df_umi.columns: df_umi[col] = 'TIDAK DIKETAHUI'

df_gabungan = pd.concat([df_kur[kolom_esensial], df_umi[kolom_esensial]], ignore_index=True)
df_gabungan['SUM_JML_PENYALURAN'] = pd.to_numeric(df_gabungan['SUM_JML_PENYALURAN'], errors='coerce').fillna(0)
df_gabungan['SUM_JML_DEBITUR'] = pd.to_numeric(df_gabungan['SUM_JML_DEBITUR'], errors='coerce').fillna(0)

# Baseline Sektoral
baseline_sektoral = df_gabungan.groupby(['NAMA_SEKTOR', 'PROGRAM']).agg(
    TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum'),
    TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum')
).reset_index()

# Menghitung Cost per Debitur
baseline_sektoral['COST_PER_DEBITUR'] = np.where(
    baseline_sektoral['TOTAL_DEBITUR'] > 0,
    baseline_sektoral['TOTAL_PENYALURAN'] / baseline_sektoral['TOTAL_DEBITUR'],
    0
)

baseline_sektoral.to_csv(out_baseline_sektoral, index=False, sep=';')
print(f"   -> Tersimpan: {out_baseline_sektoral}")

#Baseline Regional
print("\n[3] Menghitung Koefisien Pengali Regional...")
baseline_regional = df_gabungan.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT', 'PROGRAM']).agg(
    TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum'),
    TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum')
).reset_index()

baseline_regional['COST_PER_DEBITUR'] = np.where(
    baseline_regional['TOTAL_DEBITUR'] > 0,
    baseline_regional['TOTAL_PENYALURAN'] / baseline_regional['TOTAL_DEBITUR'],
    0
)

baseline_regional.to_csv(out_baseline_regional, index=False, sep=';')
print(f"   -> Tersimpan: {out_baseline_regional}")

# Preview Simulasi
preview_sektor = baseline_sektoral[baseline_sektoral['NAMA_SEKTOR'].str.contains('PERTANIAN|PERDAGANGAN', case=False, na=False)]

for index, row in preview_sektor.iterrows():
    if row['TOTAL_DEBITUR'] > 1000:
        print(f"📍 {row['PROGRAM']} - {row['NAMA_SEKTOR']}")
        print(f"   > Biaya per 1 Debitur: Rp {row['COST_PER_DEBITUR']:,.0f}\n")
