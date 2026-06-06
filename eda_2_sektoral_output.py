import pandas as pd
import os

print("=== MEMULAI EDA TAB 2: TRANSFORMASI SEKTORAL & EFISIENSI JANGKAUAN ===")

folder_data = 'data'
file_kur = os.path.join(folder_data, 'Realisasi_KUR_Kalimantan.csv')
file_umi = os.path.join(folder_data, 'Realisasi_UMI_Kalimantan.csv')

# Output files
out_sektoral = os.path.join(folder_data, 'agregat_sektoral_tab2.csv')
out_efisiensi = os.path.join(folder_data, 'agregat_efisiensi_tab2.csv')

# ==========================================
# 1. MEMBACA & STANDARISASI KOLOM DINAMIS
# ==========================================
print("\n[1] Membaca dan menstandarisasi dataset...")
df_kur = pd.read_csv(file_kur, sep=';', low_memory=False)
df_umi = pd.read_csv(file_umi, sep=';', low_memory=False)

def standarisasi_kolom(df):
    kolom_map = {}
    for col in df.columns:
        col_up = col.upper()
        if col_up in ['TAHUN', 'TAHUN_PENYALURAN', 'THN']:
            kolom_map[col] = 'TAHUN'
        elif col_up in ['NAMA_SEKTOR', 'SEKTOR_USAHA', 'SEKTOR_EKONOMI']:
            kolom_map[col] = 'NAMA_SEKTOR'
        elif col_up in ['NAMA_KABKOT', 'NAMA_WILAYAH']:
            kolom_map[col] = 'NAMA_KABKOT'
        elif col_up in ['NAMA_PROVINSI', 'PROVINSI']:
            kolom_map[col] = 'NAMA_PROVINSI'
            
    if kolom_map:
        df.rename(columns=kolom_map, inplace=True)
    return df

df_kur = standarisasi_kolom(df_kur)
df_umi = standarisasi_kolom(df_umi)

# Menyuntikkan identitas program
df_kur['PROGRAM'] = 'KUR'
df_umi['PROGRAM'] = 'UMI'

# ==========================================
# 2. PENGGABUNGAN DATA (KOLOM IRISAN)
# ==========================================
kolom_wajib = ['TAHUN', 'NAMA_PROVINSI', 'NAMA_KABKOT', 'NAMA_SEKTOR', 
               'SUM_JML_DEBITUR', 'SUM_JML_PENYALURAN', 'PROGRAM']

# Memastikan semua kolom wajib ada di kedua dataframe
for col in kolom_wajib:
    if col not in df_kur.columns: df_kur[col] = 'TIDAK DIKETAHUI'
    if col not in df_umi.columns: df_umi[col] = 'TIDAK DIKETAHUI'

df_gabungan = pd.concat([df_kur[kolom_wajib], df_umi[kolom_wajib]], ignore_index=True)

# Cleansing tipe data numerik
df_gabungan['SUM_JML_PENYALURAN'] = pd.to_numeric(df_gabungan['SUM_JML_PENYALURAN'], errors='coerce').fillna(0)
df_gabungan['SUM_JML_DEBITUR'] = pd.to_numeric(df_gabungan['SUM_JML_DEBITUR'], errors='coerce').fillna(0)
df_gabungan['NAMA_SEKTOR'] = df_gabungan['NAMA_SEKTOR'].astype(str).str.strip().str.upper()

print(f"✅ Data berhasil digabung! Total baris: {len(df_gabungan):,}")

# ==========================================
# 3. AGREGASI A: TREN SEKTORAL (TIME-SERIES)
# ==========================================
print("\n[2] Memproses Agregasi Tren Sektoral (Time-Series)...")
agregat_sektoral = df_gabungan.groupby(['TAHUN', 'NAMA_PROVINSI', 'NAMA_SEKTOR', 'PROGRAM']).agg(
    TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum'),
    TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum')
).reset_index()

# Filter untuk membuang anomali tahun
agregat_sektoral = agregat_sektoral[(agregat_sektoral['TAHUN'] >= 2015) & (agregat_sektoral['TAHUN'] <= 2026)]

agregat_sektoral.to_csv(out_sektoral, index=False, sep=';')
print(f"   -> Tersimpan: {out_sektoral}")

# --- MENAMPILKAN INFO DAN HEAD UNTUK AGREGAT SEKTORAL ---
print("\n🔍 INFO DATA: AGREGAT SEKTORAL")
print("-" * 50)
agregat_sektoral.info()
print("\n👀 HEAD DATA: AGREGAT SEKTORAL (5 Baris Pertama)")
print("-" * 50)
print(agregat_sektoral.head().to_string(index=False))

# ==========================================
# 4. AGREGASI B: EFISIENSI JANGKAUAN (SCATTER PLOT)
# ==========================================
print("\n[3] Memproses Agregasi Efisiensi Jangkauan per Wilayah...")
agregat_efisiensi = df_gabungan.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT', 'PROGRAM']).agg(
    TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum'),
    TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum')
).reset_index()

# Menghitung Rata-rata Plafon per Debitur
agregat_efisiensi['AVG_PLAFON_PER_DEBITUR'] = (agregat_efisiensi['TOTAL_PENYALURAN'] / agregat_efisiensi['TOTAL_DEBITUR']).fillna(0)

agregat_efisiensi.to_csv(out_efisiensi, index=False, sep=';')
print(f"   -> Tersimpan: {out_efisiensi}")

# --- MENAMPILKAN INFO DAN HEAD UNTUK AGREGAT EFISIENSI ---
print("\n🔍 INFO DATA: AGREGAT EFISIENSI")
print("-" * 50)
agregat_efisiensi.info()
print("\n👀 HEAD DATA: AGREGAT EFISIENSI (5 Baris Pertama)")
print("-" * 50)
print(agregat_efisiensi.head().to_string(index=False))

# ==========================================
# 5. PREVIEW INSIGHT
# ==========================================
print("\n--- PREVIEW INSIGHT: SEKTOR PERTANIAN VS PERDAGANGAN SE-KALIMANTAN ---")
df_pertanian = agregat_sektoral[agregat_sektoral['NAMA_SEKTOR'].str.contains('PERTANIAN|KEHUTANAN|PERIKANAN', na=False)]
df_perdagangan = agregat_sektoral[agregat_sektoral['NAMA_SEKTOR'].str.contains('PERDAGANGAN', na=False)]

total_pertanian = df_pertanian['TOTAL_PENYALURAN'].sum()
total_perdagangan = df_perdagangan['TOTAL_PENYALURAN'].sum()

print(f"Total Pembiayaan Sektor Pertanian   : Rp {total_pertanian:,.0f}")
print(f"Total Pembiayaan Sektor Perdagangan : Rp {total_perdagangan:,.0f}")

if total_perdagangan > total_pertanian:
    print("📈 Catatan Analisis: Sektor perdagangan masih mendominasi. Ada ruang besar untuk mendorong kebijakan afirmatif menuju hilirisasi dan ketahanan pangan di Regional Kalimantan.")

print("\n✅ EDA TAB 2 SELESAI!")