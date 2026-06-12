import pandas as pd
import os

folder_data = 'data'
file_kur = os.path.join(folder_data, 'Realisasi_KUR_Kalimantan.csv')
file_umi = os.path.join(folder_data, 'Realisasi_UMI_Kalimantan.csv')
file_output = os.path.join(folder_data, 'agregat_spasial_tab1.csv')

print("\n[1] Membaca dataset...")
df_kur = pd.read_csv(file_kur, sep=';', low_memory=False)
df_umi = pd.read_csv(file_umi, sep=';', low_memory=False)

print("    Menyelaraskan nomenklatur kolom wilayah...")
if 'NAMA_WILAYAH' in df_kur.columns:
    df_kur.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT', 'KODE_WILAYAH': 'KODE_KABKOT'}, inplace=True)

if 'NAMA_WILAYAH' in df_umi.columns:
    df_umi.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT', 'KODE_WILAYAH': 'KODE_KABKOT'}, inplace=True)

print("\n--- PROFIL REALISASI KUR ---")
print(f"Jumlah Baris : {len(df_kur):,}")
print(f"Jumlah Kolom : {len(df_kur.columns)}")

print("\n--- PROFIL REALISASI UMI ---")
print(f"Jumlah Baris : {len(df_umi):,}")
print(f"Jumlah Kolom : {len(df_umi.columns)}")

# PENGECEKAN KOLOM (MISMATCH CHECK)
print("\n[2] Mengecek Keselarasan Kolom...")
kolom_kur = set(df_kur.columns)
kolom_umi = set(df_umi.columns)

kolom_irisan = list(kolom_kur.intersection(kolom_umi))
print(f"Terdapat {len(kolom_irisan)} kolom yang sama (Match).")

if 'NAMA_KABKOT' not in kolom_irisan:
    print("ERROR FATAL: Kolom 'NAMA_KABKOT' masih tidak ditemukan di irisan!")
else:
    print("Kolom 'NAMA_KABKOT' berhasil masuk dalam irisan data.")

print("\n[3] Menambahkan program dan menggabungkan data...")

df_kur_bersih = df_kur[kolom_irisan].copy()
df_umi_bersih = df_umi[kolom_irisan].copy()

# Menambahkan program
df_kur_bersih['PROGRAM'] = 'KUR'
df_umi_bersih['PROGRAM'] = 'UMI'

# Melakukan Union
df_gabungan = pd.concat([df_kur_bersih, df_umi_bersih], ignore_index=True)
print(f"Penggabungan berhasil! Total baris gabungan: {len(df_gabungan):,}")


# AGREGASI SPASIAL (EDA TAB 1)
print("\n[4] Melakukan Agregasi Spasial Regional Kalimantan...")

# Cleansing nama wilayah agar seragam saat di-grouping (menghindari duplikasi karena spasi)
df_gabungan['NAMA_PROVINSI'] = df_gabungan['NAMA_PROVINSI'].astype(str).str.strip().str.upper()
df_gabungan['NAMA_KABKOT'] = df_gabungan['NAMA_KABKOT'].astype(str).str.strip().str.upper()

# Proses Agregasi ke level Kabupaten/Kota se-Kalimantan
agregat_spasial = df_gabungan.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT', 'PROGRAM']).agg(
    TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum'),
    TOTAL_PENYALURAN=('SUM_JML_PENYALURAN', 'sum')
).reset_index()

agregat_spasial.to_csv(file_output, index=False, sep=';')
print(f"File agregat siap: {file_output}")

# VALIDASI & PREVIEW
print("\n--- PREVIEW: TOP 5 WILAYAH INKLUSI TERTINGGI SE-KALIMANTAN ---")
top_wilayah = agregat_spasial.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT'])['TOTAL_DEBITUR'].sum().reset_index()
top_wilayah = top_wilayah.sort_values(by='TOTAL_DEBITUR', ascending=False).head(5)

for index, row in top_wilayah.iterrows():
    print(f"\n📍 {row['NAMA_PROVINSI']} - {row['NAMA_KABKOT']}")
    
    rincian = agregat_spasial[agregat_spasial['NAMA_KABKOT'] == row['NAMA_KABKOT']]
    total_debitur_gabungan = row['TOTAL_DEBITUR']
    
    for _, r in rincian.iterrows():
        porsi = (r['TOTAL_DEBITUR'] / total_debitur_gabungan) * 100 if total_debitur_gabungan > 0 else 0
        print(f"   > {r['PROGRAM']:<3}: {r['TOTAL_DEBITUR']:>10,.0f} orang ({porsi:.1f}%) | Rp {r['TOTAL_PENYALURAN']:,.0f}")