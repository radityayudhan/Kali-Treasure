import pandas as pd
import os

folder_data = 'data'
file_input = os.path.join(folder_data, 'Subsidi_KUR_Kalimantan.csv')
file_output = os.path.join(folder_data, 'Subsidi_KUR_Kalimantan_Clean.csv')

df = pd.read_csv(file_input, sep=';', low_memory=False)

# Mengisi nilai kosong (Imputasi) pada kolom Teks
kolom_teks_kosong = ['NAMA_KABKOT', 'NAMA_SEKTOR', 'NAMA_SUBSEKTOR']
for col in kolom_teks_kosong:
    if col in df.columns:
        df[col] = df[col].fillna('TIDAK DIKETAHUI')

# Mengisi nilai kosong (Imputasi) pada kolom Kode 
# (Diisi dengan '9999')
kolom_kode_kosong = ['KODE_KABKOT', 'KODE_SEKTOR', 'KODE_SUBSEKTOR']
for col in kolom_kode_kosong:
    if col in df.columns:
        df[col] = df[col].fillna('9999')

# Menyimpan file bersih
df.to_csv(file_output, index=False, sep=';')

print(f"✅ Cleansing Selesai!")
print(f"Data bersih telah disimpan sebagai: {file_output}")