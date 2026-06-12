import pandas as pd
import os

def validasi_dataset(file_path, delimiter, jenis_data):
    print(f"\n{'='*50}")
    print(f"🚀 MEMULAI VALIDASI: {file_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} tidak ditemukan! Pastikan nama dan foldernya benar.")
        return

    df = pd.read_csv(file_path, sep=delimiter, low_memory=False)
    total_baris = len(df)
    
    print(f"DIMENSI DATA")
    print(f"   - Jumlah Baris : {total_baris:,}")
    print(f"   - Jumlah Kolom : {df.shape[1]}")
    
    # Cek Baris Duplikat
    jumlah_duplikat = df.duplicated().sum()
    if jumlah_duplikat > 0:
        persentase_duplikat = (jumlah_duplikat / total_baris) * 100
        print(f"   - Terdapat {jumlah_duplikat:,} baris duplikat identik ({persentase_duplikat:.2f}% dari total data).")
    else:
        print(f"   - STATUS: Bersih (0 baris duplikat).")

    # Validasi Provinsi
    if 'NAMA_PROVINSI' in df.columns:
        prov_unik = df['NAMA_PROVINSI'].astype(str).str.strip().str.upper().unique()
        print(f"   - Provinsi terdeteksi: {', '.join(prov_unik)}")
        
        non_kalimantan = [p for p in prov_unik if 'KALIMANTAN' not in p]
        if len(non_kalimantan) > 0:
            print(f"   - Terdapat provinsi di luar Kalimantan: {non_kalimantan}")
        else:
            print(f"   - STATUS: Valid! 100% Data Regional Kalimantan.")
    else:
        print("   - Kolom 'NAMA_PROVINSI' tidak ditemukan.")

    # Cek Missing Vallues
    
    # Menghitung total baris yang memiliki setidaknya 1 kolom kosong
    baris_ada_kosong = df.isnull().any(axis=1).sum()
    
    if baris_ada_kosong > 0:
        persentase_kosong = (baris_ada_kosong / total_baris) * 100
        print(f"   - ⚠️ Terdapat {baris_ada_kosong:,} baris yang memiliki sel kosong ({persentase_kosong:.2f}% dari total baris).")
        print("   - Rincian kolom beserta lokasi baris (Index Pandas dimulai dari 0):")
        
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        
        for col, count in missing_cols.items():
            list_baris_kosong = df[df[col].isnull()].index.tolist()
            batas_tampil = 10
            contoh_baris = ", ".join(map(str, list_baris_kosong[:batas_tampil]))
            
            if count > batas_tampil:
                print(f"     > {col}: {count:,} sel kosong | (Di baris index: [{contoh_baris}, dan {count - batas_tampil} lainnya...])")
            else:
                print(f"     > {col}: {count:,} sel kosong | (Di baris index: [{contoh_baris}])")
    else:
        print(f"   -STATUS: Sangat Bersih! Tidak ada sel data yang kosong di seluruh kolom.")

if __name__ == "__main__":
    
    folder_data = 'data'
    
    # Validasi Realisasi KUR 
    validasi_dataset(
        file_path=os.path.join(folder_data, 'Realisasi_KUR_Kalimantan.csv'), 
        delimiter=';', 
        jenis_data='Realisasi'
    )
    
    # Validasi Subsidi KUR 
    validasi_dataset(
        file_path=os.path.join(folder_data, 'Subsidi_KUR_Kalimantan.csv'), 
        delimiter=';', 
        jenis_data='Subsidi'
    )
    
    # Validasi Realisasi UMi 
    validasi_dataset(
        file_path=os.path.join(folder_data, 'Realisasi_UMI_Kalimantan.csv'), 
        delimiter=';', 
        jenis_data='Realisasi'
    )