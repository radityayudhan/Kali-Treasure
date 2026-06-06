import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# Direktori Data
folder_data = 'data'
file_mentah_bersih = os.path.join(folder_data, 'Subsidi_KUR_Kalimantan_Clean.csv')
file_agregat_geo = os.path.join(folder_data, 'agregat_forecast_geo.csv')

# ==========================================
# 1. PERSIAPAN DATA BERBASIS GEOGRAFIS
# ==========================================
def siapkan_data_geografis():
    df_subsidi = pd.read_csv(file_mentah_bersih, sep=';', low_memory=False)
    
    # Standarisasi kolom
    kolom_map = {}
    for col in df_subsidi.columns:
        if col.upper() in ['NAMA_KABKOT', 'NAMA_WILAYAH']: kolom_map[col] = 'NAMA_KABKOT'
        elif col.upper() in ['NAMA_PROVINSI', 'PROVINSI']: kolom_map[col] = 'NAMA_PROVINSI'
    df_subsidi.rename(columns=kolom_map, inplace=True)
    
    # Agregasi tetapi tetap mempertahankan Provinsi dan Kab/Kota
    df_geo = df_subsidi.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT', 'TAHUN', 'BULAN']).agg(
        TOTAL_TAGIHAN=('SUM_NILAI_SUBSIDI', 'sum')
    ).reset_index()
    
    df_geo = df_geo[(df_geo['TAHUN'] >= 2015) & (df_geo['TAHUN'] <= 2026)]
    
    # Menyimpan file agregat geografi agar Streamlit tidak perlu memproses ulang data mentah yang berat
    if not os.path.exists(folder_data):
        os.makedirs(folder_data)
    df_geo.to_csv(file_agregat_geo, index=False, sep=';')
    return df_geo

# ==========================================
# 2. FUNGSI ML "ON-THE-FLY" (UNTUK STREAMLIT)
# ==========================================
def jalankan_prediksi_wilayah(df_geo, filter_provinsi='SEMUA', filter_kabkot='SEMUA'):
    """
    Fungsi ini dipanggil oleh Streamlit.
    Mengembalikan 2 objek In-Memory: (DataFrame Data Tabel, Matplotlib Figure)
    """
    # a. Terapkan Filter Geografis
    df_filtered = df_geo.copy()
    if filter_provinsi != 'SEMUA':
        df_filtered = df_filtered[df_filtered['NAMA_PROVINSI'].str.upper() == filter_provinsi.upper()]
    if filter_kabkot != 'SEMUA':
        df_filtered = df_filtered[df_filtered['NAMA_KABKOT'].str.upper() == filter_kabkot.upper()]
        
    # Validasi jika data kosong
    if len(df_filtered) == 0:
        return None, None
        
    # b. Agregasi Berdasarkan Waktu Saja (Setelah difilter)
    df_agg = df_filtered.groupby(['TAHUN', 'BULAN'])['TOTAL_TAGIHAN'].sum().reset_index()
    df_agg['PERIODE'] = df_agg['TAHUN'].astype(int).astype(str) + '-' + df_agg['BULAN'].astype(int).astype(str).str.zfill(2)
    df_agg['TANGGAL'] = pd.to_datetime(df_agg['PERIODE'] + '-01')
    
    ts_data = df_agg.set_index('TANGGAL')['TOTAL_TAGIHAN'].sort_index()

    # c. Pemotongan Waktu Ekstrem (Mutlak sebelum Agustus 2022 akibat Administrative Lag)
    ts_data_clean = ts_data[ts_data.index < '2022-08-01']
    ts_data_clean = ts_data_clean.resample('MS').sum().replace(0, np.nan).interpolate(method='linear').fillna(0)

    # d. Pelatihan Multi-Skenario secara In-Memory
    model_optimis = ExponentialSmoothing(
        ts_data_clean, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True, initialization_method="estimated"
    ).fit(optimized=True)

    model_moderat = ExponentialSmoothing(
        ts_data_clean, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True, initialization_method="estimated"
    ).fit(optimized=True, damping_trend=0.85)

    # e. Proyeksi 53 Bulan ke depan (Hingga Akhir 2026)
    pred_optimis = model_optimis.forecast(53).clip(lower=0)
    pred_moderat = model_moderat.forecast(53).clip(lower=0)
    tanggal_forecast = pd.date_range(start=ts_data_clean.index[-1] + pd.DateOffset(months=1), periods=53, freq='MS')

    # f. Menyusun In-Memory DataFrame untuk Tabel Streamlit
    df_hist = pd.DataFrame({'TANGGAL': ts_data_clean.index, 'TOTAL_TAGIHAN_SUBSIDI': ts_data_clean.values, 'TIPE': 'Historis', 'SKENARIO': 'Data Aktual (Realisasi)'})
    df_proj_opt = pd.DataFrame({'TANGGAL': tanggal_forecast, 'TOTAL_TAGIHAN_SUBSIDI': pred_optimis.values, 'TIPE': 'Proyeksi', 'SKENARIO': 'Skenario Optimis (Tanpa Batas Pagu)'})
    df_proj_mod = pd.DataFrame({'TANGGAL': tanggal_forecast, 'TOTAL_TAGIHAN_SUBSIDI': pred_moderat.values, 'TIPE': 'Proyeksi', 'SKENARIO': 'Skenario Moderat (Pagu Terkendali)'})
    
    df_final = pd.concat([df_hist, df_proj_opt, df_proj_mod], ignore_index=True)
    df_final['PERIODE'] = df_final['TANGGAL'].dt.strftime('%Y-%m')

    # g. Visualisasi Dinamis (In-Memory Figure)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(ts_data_clean.index, ts_data_clean.values, color='#1f77b4', linewidth=2.5, label='Subsidi Historis')
    
    # Garis sambung
    titik_akhir = pd.Series([ts_data_clean.values[-1]], index=[ts_data_clean.index[-1]])
    garis_opt = pd.concat([titik_akhir, pred_optimis])
    garis_mod = pd.concat([titik_akhir, pred_moderat])
    
    ax.plot(garis_opt.index, garis_opt.values, color='#ff7f0e', linewidth=2.5, linestyle='--', label='Skenario Optimis')
    ax.plot(garis_mod.index, garis_mod.values, color='#2ca02c', linewidth=2.5, linestyle='-', label='Skenario Moderat (Damping Trend 0.85)')

    judul_plot = f"Proyeksi Subsidi KUR (2023-2026): Model Holt-Winters\nWilayah: {filter_provinsi} - {filter_kabkot}"
    ax.set_title(judul_plot, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Total Tagihan (Rupiah)", fontsize=12)
    ax.set_xlabel("Periode (Tahun)", fontsize=12)
    
    def format_rupiah(x, pos):
        if x >= 1e12: return f'Rp {x*1e-12:.1f} T'
        elif x >= 1e9: return f'Rp {x*1e-9:.0f} M'
        return f'Rp {x:,.0f}'

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_rupiah))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='upper left', fontsize=11, shadow=True)
    
    plt.tight_layout()
    
    # Mengembalikan objek ke aplikasi pemanggil (Streamlit)
    return df_final, fig


# ==========================================
# BLOK UJI COBA MANUAL (Tidak akan jalan di Streamlit)
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(file_agregat_geo):
        print("Menyiapkan file cache geografi...")
        df_master = siapkan_data_geografis()
    else:
        df_master = pd.read_csv(file_agregat_geo, sep=';', low_memory=False)

    print("Menjalankan uji coba In-Memory Holt-Winters...")
    df_hasil, fig_hasil = jalankan_prediksi_wilayah(df_master, filter_provinsi='KALIMANTAN UTARA', filter_kabkot='SEMUA')
    
    if df_hasil is not None:
        print("✅ Berhasil! DataFrame terbentuk dengan jumlah baris:", df_hasil.shape[0])
        print("✅ Berhasil! Objek grafik bertipe:", type(fig_hasil))
        # plt.show() # Uncomment jika ingin melihat grafik saat uji coba di terminal
    else:
        print("❌ Gagal, data tidak ditemukan.")