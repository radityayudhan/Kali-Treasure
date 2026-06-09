import pandas as pd
import numpy as np
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Direktori Data
folder_data = 'data'
file_subsidi = os.path.join(folder_data, 'Subsidi_KUR_Kalimantan_Clean.csv')
file_birate = os.path.join(folder_data, 'data_birate.csv')
file_pdrb = os.path.join(folder_data, 'pdrb_tahunan_kalimantan.csv')
file_realisasi = os.path.join(folder_data, 'Realisasi_KUR_Kalimantan.csv')

WAKTU_AWAL = '2016-07-01'
WAKTU_AKHIR = '2026-12-01'
idx_full = pd.date_range(start=WAKTU_AWAL, end=WAKTU_AKHIR, freq='MS')


def siapkan_exog_birate():
    df_bi = pd.read_csv(file_birate, sep=';')
    df_bi['BI_RATE'] = df_bi['BI_RATE'].astype(str).str.replace(',', '.').astype(float)
    df_bi['TANGGAL'] = pd.to_datetime(df_bi['TAHUN'].astype(str) + '-' + df_bi['BULAN'].astype(str) + '-01')
    ts_bi = df_bi.groupby('TANGGAL')['BI_RATE'].mean().sort_index()
    return ts_bi.reindex(idx_full).bfill().ffill()

def siapkan_exog_pdrb(filter_kabkot, list_kabkot_provinsi=None):
    df_pdrb = pd.read_csv(file_pdrb, sep=';')
    df_pdrb['NILAI_PDRB'] = df_pdrb['NILAI_PDRB'].astype(str).str.replace(',', '.').astype(float)
    
    if filter_kabkot != 'SEMUA':
        df_pdrb = df_pdrb[df_pdrb['NAMA_KABKOT'].str.upper() == filter_kabkot.upper()]
    elif list_kabkot_provinsi is not None:
        df_pdrb = df_pdrb[df_pdrb['NAMA_KABKOT'].str.upper().isin(list_kabkot_provinsi)]
        
    df_agg = df_pdrb.groupby('TAHUN')['NILAI_PDRB'].sum().reset_index()
    df_agg['TANGGAL'] = pd.to_datetime(df_agg['TAHUN'].astype(str) + '-07-01')
    
    ts_pdrb = df_agg.set_index('TANGGAL')['NILAI_PDRB'].sort_index()
    ts_pdrb = ts_pdrb.reindex(idx_full).interpolate(method='time').bfill().ffill()
    return np.log(ts_pdrb)

def siapkan_exog_realisasi(filter_provinsi='SEMUA', filter_kabkot='SEMUA'):
    """
    Mesin ini menarik data riil penyaluran murni KUR (tanpa UMi), 
    mengagregasi secara bulanan, dan melakukan forward-fill dari bulan terakhir (April 2024)
    hingga masa depan (2025-2026) untuk stabilitas ML.
    """
    if not os.path.exists(file_realisasi):
        return pd.Series(0, index=idx_full)
        
    df_kur = pd.read_csv(file_realisasi, sep=';', low_memory=False)
    
    # Hilangkan UMi dari variabel Eksogen
    if 'NAMA_SKEMA' in df_kur.columns:
        df_kur = df_kur[df_kur['NAMA_SKEMA'].astype(str).str.upper() != 'UMI']
    
    if 'NAMA_WILAYAH' in df_kur.columns:
        df_kur.rename(columns={'NAMA_WILAYAH': 'NAMA_KABKOT'}, inplace=True)
        
    df_kur['NAMA_PROVINSI'] = df_kur['NAMA_PROVINSI'].astype(str).str.strip().str.upper()
    df_kur['NAMA_KABKOT'] = df_kur['NAMA_KABKOT'].astype(str).str.strip().str.upper()
    
    df_kur['TAHUN'] = pd.to_numeric(df_kur['TAHUN'], errors='coerce').fillna(0).astype(int)
    df_kur['BULAN'] = pd.to_numeric(df_kur.get('BULAN', 1), errors='coerce').fillna(1).astype(int)
    
    if filter_provinsi != 'SEMUA':
        df_kur = df_kur[df_kur['NAMA_PROVINSI'] == filter_provinsi.upper()]
    if filter_kabkot != 'SEMUA':
        df_kur = df_kur[df_kur['NAMA_KABKOT'] == filter_kabkot.upper()]
        
    # Agregasi Total Penyaluran per BULAN
    df_agg = df_kur.groupby(['TAHUN', 'BULAN'])['SUM_JML_PENYALURAN'].sum().reset_index()
    
    # Memposisikan data pada tanggal 1 setiap bulannya
    df_agg['TANGGAL'] = pd.to_datetime(df_agg['TAHUN'].astype(str) + '-' + df_agg['BULAN'].astype(str).str.zfill(2) + '-01', errors='coerce')
    df_agg = df_agg.dropna(subset=['TANGGAL'])
    
    if df_agg.empty:
        return pd.Series(0, index=idx_full)
         
    ts_real = df_agg.set_index('TANGGAL')['SUM_JML_PENYALURAN'].sort_index()
    
    ts_real = ts_real.reindex(idx_full).interpolate(method='linear').ffill().bfill()
    
    return np.log(ts_real + 1)

def jalankan_sarimax(filter_provinsi='SEMUA', filter_kabkot='SEMUA'):
    df_subsidi = pd.read_csv(file_subsidi, sep=';', low_memory=False)
    
    list_kabkot_prov = None
    if filter_provinsi != 'SEMUA':
        list_kabkot_prov = df_subsidi[df_subsidi['NAMA_PROVINSI'].str.upper() == filter_provinsi.upper()]['NAMA_KABKOT'].str.upper().unique()
        df_subsidi = df_subsidi[df_subsidi['NAMA_PROVINSI'].str.upper() == filter_provinsi.upper()]
        
    if filter_kabkot != 'SEMUA':
        df_subsidi = df_subsidi[df_subsidi['NAMA_KABKOT'].str.upper() == filter_kabkot.upper()]
        
    df_sub_agg = df_subsidi.groupby(['TAHUN', 'BULAN'])['SUM_NILAI_SUBSIDI'].sum().reset_index()
    df_sub_agg['TANGGAL'] = pd.to_datetime(df_sub_agg['TAHUN'].astype(str) + '-' + df_sub_agg['BULAN'].astype(str) + '-01')
    
    ts_y = df_sub_agg.set_index('TANGGAL')['SUM_NILAI_SUBSIDI'].sort_index()
    ts_y_clean = ts_y[(ts_y.index >= WAKTU_AWAL) & (ts_y.index < '2022-08-01')]
    ts_y_clean = ts_y_clean.resample('MS').sum().replace(0, np.nan).interpolate(method='linear').fillna(0)
    
    ts_bi = siapkan_exog_birate()
    ts_pdrb = siapkan_exog_pdrb(filter_kabkot, list_kabkot_prov)
    ts_realisasi = siapkan_exog_realisasi(filter_provinsi, filter_kabkot)
    
    df_exog = pd.DataFrame({
        'BI_RATE': ts_bi, 
        'PDRB_LOG': ts_pdrb,
        'REALISASI_LOG': ts_realisasi
    })
    
    # Membelah matrix X untuk masa lalu (Training) dan masa depan (Forecasting)
    exog_train = df_exog.loc[ts_y_clean.index]
    tanggal_forecast = pd.date_range(start=ts_y_clean.index[-1] + pd.DateOffset(months=1), end=WAKTU_AKHIR, freq='MS')
    exog_future = df_exog.loc[tanggal_forecast]
    
    # Melatih 2 Skenario ML secara senyap (tanpa print di UI)
    model_opt = SARIMAX(endog=ts_y_clean, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
    pred_opt = model_opt.fit(disp=False).forecast(steps=len(tanggal_forecast), exog=exog_future).clip(lower=0)

    model_mod = SARIMAX(endog=ts_y_clean, exog=exog_train, order=(1, 0, 0), seasonal_order=(0, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
    pred_mod = model_mod.fit(disp=False).forecast(steps=len(tanggal_forecast), exog=exog_future).clip(lower=0)
    
    # Menyusun In-Memory DataFrame (df_final)
    df_hist = pd.DataFrame({'TANGGAL': ts_y_clean.index, 'TOTAL_TAGIHAN_SUBSIDI': ts_y_clean.values, 'TIPE': 'Historis', 'SKENARIO': 'Data Aktual (Realisasi)'})
    df_proj_opt = pd.DataFrame({'TANGGAL': tanggal_forecast, 'TOTAL_TAGIHAN_SUBSIDI': pred_opt.values, 'TIPE': 'Proyeksi', 'SKENARIO': 'Skenario Optimis (Kebutuhan Makro)'})
    df_proj_mod = pd.DataFrame({'TANGGAL': tanggal_forecast, 'TOTAL_TAGIHAN_SUBSIDI': pred_mod.values, 'TIPE': 'Proyeksi', 'SKENARIO': 'Skenario Moderat (Pagu Terkendali)'})
    
    df_final = pd.concat([df_hist, df_proj_opt, df_proj_mod], ignore_index=True)
    df_final['PERIODE'] = df_final['TANGGAL'].dt.strftime('%Y-%m')
    
    # Menyusun In-Memory Plot Figure (fig)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(ts_y_clean.index, ts_y_clean.values, color='#1f77b4', linewidth=2.5, label='Subsidi Historis')
    
    titik_akhir = pd.Series([ts_y_clean.values[-1]], index=[ts_y_clean.index[-1]])
    garis_opt = pd.concat([titik_akhir, pred_opt])
    garis_mod = pd.concat([titik_akhir, pred_mod])
    
    ax.plot(garis_opt.index, garis_opt.values, color='#ff7f0e', linewidth=2.5, linestyle='--', label='Skenario Optimis')
    ax.plot(garis_mod.index, garis_mod.values, color='#2ca02c', linewidth=2.5, linestyle='-', label='Skenario Moderat (Tanpa Tren Bulanan)')

    judul = f"Proyeksi Subsidi KUR (2023-2026) : Model Sarimax\nWilayah: {filter_provinsi} - {filter_kabkot}"
    ax.set_title(judul, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Periode (Tahun)", fontsize=12, labelpad=10)
    ax.set_ylabel("Total Tagihan (Rupiah)", fontsize=12, labelpad=10)
    
    def format_rupiah(x, pos):
        if x >= 1e12: return f'Rp {x*1e-12:.1f} T'
        elif x >= 1e9: return f'Rp {x*1e-9:.0f} M'
        return f'Rp {x:,.0f}'

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_rupiah))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.axvspan(ts_y_clean.index[-1], pd.to_datetime(WAKTU_AKHIR), color='grey', alpha=0.08)
    ax.legend(loc='upper left', fontsize=12, shadow=True)
    
    plt.tight_layout()
    
    return df_final, fig

if __name__ == "__main__":
    print("Menjalankan uji coba in-memory dengan Trinitas Exogen...")
    df_hasil, fig_hasil = jalankan_sarimax('KALIMANTAN UTARA', 'SEMUA')
    print("Berhasil menghasilkan DataFrame dengan shape:", df_hasil.shape)
    print("Tipe objek grafik:", type(fig_hasil))
    # plt.show() # Jika ingin melihat grafiknya saat uji coba manual