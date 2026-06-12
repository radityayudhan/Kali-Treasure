import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

folder_data = 'data'
file_realisasi = os.path.join(folder_data, 'Realisasi_KUR_Kalimantan.csv')
file_subsidi = os.path.join(folder_data, 'Subsidi_KUR_Kalimantan.csv')

# Output files
out_ews = os.path.join(folder_data, 'agregat_ews_tab3.csv')
out_forecast = os.path.join(folder_data, 'agregat_forecast_tab3.csv')
out_plot_img = 'Anomali_Subsidi_Kalimantan.png'

df_realisasi = pd.read_csv(file_realisasi, sep=';', low_memory=False)
df_subsidi = pd.read_csv(file_subsidi, sep=';', low_memory=False)

def standarisasi_kolom(df):
    kolom_map = {}
    for col in df.columns:
        col_up = col.upper()
        if col_up in ['NAMA_KABKOT', 'NAMA_WILAYAH']:
            kolom_map[col] = 'NAMA_KABKOT'
        elif col_up in ['NAMA_PROVINSI', 'PROVINSI']:
            kolom_map[col] = 'NAMA_PROVINSI'
    if kolom_map:
        df.rename(columns=kolom_map, inplace=True)
    return df

df_realisasi = standarisasi_kolom(df_realisasi)
df_subsidi = standarisasi_kolom(df_subsidi)

# Membersihkan teks untuk penggabungan
for df in [df_realisasi, df_subsidi]:
    if 'NAMA_PROVINSI' in df.columns:
        df['NAMA_PROVINSI'] = df['NAMA_PROVINSI'].astype(str).str.strip().str.upper()
    if 'NAMA_KABKOT' in df.columns:
        df['NAMA_KABKOT'] = df['NAMA_KABKOT'].astype(str).str.strip().str.upper()


# AGREGASI EWS (RASIO SUBSIDI PER DEBITUR)

realisasi_agg = df_realisasi.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT']).agg(
    TOTAL_DEBITUR=('SUM_JML_DEBITUR', 'sum')
).reset_index()

subsidi_agg = df_subsidi.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT']).agg(
    TOTAL_SUBSIDI=('SUM_NILAI_SUBSIDI', 'sum')
).reset_index()

df_ews = pd.merge(realisasi_agg, subsidi_agg, on=['NAMA_PROVINSI', 'NAMA_KABKOT'], how='inner')
df_ews['TOTAL_DEBITUR'] = pd.to_numeric(df_ews['TOTAL_DEBITUR'], errors='coerce').fillna(0)
df_ews['TOTAL_SUBSIDI'] = pd.to_numeric(df_ews['TOTAL_SUBSIDI'], errors='coerce').fillna(0)

df_ews['BIAYA_SUBSIDI_PER_DEBITUR'] = np.where(
    df_ews['TOTAL_DEBITUR'] > 0, 
    df_ews['TOTAL_SUBSIDI'] / df_ews['TOTAL_DEBITUR'], 
    0
)

df_ews.to_csv(out_ews, index=False, sep=';')
print(f"   -> File EWS tersimpan: {out_ews}")

# AGREGASI TIME-SERIES FORECASTING

df_subsidi['TAHUN'] = pd.to_numeric(df_subsidi['TAHUN'], errors='coerce')
df_subsidi['BULAN'] = pd.to_numeric(df_subsidi['BULAN'], errors='coerce')
df_subsidi = df_subsidi.dropna(subset=['TAHUN', 'BULAN'])

df_forecast = df_subsidi.groupby(['TAHUN', 'BULAN']).agg(
    TOTAL_TAGIHAN_SUBSIDI=('SUM_NILAI_SUBSIDI', 'sum')
).reset_index()

df_forecast = df_forecast[(df_forecast['TAHUN'] >= 2015) & (df_forecast['TAHUN'] <= 2026)]
df_forecast = df_forecast.sort_values(by=['TAHUN', 'BULAN'])
df_forecast['PERIODE'] = df_forecast['TAHUN'].astype(int).astype(str) + '-' + df_forecast['BULAN'].astype(int).astype(str).str.zfill(2)

df_forecast.to_csv(out_forecast, index=False, sep=';')
print(f"   -> File Time-Series mentah tersimpan: {out_forecast}")

print("\n👀 BUKTI ANOMALI PELAPORAN (5 Bulan Terakhir Tahun 2022):")
df_2022 = df_forecast[df_forecast['TAHUN'] == 2022]
for index, row in df_2022.tail(5).iterrows():
    print(f"Periode {row['PERIODE']} : Realisasi Tagihan Rp {row['TOTAL_TAGIHAN_SUBSIDI']:,.0f}")

# VISUALISASI ANOMALI

df_plot = df_forecast.copy()
df_plot['TANGGAL'] = pd.to_datetime(df_plot['PERIODE'] + '-01')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 7))

# Plot Garis Historis Utama
plt.plot(df_plot['TANGGAL'], df_plot['TOTAL_TAGIHAN_SUBSIDI'], 
         color='#1f77b4', linewidth=2.5, marker='o', markersize=4, label='Tagihan Historis')

tgl_mulai_anomali = pd.to_datetime('2022-08-01')
tgl_akhir_anomali = pd.to_datetime('2022-10-01')
plt.axvspan(tgl_mulai_anomali, tgl_akhir_anomali, color='red', alpha=0.15, label='Area Anomali (Administrative Lag)')

nilai_puncak = df_plot[df_plot['TANGGAL'] == pd.to_datetime('2022-07-01')]['TOTAL_TAGIHAN_SUBSIDI'].values
if len(nilai_puncak) > 0:
    plt.annotate(
        'Puncak Normal\n(Juli 2022)', 
        xy=(pd.to_datetime('2022-07-01'), nilai_puncak[0]), 
        xytext=(pd.to_datetime('2021-12-01'), nilai_puncak[0] + 15_000_000_000),
        arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
        fontsize=11, fontweight='bold', color='green'
    )

nilai_anjlok = df_plot[df_plot['TANGGAL'] == pd.to_datetime('2022-10-01')]['TOTAL_TAGIHAN_SUBSIDI'].values
if len(nilai_anjlok) > 0:
    plt.annotate(
        'Pelaporan Anjlok\n(Agustus-Okt 2022)', 
        xy=(pd.to_datetime('2022-10-01'), nilai_anjlok[0]), 
        xytext=(pd.to_datetime('2020-05-01'), nilai_anjlok[0] + 60_000_000_000),
        arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
        fontsize=11, fontweight='bold', color='red'
    )

plt.title("Evaluasi Integritas Data: Identifikasi Keterlambatan Pelaporan Subsidi KUR", 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Periode (Tahun)", fontsize=12, labelpad=10)
plt.ylabel("Total Tagihan (Rupiah)", fontsize=12, labelpad=10)

def format_rupiah(x, pos):
    if x >= 1e12: return f'Rp {x*1e-12:.1f} T'
    elif x >= 1e9: return f'Rp {x*1e-9:.0f} M'
    return f'Rp {x:,.0f}'

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_rupiah))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=0)
plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(out_plot_img, dpi=300, bbox_inches='tight')
print(f"   -> Grafik anomali berhasil disimpan: {out_plot_img}")