import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

@st.cache_data(show_spinner=False)
def jalankan_kmeans_clustering(df_master, filter_provinsi="SEMUA", mode_lensa="Indikator dan Sektor Penyaluran"):
    df_ml = df_master.copy()
    
    # Cleansing area NV dan Provinsi
    df_ml = df_ml[df_ml['NAMA_KABKOT'].astype(str) != 'NV']
    df_ml = df_ml[~df_ml['NAMA_KABKOT'].astype(str).str.startswith('KALIMANTAN')]
    # ------------------------------------

    # Agregasi wilayah untuk mendapatkan fitur-fitur utama
    df_agg = df_ml.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT']).agg(
        TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
        TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum'),
        PR_DEBITUR=('DEBITUR_PEREMPUAN', 'sum'),
        SD_DEBITUR=('PENDIDIKAN_SD_SMP', 'sum'),
        JUMLAH_PENDUDUK=('JUMLAH_PENDUDUK', 'max') 
    ).reset_index()
    
    # Agregasi Sektoral
    df_sektor = df_ml.groupby(['NAMA_KABKOT', 'SEKTOR_USAHA'])['TOTAL_PENYALURAN'].sum().unstack(fill_value=0).reset_index()
    df_agg = pd.merge(df_agg, df_sektor, on='NAMA_KABKOT', how='left')
    
    # Feature Engineering
    df_agg['TICKET_SIZE'] = np.where(df_agg['TOTAL_DEBITUR'] > 0, df_agg['TOTAL_PENYALURAN'] / df_agg['TOTAL_DEBITUR'], 0)
    df_agg['RASIO_PENETRASI'] = np.where(df_agg['JUMLAH_PENDUDUK'] > 0, (df_agg['TOTAL_DEBITUR'] / df_agg['JUMLAH_PENDUDUK']) * 100, 0)
    df_agg['PCT_PEREMPUAN'] = np.where(df_agg['TOTAL_DEBITUR'] > 0, (df_agg['PR_DEBITUR'] / df_agg['TOTAL_DEBITUR']) * 100, 0)
    df_agg['PCT_SD_SMP'] = np.where(df_agg['TOTAL_DEBITUR'] > 0, (df_agg['SD_DEBITUR'] / df_agg['TOTAL_DEBITUR']) * 100, 0)
    
    kolom_perdagangan = [col for col in df_agg.columns if 'PERDAGANGAN' in str(col).upper()]
    kolom_pertanian = [col for col in df_agg.columns if 'PERTANIAN' in str(col).upper() or 'KEHUTANAN' in str(col).upper()]
    val_perdagangan = df_agg[kolom_perdagangan].sum(axis=1) if kolom_perdagangan else 0
    val_pertanian = df_agg[kolom_pertanian].sum(axis=1) if kolom_pertanian else 0
    
    df_agg['PCT_PERDAGANGAN'] = np.where(df_agg['TOTAL_PENYALURAN'] > 0, (val_perdagangan / df_agg['TOTAL_PENYALURAN']) * 100, 0)
    df_agg['PCT_PERTANIAN'] = np.where(df_agg['TOTAL_PENYALURAN'] > 0, (val_pertanian / df_agg['TOTAL_PENYALURAN']) * 100, 0)
    df_agg['TICKET_SIZE_JUTA'] = df_agg['TICKET_SIZE'] / 1e6

    if mode_lensa == "Indikator dan Sektor Penyaluran":
        fitur_utama = ['TICKET_SIZE', 'RASIO_PENETRASI', 'PCT_PERDAGANGAN', 'PCT_PERTANIAN']
        var_x = 'PCT_PERTANIAN'
        var_y = 'TICKET_SIZE_JUTA'
        var_size = 'PCT_PERDAGANGAN'
        judul_teks = "Peta Indikator dan Sektor Penyaluran"
        palet_warna = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'] # Biru, Hijau, dll
    else:
        # GEDSI
        fitur_utama = ['PCT_PEREMPUAN', 'PCT_SD_SMP']
        var_x = 'PCT_SD_SMP'
        var_y = 'PCT_PEREMPUAN'
        var_size = 'RASIO_PENETRASI' # Ukuran bubble menunjukkan kepadatan masyarakat yang terjangkau
        judul_teks = "Peta Radar Inklusi Sosial (GEDSI)"
        palet_warna = ['#EC4899', '#8B5CF6', '#F472B6', '#C4B5FD', '#10B981'] # Pink/Ungu untuk asosiasi GEDSI
    
    X = df_agg[fitur_utama].fillna(0)
    
    if len(X) < 2:
        return px.scatter(title="Data wilayah tidak mencukupi untuk klastering"), 0, 0
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    best_k = 2
    best_score = -1
    max_k = min(6, len(df_agg) - 1) 
    
    if max_k >= 2:
        for k in range(2, max_k + 1):
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            score = silhouette_score(X_scaled, kmeans_temp.fit_predict(X_scaled))
            if score > best_score:
                best_score, best_k = score, k
                
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df_agg['KLASTER'] = kmeans_final.fit_predict(X_scaled)
        df_agg['NAMA_KLASTER'] = 'Klaster ' + (df_agg['KLASTER'] + 1).astype(str)
        teks_validasi = f"✅ Optimal pada {best_k} Klaster (Silhouette: {best_score:.2f})"
    else:
        df_agg['NAMA_KLASTER'] = 'Tunggal'
        teks_validasi = "Gagal Klastering."

    fig_kmeans = px.scatter(
        df_agg, 
        x=var_x, 
        y=var_y,
        color='NAMA_KLASTER', 
        size=var_size,
        hover_name='NAMA_KABKOT',
        hover_data={
            'NAMA_PROVINSI': True, 
            'PCT_PERTANIAN': ':.1f', 
            'PCT_PERDAGANGAN': ':.1f',
            'TICKET_SIZE_JUTA': ':.1f',
            'RASIO_PENETRASI': ':.2f',
            'PCT_PEREMPUAN': ':.1f', 
            'PCT_SD_SMP': ':.1f'     
        },
        color_discrete_sequence=palet_warna,
        title=f"{judul_teks}<br><sup>{teks_validasi}</sup>",
        labels={
            'PCT_PERTANIAN': 'Porsi Sektor Pertanian (%)', 
            'TICKET_SIZE_JUTA': 'Plafon/Orang (Juta Rp)', 
            'NAMA_KLASTER': 'Grup',
            'RASIO_PENETRASI': 'Penetrasi Kredit (%)',
            'PCT_PERDAGANGAN': 'Porsi Sektor Dagang (%)',
            'PCT_PEREMPUAN': 'Porsi Perempuan (%)',
            'PCT_SD_SMP': 'Porsi Pend. Dasar SD/SMP (%)'
        }
    )

    fig_kmeans.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    if filter_provinsi != "SEMUA":
        for trace in fig_kmeans.data:
            if trace.type == 'scatter':
                opacities = [1.0 if prov == filter_provinsi else 0.1 for prov in trace.customdata[:, 0]]
                trace.marker.opacity = opacities

    fig_kmeans.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#374151'),
        yaxis=dict(showgrid=True, gridcolor='#374151')
    )
    
    return fig_kmeans, best_score, best_k

# Uji Coba
if __name__ == "__main__":
    import os
    import sys
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from utils.data_loader import load_and_process_raw_data
        
        print("⏳ Mempersiapkan data master untuk uji coba...")
        df_master = load_and_process_raw_data()
        
        if not df_master.empty:
            wilayah_tes = "SEMUA" 
            
            print(f"\nCek indikator untuk {wilayah_tes}...")
            fig_eko, score_eko, k_eko = jalankan_kmeans_clustering(df_master, wilayah_tes, mode_lensa="Indikator dan Sektor Penyaluran")
            
            if k_eko > 0:
                print(f"Membagi wilayah menjadi {k_eko} klaster.")
                print(f"Silhouette Score: {score_eko:.2f}")
                fig_eko.show()
            else:
                print("Gagal")

            # Tes GEDSI
            print(f"\nCek GEDSI untuk {wilayah_tes}...")
            fig_sosial, score_sosial, k_sosial = jalankan_kmeans_clustering(df_master, wilayah_tes, mode_lensa="Sosial / GEDSI")
            
            if k_sosial > 0:
                print(f"GEDSI membagi wilayah menjadi {k_sosial} klaster.")
                print(f"Silhouette Score: {score_sosial:.2f}")
                
                fig_sosial.show() 
            else:
                print("Gagal GEDSI.")
                
        else:
            print("Data master kosong atau file CSV tidak ditemukan.")
            
    except ImportError as e:
        print(f"kesalahan Import: {e}.")