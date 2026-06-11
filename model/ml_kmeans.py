import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")

def jalankan_kmeans_clustering(df_master, filter_provinsi="SEMUA"):
    df_ml = df_master.copy()
    
    df_ml = df_ml[df_ml['NAMA_KABKOT'].astype(str) != 'NV']
    df_ml = df_ml[~df_ml['NAMA_KABKOT'].astype(str).str.startswith('KALIMANTAN')]

    df_agg = df_ml.groupby(['NAMA_PROVINSI', 'NAMA_KABKOT']).agg(
        TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
        TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum'),
        JUMLAH_PENDUDUK=('JUMLAH_PENDUDUK', 'max') # Diambil max karena populasinya konstan per tahun/wilayah
    ).reset_index()
    
    df_sektor = df_ml.groupby(['NAMA_KABKOT', 'SEKTOR_USAHA'])['TOTAL_PENYALURAN'].sum().unstack(fill_value=0).reset_index()
    df_agg = pd.merge(df_agg, df_sektor, on='NAMA_KABKOT', how='left')
    
    df_agg['TICKET_SIZE'] = np.where(df_agg['TOTAL_DEBITUR'] > 0, df_agg['TOTAL_PENYALURAN'] / df_agg['TOTAL_DEBITUR'], 0)
    df_agg['RASIO_PENETRASI'] = np.where(df_agg['JUMLAH_PENDUDUK'] > 0, (df_agg['TOTAL_DEBITUR'] / df_agg['JUMLAH_PENDUDUK']) * 100, 0)
    
    kolom_perdagangan = [col for col in df_agg.columns if 'PERDAGANGAN' in str(col).upper()]
    kolom_pertanian = [col for col in df_agg.columns if 'PERTANIAN' in str(col).upper() or 'KEHUTANAN' in str(col).upper()]
    
    val_perdagangan = df_agg[kolom_perdagangan].sum(axis=1) if kolom_perdagangan else 0
    val_pertanian = df_agg[kolom_pertanian].sum(axis=1) if kolom_pertanian else 0
    
    df_agg['PCT_PERDAGANGAN'] = np.where(df_agg['TOTAL_PENYALURAN'] > 0, (val_perdagangan / df_agg['TOTAL_PENYALURAN']) * 100, 0)
    df_agg['PCT_PERTANIAN'] = np.where(df_agg['TOTAL_PENYALURAN'] > 0, (val_pertanian / df_agg['TOTAL_PENYALURAN']) * 100, 0)
    
    fitur_utama = ['TICKET_SIZE', 'RASIO_PENETRASI', 'PCT_PERDAGANGAN', 'PCT_PERTANIAN']
    X = df_agg[fitur_utama].fillna(0)
    
    if len(X) < 2:
        fig_kosong = px.scatter(title="Data wilayah tidak mencukupi untuk klastering")
        return fig_kosong, 0, 0
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Silhouette Score
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
        teks_validasi = f"✅ Model optimal pada {best_k} Klaster (Silhouette Score: {best_score:.2f})"
    else:
        df_agg['NAMA_KLASTER'] = 'Tunggal'
        teks_validasi = "Gagal Klastering."

    df_agg['TICKET_SIZE_JUTA'] = df_agg['TICKET_SIZE'] / 1e6
    
    fig_kmeans = px.scatter(
        df_agg, 
        x='PCT_PERTANIAN', 
        y='TICKET_SIZE_JUTA',
        color='NAMA_KLASTER', 
        size='PCT_PERDAGANGAN',
        hover_name='NAMA_KABKOT',
        hover_data={
            'NAMA_PROVINSI': True, 
            'PCT_PERTANIAN': ':.1f', 
            'PCT_PERDAGANGAN': ':.1f',
            'TICKET_SIZE_JUTA': ':.1f',
            'RASIO_PENETRASI': ':.2f'
        },
        color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'],
        title=f"Peta DNA Ekonomi Wilayah<br><sup>{teks_validasi}</sup>",
        labels={
            'PCT_PERTANIAN': 'Porsi Sektor Pertanian (%)', 
            'TICKET_SIZE_JUTA': 'Plafon/Orang (Juta Rp)', 
            'NAMA_KLASTER': 'Grup',
            'RASIO_PENETRASI': 'Rasio Penetrasi Pasar (%)',
            'PCT_PERDAGANGAN': 'Porsi Sektor Dagang (%)'
        }
    )
    
    fig_kmeans.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    if filter_provinsi != "SEMUA":
        for trace in fig_kmeans.data:
            opacities = [1.0 if prov == filter_provinsi else 0.1 for prov in trace.customdata[:, 0]]
            trace.marker.opacity = opacities

    fig_kmeans.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#374151'),
        yaxis=dict(showgrid=True, gridcolor='#374151')
    )
    
    return fig_kmeans, best_score, best_k

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
            print(f"🧠 Menjalankan K-Means Clustering dengan Metrik Penetrasi untuk {wilayah_tes}...")
            
            fig, score, k = jalankan_kmeans_clustering(df_master, wilayah_tes)
            
            if k > 0:
                print(f"✅ SUKSES! Algoritma membagi wilayah menjadi {k} klaster.")
                print(f"📊 Nilai Validasi (Silhouette Score): {score:.2f}")
                print("🖼️ Membuka hasil visualisasi di browser...")
                fig.show()
            else:
                print("❌ Data tidak cukup setelah dibersihkan.")
        else:
            print("❌ Gagal: Data master kosong atau file CSV tidak ditemukan.")
            
    except ImportError as e:
        print(f"❌ Terjadi kesalahan Import: {e}. Pastikan Anda menjalankan skrip dari root folder proyek.")