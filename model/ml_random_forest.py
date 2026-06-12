import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

def siapkan_data_rf(df_master, filter_provinsi="SEMUA"):
    df_ml = df_master.copy()
    
    df_ml = df_ml[df_ml['NAMA_KABKOT'].astype(str) != 'NV']
    df_ml = df_ml[~df_ml['NAMA_KABKOT'].astype(str).str.startswith('KALIMANTAN')]
    
    if filter_provinsi != "SEMUA":
        df_ml = df_ml[df_ml['NAMA_PROVINSI'].astype(str).str.upper() == str(filter_provinsi).upper()]
        
    df_rf = df_ml.groupby(['NAMA_KABKOT', 'PROGRAM', 'SEKTOR_USAHA']).agg(
        TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
        TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum'),
        PR_DEBITUR=('DEBITUR_PEREMPUAN', 'sum'),
        SD_DEBITUR=('PENDIDIKAN_SD_SMP', 'sum'),
        JUMLAH_PENDUDUK=('JUMLAH_PENDUDUK', 'max')
    ).reset_index()
    
    df_rf['PCT_PEREMPUAN'] = np.where(df_rf['TOTAL_DEBITUR'] > 0, (df_rf['PR_DEBITUR'] / df_rf['TOTAL_DEBITUR']) * 100, 0)
    df_rf['PCT_PEND_DASAR'] = np.where(df_rf['TOTAL_DEBITUR'] > 0, (df_rf['SD_DEBITUR'] / df_rf['TOTAL_DEBITUR']) * 100, 0)
    df_rf['RASIO_PENETRASI'] = np.where(df_rf['JUMLAH_PENDUDUK'] > 0, (df_rf['TOTAL_DEBITUR'] / df_rf['JUMLAH_PENDUDUK']) * 100, 0)
    
    df_model = pd.get_dummies(df_rf, columns=['PROGRAM', 'SEKTOR_USAHA'], drop_first=False)
    
    y = df_model['TOTAL_PENYALURAN'] 
    kolom_buang = ['TOTAL_PENYALURAN', 'NAMA_KABKOT', 'TOTAL_DEBITUR', 'PR_DEBITUR', 'SD_DEBITUR', 'JUMLAH_PENDUDUK']
    X = df_model.drop(columns=kolom_buang, errors='ignore')
    
    return X, y

@st.cache_data(show_spinner=False)
def jalankan_rf_importance(df_master, filter_provinsi="SEMUA"):
    X, y = siapkan_data_rf(df_master, filter_provinsi)
    
    if len(X) < 15:
        fig_kosong = go.Figure().update_layout(title="Data wilayah tidak cukup untuk melatih Random Forest")
        return fig_kosong, 0 

    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)
    
    r2_full = rf.score(X, y)
    importances = rf.feature_importances_
    
    bobot_rf = dict(zip(X.columns, importances))

    nama_fitur = X.columns.str.replace('PROGRAM_', 'Program: ').str.replace('SEKTOR_USAHA_', 'Sektor: ')

    new_label = {
        'RASIO_PENETRASI': 'Rasio Debitur per Penduduk Produktif (%)',
        'PCT_PEND_DASAR': 'Porsi Debitur Pend. Dasar (SD/SMP)',
        'PCT_PEREMPUAN': 'Porsi Debitur Perempuan'
    }
    
    nama_fitur_bersih = pd.Series(nama_fitur).replace(new_label).tolist()
    
    df_imp = pd.DataFrame({'Fitur': nama_fitur_bersih, 'Importance': importances * 100})
    df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(8) 
    
    teks_validasi = f"✅ Kesesuaian Model (R-Squared): {r2_full*100:.1f}%"
    
    fig_rf = px.bar(
        df_imp, x='Importance', y='Fitur', orientation='h',
        title=f"Deteksi Faktor Pendorong Plafon Pembiayaan<br><sup>{teks_validasi}</sup>",
        labels={'Importance': 'Tingkat Pengaruh Keberhasilan (%)', 'Fitur': ''},
        color_discrete_sequence=['#8B5CF6']
    )
    fig_rf.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig_rf.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#374151', range=[0, df_imp['Importance'].max() * 1.2]), 
        yaxis=dict(showgrid=False)
    )
    
    return fig_rf, r2_full, bobot_rf

# Untuk validasi model
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from utils.data_loader import load_and_process_raw_data
        df_master = load_and_process_raw_data()
        
        if not df_master.empty:
            wilayah_tes = "SEMUA" 
            print(f"Evaluasi Random Forest untuk {wilayah_tes}...")
            
            X, y = siapkan_data_rf(df_master, wilayah_tes)
            
            if len(X) >= 15:
                # Pemisahan training dan testing
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                rf_test = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
                rf_test.fit(X_train, y_train)
                
                r2_train = rf_test.score(X_train, y_train)
                r2_test = rf_test.score(X_test, y_test)
                
                print(f"R-Squared (Latihan) : {r2_train*100:.1f}%")
                print(f"R-Squared (Ujian)   : {r2_test*100:.1f}%")
                print(f"Delta (Selisih)     : {(r2_train - r2_test)*100:.1f}%")
            else:
                 print("Data tidak mencukupi untuk training.")
                 
    except ImportError as e:
        print(f"Terjadi kesalahan Import: {e}")