import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium
import plotly.express as px

# Import Model
from model.ml_forecasting import jalankan_prediksi_wilayah, siapkan_data_geografis
from model.ml_sarimax import jalankan_sarimax
from model.ml_kmeans import jalankan_kmeans_clustering
from model.ml_random_forest import jalankan_rf_importance

# Import Utils
from utils.data_loader import load_and_filter_geojson, load_and_process_raw_data, load_ews_skema_data, load_leverage_trend_data, load_graduasi_trend_data

st.set_page_config(
    page_title="KALI-TREASURE | DJPb",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 42px !important; font-weight: 900; color: #38BDF8; margin-bottom: 0px; letter-spacing: 1px;}
    .sub-header { font-size: 20px !important; font-weight: 500; color: var(--text-color); opacity: 0.8; margin-bottom: 30px; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; }
    div[data-testid="metric-container"] { margin-bottom: 10px; }
    hr { margin-top: 15px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

def format_rupiah_dinamis(angka):
    if pd.isna(angka) or angka == 0: return "Rp0"
    if angka >= 1e12:
        triliun = angka / 1e12
        teks = f"{triliun:,.2f}"
        return "Rp" + teks.replace(',', 'X').replace('.', ',').replace('X', '.') + " T"
    else:
        miliar = angka / 1e9
        teks = f"{miliar:,.1f}"
        return "Rp" + teks.replace(',', 'X').replace('.', ',').replace('X', '.') + " M"

def format_orang_indo(angka):
    if pd.isna(angka) or angka == 0: return "0"
    return f"{int(angka):,}".replace(',', '.')


# SIDEBAR
df_master = load_and_process_raw_data()

with st.sidebar:
    st.image("image/intress.png", width=120)
    st.markdown("### 🎛️ Panel Filter")
    st.markdown("---")
    
    if not df_master.empty:
        daftar_prov = ["SEMUA"] + sorted(df_master['NAMA_PROVINSI'].unique().tolist())
        pilih_prov = st.selectbox("📍 Filter Provinsi:", daftar_prov)
        
        if pilih_prov == "SEMUA":
            daftar_kabkot = ["SEMUA"]
        else:
            kabkot_tersedia = df_master[df_master['NAMA_PROVINSI'] == pilih_prov]['NAMA_KABKOT'].unique().tolist()
            daftar_kabkot = ["SEMUA"] + sorted(kabkot_tersedia)

        pilih_kabkot = st.selectbox("🏢 Filter Kab/Kota:", daftar_kabkot)
    else:
        pilih_prov, pilih_kabkot = "SEMUA", "SEMUA"
    
    st.markdown("---")
    st.info("💡 **Tips:** Gunakan filter di atas untuk review kinerja kewilayahan secara spesifik.")
    st.caption("Tim Bu Novi | Kanwil DJPb Prov. Kaltara 2026")

# AREA DASHBOARD
st.markdown('<p class="main-header">KALI-TREASURE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Kalimantan Treasury Radar for SME Financing</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Peta Realisasi & Inklusi", 
    "📈 Tren & Proporsi Sektor Usaha", 
    "🤖 Anomali Inefisiensi Subsidi & Forecasting", 
    "🧩 Clustering & Feature Importance",
    "⚖️ Simulasi Kebijakan"
])

# TAB 1
with tab1:
    st.markdown("#### Sebaran Penyaluran & Inklusi Sosial (GEDSI)")
    
    if not df_master.empty:
        min_tahun = int(df_master['TAHUN'].min())
        max_tahun = int(df_master['TAHUN'].max())
        tahun_pilih = st.slider(
            "📅 Rentang Waktu (Tahun):", 
            min_value=min_tahun, max_value=max_tahun, 
            value=(min_tahun, max_tahun)
        )
    if tahun_pilih[0] > 2024 or tahun_pilih[1] > 2024:
        st.warning("⚠️ Catatan: Data Realisasi KUR hanya tersedia hingga April 2024. Data tahun 2025 dan 2026 hanya mencakup Data Realisasi UMi.")
    
    st.markdown("---")
    
    # BARIS 1: PETA DAN REALISASI AGREGAT
    col_map, col_metric = st.columns([2.5, 1.5])
    
    with col_map:
        lensa_peta = st.selectbox(
            "🗺️ Tampilkan Peta Berdasarkan (Metrik):",
            [
                "💰 Nominal Penyaluran", 
                "🚨 Rasio Debitur per Penduduk Usia Produktif",
                "👩 Porsi Perempuan (%)", 
                "🎓 Porsi Pendidikan Dasar (%)"
            ]
        )
        
        filter_program_peta = st.radio(
            "🎯 Filter Program (Peta):",
            ["Gabungan", "Khusus KUR", "Khusus UMi"],
            horizontal=True
        )

        if not df_master.empty:
            df_map = df_master[(df_master['TAHUN'] >= tahun_pilih[0]) & (df_master['TAHUN'] <= tahun_pilih[1])]
            
            if pilih_prov != "SEMUA":
                df_map = df_map[df_map['NAMA_PROVINSI'] == pilih_prov]
            if pilih_kabkot != "SEMUA":
                df_map = df_map[df_map['NAMA_KABKOT'] == pilih_kabkot]
                
            teks_program_peta = "Gabungan"
            df_peta = df_map.copy()
            if filter_program_peta == "Khusus KUR":
                df_peta = df_peta[df_peta['PROGRAM'] == 'KUR']
                teks_program_peta = "KUR"
            elif filter_program_peta == "Khusus UMi":
                df_peta = df_peta[df_peta['PROGRAM'] == 'UMI']
                teks_program_peta = "UMi"

            # Agregasi untuk Map 
            df_agregat_map = df_peta.groupby('NAMA_KABKOT').agg(
                TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
                TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum'),
                DEBITUR_PEREMPUAN=('DEBITUR_PEREMPUAN', 'sum'),
                PENDIDIKAN_SD_SMP=('PENDIDIKAN_SD_SMP', 'sum'),
                JUMLAH_PENDUDUK=('JUMLAH_PENDUDUK', 'max') 
            ).reset_index()
            
            df_agregat_map['MAP_KABKOT'] = df_agregat_map['NAMA_KABKOT'].str.title().str.replace('Kab. ', '', regex=False)
            df_agregat_map['PENYALURAN_MILIAR'] = df_agregat_map['TOTAL_PENYALURAN'] / 1e9
            
            df_agregat_map['PCT_PR'] = np.where(df_agregat_map['TOTAL_DEBITUR'] > 0, (df_agregat_map['DEBITUR_PEREMPUAN'] / df_agregat_map['TOTAL_DEBITUR']) * 100, 0)
            df_agregat_map['PCT_SD'] = np.where(df_agregat_map['TOTAL_DEBITUR'] > 0, (df_agregat_map['PENDIDIKAN_SD_SMP'] / df_agregat_map['TOTAL_DEBITUR']) * 100, 0)
            df_agregat_map['RASIO_PENETRASI'] = np.where(df_agregat_map['JUMLAH_PENDUDUK'] > 0, (df_agregat_map['TOTAL_DEBITUR'] / df_agregat_map['JUMLAH_PENDUDUK']) * 100, 0)

            warna_peta = "YlGnBu" 
            
            if lensa_peta == "💰 Nominal Penyaluran":
                kolom_aktif = "PENYALURAN_MILIAR"
                judul_legenda = f"Penyaluran {teks_program_peta} (Miliar Rp)"
            elif lensa_peta == "🚨 Rasio Debitur per Penduduk Usia Produktif":
                kolom_aktif = "RASIO_PENETRASI"
                judul_legenda = f"Rasio Debitur {teks_program_peta} (%)"
                warna_peta = "RdYlGn" 
            elif lensa_peta == "👩 Porsi Perempuan (%)":
                kolom_aktif = "PCT_PR"
                judul_legenda = f"Porsi Pr {teks_program_peta} (%)"
            elif lensa_peta == "🎓 Porsi Pendidikan Dasar (%)":
                kolom_aktif = "PCT_SD"
                judul_legenda = f"Porsi Pend. Dasar {teks_program_peta} (%)"

            file_geojson_kab = os.path.join('data', '38 Provinsi Indonesia - Kabupaten.json')
            geojson_kab = load_and_filter_geojson(file_geojson_kab, level='kabkot')
            
            if geojson_kab and not df_agregat_map.empty:
                for feature in geojson_kab['features']:
                    nama_kab = feature['properties'].get('WADMKK', '')
                    match = df_agregat_map[df_agregat_map['MAP_KABKOT'] == nama_kab]
                    
                    if not match.empty:
                        feature['properties']['TOOLTIP_TOTAL'] = format_rupiah_dinamis(match['TOTAL_PENYALURAN'].values[0])
                        feature['properties']['TOOLTIP_PCT_PR'] = f"{match['PCT_PR'].values[0]:.1f}%"
                        feature['properties']['TOOLTIP_PCT_SD'] = f"{match['PCT_SD'].values[0]:.1f}%"
                        feature['properties']['TOOLTIP_POPULASI'] = format_orang_indo(match['JUMLAH_PENDUDUK'].values[0])
                        feature['properties']['TOOLTIP_PENETRASI'] = f"{match['RASIO_PENETRASI'].values[0]:.2f}%"
                        feature['properties']['TOOLTIP_DEBITUR'] = f"{format_orang_indo(match['TOTAL_DEBITUR'].values[0])} Orang"
                    else:
                        feature['properties']['TOOLTIP_TOTAL'] = "Rp 0"
                        feature['properties']['TOOLTIP_PCT_PR'] = "0.0%"
                        feature['properties']['TOOLTIP_PCT_SD'] = "0.0%"
                        feature['properties']['TOOLTIP_POPULASI'] = "0"
                        feature['properties']['TOOLTIP_PENETRASI'] = "0.0%"
                        feature['properties']['TOOLTIP_DEBITUR'] = "0 Orang"

                m = folium.Map(location=[0.5, 114.0], zoom_start=6, tiles='CartoDB positron')
                
                choro = folium.Choropleth(
                    geo_data=geojson_kab,
                    name="Peta Penyaluran",
                    data=df_agregat_map,
                    columns=["MAP_KABKOT", kolom_aktif], 
                    key_on="feature.properties.WADMKK",
                    fill_color=warna_peta, 
                    fill_opacity=0.8,
                    line_opacity=0.3,
                    legend_name=judul_legenda, 
                    nan_fill_color="white"
                )
                
                tooltip = folium.GeoJsonTooltip(
                    fields=['WADMKK', 'TOOLTIP_TOTAL', 'TOOLTIP_DEBITUR', 'TOOLTIP_POPULASI', 'TOOLTIP_PENETRASI', 'TOOLTIP_PCT_PR', 'TOOLTIP_PCT_SD'],
                    aliases=['Wilayah:', f'Penyaluran ({teks_program_peta}):', f'Debitur ({teks_program_peta}):', '👨‍👩‍👧‍👦 Penduduk Usia Produktif:', f'🎯 Rasio Debitur ({teks_program_peta}):', '👩 Porsi Perempuan:', '🎓 Porsi Pend. Dasar:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 11px; padding: 8px; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);")
                )
                choro.geojson.add_child(tooltip)
                choro.add_to(m)
                
                st_folium(m, width="100%", height=450, returned_objects=[])
                
                if lensa_peta == "🚨 Rasio Debitur per Penduduk Usia Produktif":
                    st.caption("🔴 **Merah**: Rasio debitur terhadap penduduk usia produktif tergolong rendah (perlu dorongan ekspansi). 🟢 **Hijau**: Rasio sasaran inklusi sudah optimal.")
            else:
                st.warning("Tidak ada data untuk kombinasi filter ini, atau file GeoJSON tidak tersedia.")
    
    with col_metric:
        if not df_master.empty and not df_map.empty:
            total_penyaluran = df_map['TOTAL_PENYALURAN'].sum()
            total_debitur = df_map['TOTAL_DEBITUR'].sum()
            total_out = df_map['TOTAL_OUTSTANDING'].sum()
            
            df_kur = df_map[df_map['PROGRAM'] == 'KUR']
            df_umi = df_map[df_map['PROGRAM'] == 'UMI']
            
            st.markdown(f"##### 📊 Realisasi Agregat ({tahun_pilih[0]}-{tahun_pilih[1]})")
            
            # KPI Penyaluran
            st.metric(label="Total Penyaluran", value=format_rupiah_dinamis(total_penyaluran))
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("KUR", format_rupiah_dinamis(df_kur['TOTAL_PENYALURAN'].sum()))
            col_p2.metric("UMi", format_rupiah_dinamis(df_umi['TOTAL_PENYALURAN'].sum()))
            
            st.markdown("<hr style='border:1px dashed #4B5563; margin-top:-10px; margin-bottom:-15px'>", unsafe_allow_html=True)
        
            # KPI Debitur
            st.metric(label="Total Debitur", value=f"{format_orang_indo(total_debitur)} Orang")
            col_d1, col_d2 = st.columns(2)
            col_d1.metric("KUR", f"{format_orang_indo(df_kur['TOTAL_DEBITUR'].sum())}")
            col_d2.metric("UMi", f"{format_orang_indo(df_umi['TOTAL_DEBITUR'].sum())}")

            st.markdown("<hr style='border:1px dashed #4B5563; margin-top:-10px; margin-bottom:-15px'>", unsafe_allow_html=True)

            # KPI Outstanding
            rasio_out = (total_out / total_penyaluran * 100) if total_penyaluran > 0 else 0
            st.metric(label="Total Outstanding", value=format_rupiah_dinamis(total_out), delta=f"Rasio Mengendap: {rasio_out:.1f}%", delta_color="off")


    # BARIS 2: TOP 5 KABUPATEN/KOTA (Penyaluran & Debitur)
    st.markdown("---")
    st.markdown(f"##### 🏆 Top 5 Kabupaten/Kota ({tahun_pilih[0]}-{tahun_pilih[1]})")

    df_top5 = df_master[(df_master['TAHUN'] >= tahun_pilih[0]) & (df_master['TAHUN'] <= tahun_pilih[1])]
    
    if pilih_prov != "SEMUA":
        df_top5 = df_top5[df_top5['NAMA_PROVINSI'] == pilih_prov]
        
    if not df_top5.empty:
        df_top5_agg = df_top5.groupby('NAMA_KABKOT').agg(
            TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
            TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum')
        ).reset_index()
        
        df_top5_peny = df_top5_agg.nlargest(5, 'TOTAL_PENYALURAN').sort_values('TOTAL_PENYALURAN', ascending=True)
        df_top5_deb = df_top5_agg.nlargest(5, 'TOTAL_DEBITUR').sort_values('TOTAL_DEBITUR', ascending=True)
        
        df_top5_peny['LABEL'] = df_top5_peny['TOTAL_PENYALURAN'].apply(format_rupiah_dinamis)
        df_top5_deb['LABEL'] = df_top5_deb['TOTAL_DEBITUR'].apply(format_orang_indo)
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            fig_t1 = px.bar(
                df_top5_peny, x='TOTAL_PENYALURAN', y='NAMA_KABKOT', orientation='h',
                text='LABEL',
                title="Top 5 Penyaluran (Rupiah)",
                labels={'TOTAL_PENYALURAN': '', 'NAMA_KABKOT': ''},
                color_discrete_sequence=['#1E3A8A'] 
            )
            fig_t1.update_traces(textposition='inside', textfont=dict(color='white', size=13))
            fig_t1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=40, b=0), 
                height=230,
                xaxis=dict(showticklabels=False, showgrid=False) 
            )
            st.plotly_chart(fig_t1, use_container_width=True)
            
        with col_t2:
            fig_t2 = px.bar(
                df_top5_deb, x='TOTAL_DEBITUR', y='NAMA_KABKOT', orientation='h',
                text='LABEL',
                title="Top 5 Jumlah Debitur Terbanyak",
                labels={'TOTAL_DEBITUR': '', 'NAMA_KABKOT': ''},
                color_discrete_sequence=['#F59E0B'] 
            )
            fig_t2.update_traces(textposition='inside', textfont=dict(color='white', size=13))
            fig_t2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=40, b=0), 
                height=230,
                xaxis=dict(showticklabels=False, showgrid=False)
            )
            st.plotly_chart(fig_t2, use_container_width=True)
    else:
        st.warning("Data kewilayahan tidak tersedia untuk rentang tahun ini.")
        
    # BARIS 3: INKLUSI (GEDSI)
    st.markdown("<hr style='border:1px solid #E5E7EB; margin-top:0px; margin-bottom:15px'>", unsafe_allow_html=True)
    st.markdown("##### 🎯 Perspektif Inklusi (GEDSI)")
    
    if not df_master.empty and not df_map.empty:
        tot_all = df_map['TOTAL_DEBITUR'].sum()
        pr_all = df_map['DEBITUR_PEREMPUAN'].sum()
        lk_all = tot_all - pr_all
        pct_pr_all = (pr_all / tot_all) * 100 if tot_all > 0 else 0.0
        pct_lk_all = (100.0 - pct_pr_all) if tot_all > 0 else 0.0
        
        sd_all = df_map['PENDIDIKAN_SD_SMP'].sum()
        ti_all = tot_all - sd_all
        pct_sd_all = (sd_all / tot_all) * 100 if tot_all > 0 else 0.0
        pct_ti_all = (100.0 - pct_sd_all) if tot_all > 0 else 0.0

        tot_kur = df_kur['TOTAL_DEBITUR'].sum()
        pr_kur = df_kur['DEBITUR_PEREMPUAN'].sum()
        lk_kur = tot_kur - pr_kur
        pct_pr_kur = (pr_kur / tot_kur) * 100 if tot_kur > 0 else 0.0
        pct_lk_kur = (100.0 - pct_pr_kur) if tot_kur > 0 else 0.0
        
        sd_kur = df_kur['PENDIDIKAN_SD_SMP'].sum()
        ti_kur = tot_kur - sd_kur
        pct_sd_kur = (sd_kur / tot_kur) * 100 if tot_kur > 0 else 0.0
        pct_ti_kur = (100.0 - pct_sd_kur) if tot_kur > 0 else 0.0
        
        tot_umi = df_umi['TOTAL_DEBITUR'].sum()
        pr_umi = df_umi['DEBITUR_PEREMPUAN'].sum()
        lk_umi = tot_umi - pr_umi
        pct_pr_umi = (pr_umi / tot_umi) * 100 if tot_umi > 0 else 0.0
        pct_lk_umi = (100.0 - pct_pr_umi) if tot_umi > 0 else 0.0
        
        sd_umi = df_umi['PENDIDIKAN_SD_SMP'].sum()
        ti_umi = tot_umi - sd_umi
        pct_sd_umi = (sd_umi / tot_umi) * 100 if tot_umi > 0 else 0.0
        pct_ti_umi = (100.0 - pct_sd_umi) if tot_umi > 0 else 0.0

        col_gedsi_gender, col_gedsi_edu = st.columns(2)

        with col_gedsi_gender:
            st.markdown("**👩 Pemberdayaan Perempuan (Gender Equality)**")
            
            st.caption("Penyaluran TOTAL (KUR & UMi)")
            st.progress(int(pct_pr_all))
            cg_a1, cg_a2 = st.columns(2)
            cg_a1.metric("Perempuan", f"{pct_pr_all:.1f}%", f"{format_orang_indo(pr_all)} Org", delta_color="off")
            cg_a2.metric("Laki-laki", f"{pct_lk_all:.1f}%", f"{format_orang_indo(lk_all)} Org", delta_color="off")
            
            st.markdown("<hr style='border:1px dashed #4B5563; margin-top:5px; margin-bottom:15px'>", unsafe_allow_html=True)
            
            st.caption("Penyaluran Program KUR")
            st.progress(int(pct_pr_kur))
            cg_k1, cg_k2 = st.columns(2)
            cg_k1.metric("Perempuan", f"{pct_pr_kur:.1f}%", f"{format_orang_indo(pr_kur)} Org", delta_color="off")
            cg_k2.metric("Laki-laki", f"{pct_lk_kur:.1f}%", f"{format_orang_indo(lk_kur)} Org", delta_color="off")
            
            st.caption("Penyaluran Program UMi")
            st.progress(int(pct_pr_umi))
            cg_u1, cg_u2 = st.columns(2)
            cg_u1.metric("Perempuan", f"{pct_pr_umi:.1f}%", f"{format_orang_indo(pr_umi)} Org", delta_color="off")
            cg_u2.metric("Laki-laki", f"{pct_lk_umi:.1f}%", f"{format_orang_indo(lk_umi)} Org", delta_color="off")

        with col_gedsi_edu:
            st.markdown("**🎓 Inklusi Latar Belakang Pendidikan**")
            
            st.caption("Penyaluran TOTAL (KUR & UMi)")
            st.progress(int(pct_sd_all))
            ce_a1, ce_a2 = st.columns(2)
            ce_a1.metric("Pendidikan Dasar", f"{pct_sd_all:.1f}%", f"{format_orang_indo(sd_all)} Org", delta_color="off")
            ce_a2.metric("Menengah/Tinggi", f"{pct_ti_all:.1f}%", f"{format_orang_indo(ti_all)} Org", delta_color="off")
            
            st.markdown("<hr style='border:1px dashed #4B5563; margin-top:5px; margin-bottom:15px'>", unsafe_allow_html=True)
            
            st.caption("Penyaluran Program KUR")
            st.progress(int(pct_sd_kur))
            ce_k1, ce_k2 = st.columns(2)
            ce_k1.metric("Pendidikan Dasar", f"{pct_sd_kur:.1f}%", f"{format_orang_indo(sd_kur)} Org", delta_color="off")
            ce_k2.metric("Menengah/Tinggi", f"{pct_ti_kur:.1f}%", f"{format_orang_indo(ti_kur)} Org", delta_color="off")
            
            st.caption("Penyaluran Program UMi")
            st.progress(int(pct_sd_umi))
            ce_u1, ce_u2 = st.columns(2)
            ce_u1.metric("Pendidikan Dasar", f"{pct_sd_umi:.1f}%", f"{format_orang_indo(sd_umi)} Org", delta_color="off")
            ce_u2.metric("Menengah/Tinggi", f"{pct_ti_umi:.1f}%", f"{format_orang_indo(ti_umi)} Org", delta_color="off")

        # GRAFIK TREN GEDSI
        st.markdown("<br>", unsafe_allow_html=True)
        col_chart_gender, col_chart_edu = st.columns(2)
        
        df_trend = df_map.groupby(['TAHUN', 'PROGRAM']).agg(
            TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum'),
            PEREMPUAN=('DEBITUR_PEREMPUAN', 'sum'),
            SD_SMP=('PENDIDIKAN_SD_SMP', 'sum')
        ).reset_index()
        
        df_trend['PCT_PEREMPUAN'] = np.where(
            df_trend['TOTAL_DEBITUR'] > 0, 
            (df_trend['PEREMPUAN'] / df_trend['TOTAL_DEBITUR']) * 100, 0
        )
        df_trend['PCT_SD_SMP'] = np.where(
            df_trend['TOTAL_DEBITUR'] > 0, 
            (df_trend['SD_SMP'] / df_trend['TOTAL_DEBITUR']) * 100, 0
        )
        
        # Grafik Garis Gender
        fig_gender = px.line(
            df_trend, x='TAHUN', y='PCT_PEREMPUAN', color='PROGRAM',
            markers=True,
            hover_data={'PCT_PEREMPUAN': ':.1f', 'PEREMPUAN': True, 'TOTAL_DEBITUR': True},
            labels={'PCT_PEREMPUAN': 'Porsi (%)', 'PEREMPUAN': 'Jml Perempuan', 'TOTAL_DEBITUR': 'Total Debitur'},
            color_discrete_sequence=['#1E3A8A', '#F59E0B'],
            title='Tren Pemberdayaan Perempuan per Tahun (%)'
        )
        fig_gender.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
            xaxis=dict(dtick=1, title="Tahun"),
            yaxis=dict(title="Porsi Perempuan (%)", range=[0, 105]),
            hovermode="x unified" # Memunculkan tooltip komparasi dalam 1 garis vertikal
        )
        
        with col_chart_gender:
            st.plotly_chart(fig_gender, use_container_width=True)

        # Grafik Garis Pendidikan
        fig_edu = px.line(
            df_trend, x='TAHUN', y='PCT_SD_SMP', color='PROGRAM',
            markers=True,
            hover_data={'PCT_SD_SMP': ':.1f', 'SD_SMP': True, 'TOTAL_DEBITUR': True},
            labels={'PCT_SD_SMP': 'Porsi (%)', 'SD_SMP': 'Jml SD/SMP', 'TOTAL_DEBITUR': 'Total Debitur'},
            color_discrete_sequence=['#1E3A8A', '#F59E0B'],
            title='Tren Inklusi Pendidikan Dasar per Tahun (%)'
        )
        fig_edu.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
            xaxis=dict(dtick=1, title="Tahun"),
            yaxis=dict(title="Porsi Debitur (SD/SMP) (%)", range=[0, 105]),
            hovermode="x unified" # Memunculkan tooltip komparasi dalam 1 garis vertikal
        )

        with col_chart_edu:
            st.plotly_chart(fig_edu, use_container_width=True)
    
    st.markdown("---")
    st.markdown("**👩‍👧‍👦 Profil Demografi Perempuan**")
    st.info("Gunakan panel di bawah ini untuk melihat karakteristik debitur perempuan secara dinamis berdasarkan berbagai dimensi dan metrik.")
    
    if pilih_prov == "SEMUA":
        teks_lokasi = "Seluruh Provinsi di Kalimantan"
    elif pilih_kabkot == "SEMUA":
        teks_lokasi = f"Provinsi {pilih_prov.title()} (Semua Kabupaten/Kota)"
    else:
        teks_lokasi = f"{pilih_kabkot.title()}, Provinsi {pilih_prov.title()}"

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    
    with col_ctrl1:
        pilihan_program_gedsi = st.radio(
            "1. Pilih Program:",
            ["SEMUA (Gabungan)", "Khusus KUR", "Khusus UMi"],
            key="radio_tab1_gedsi"
        )
        
    with col_ctrl2:
        dimensi_analisis = st.selectbox(
            "2. Lihat Berdasarkan (Dimensi):",
            ["Status Perkawinan", "Sektor Usaha", "Latar Belakang Pendidikan (Semua Perempuan)", "Pendidikan Ibu Rumah Tangga (Perempuan Menikah)"]
        )
        
    with col_ctrl3:
        metrik_analisis = st.selectbox(
            "3. Ukur Berdasarkan (Metrik):",
            ["Jumlah Debitur (Orang)", "Nominal Penyaluran (Rupiah)"]
        )

    df_gedsi_filtered = df_map.copy()
    teks_program = "Gabungan KUR & UMi"
    
    if pilihan_program_gedsi == "Khusus KUR":
        df_gedsi_filtered = df_gedsi_filtered[df_gedsi_filtered['PROGRAM'] == 'KUR']
        teks_program = "Program KUR"
    elif pilihan_program_gedsi == "Khusus UMi":
        df_gedsi_filtered = df_gedsi_filtered[df_gedsi_filtered['PROGRAM'] == 'UMI']
        teks_program = "Program UMi"

    # hanya data Perempuan
    if 'NAMA_JNS_KELAMIN' in df_gedsi_filtered.columns:
        df_pr_aktif = df_gedsi_filtered[df_gedsi_filtered['NAMA_JNS_KELAMIN'] == 'PEREMPUAN']
    else:
        df_pr_aktif = pd.DataFrame()

    # Grafik Dinamis sesuai Req User
    fig_dinamis = None
    if not df_pr_aktif.empty:
        kolom_metrik = 'TOTAL_DEBITUR' if metrik_analisis == "Jumlah Debitur (Orang)" else 'TOTAL_PENYALURAN'
        format_hover = ":,.0f" if metrik_analisis == "Jumlah Debitur (Orang)" else "Rp :,.0f"
        
        if dimensi_analisis in ["Status Perkawinan", "Sektor Usaha"]:
            kolom_dimensi = 'NAMA_MARITAL_STS' if dimensi_analisis == "Status Perkawinan" else 'SEKTOR_USAHA'
            
            df_chart = df_pr_aktif.groupby(kolom_dimensi)[kolom_metrik].sum().reset_index()
            df_chart = df_chart.sort_values(by=kolom_metrik, ascending=False)
            
            # Jika Sektor Usaha (kategori banyak) menggunakan Bar Chart. Jika Status Perkawinan menggunakan Donut Chart.
            if dimensi_analisis == "Sektor Usaha":
                top_sektor = df_chart.head(5)[kolom_dimensi].tolist()
                df_chart['KATEGORI_FINAL'] = np.where(df_chart[kolom_dimensi].isin(top_sektor), df_chart[kolom_dimensi], 'Sektor Lainnya')
                df_chart = df_chart.groupby('KATEGORI_FINAL')[kolom_metrik].sum().reset_index().sort_values(by=kolom_metrik, ascending=True)
                
                fig_dinamis = px.bar(
                    df_chart, x=kolom_metrik, y='KATEGORI_FINAL', orientation='h',
                    title=f"Distribusi {dimensi_analisis} Perempuan<br><sup>{teks_program}</sup>",
                    color_discrete_sequence=['#EC4899']
                )
                fig_dinamis.update_layout(xaxis_title=metrik_analisis, yaxis_title="")
                
            else:
                fig_dinamis = px.pie(
                    df_chart, values=kolom_metrik, names=kolom_dimensi, hole=0.45,
                    title=f"Distribusi {dimensi_analisis} Perempuan<br><sup>{teks_program}</sup>",
                    color_discrete_sequence=['#EC4899', '#8B5CF6', '#F472B6', '#C4B5FD']
                )
                fig_dinamis.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)

        elif dimensi_analisis == "Latar Belakang Pendidikan (Semua Perempuan)":
            if metrik_analisis == "Jumlah Debitur (Orang)":
                tot_pr = df_pr_aktif['TOTAL_DEBITUR'].sum()
                sd_pr = df_pr_aktif['PENDIDIKAN_SD_SMP'].sum()
                ti_pr = tot_pr - sd_pr
                
                df_chart = pd.DataFrame({
                    'KATEGORI': ['Pendidikan Dasar (SD/SMP)', 'Menengah / Tinggi'],
                    'NILAI': [sd_pr, ti_pr]
                })
                
                fig_dinamis = px.pie(
                    df_chart, values='NILAI', names='KATEGORI', hole=0.45,
                    title=f"Pendidikan Keseluruhan Debitur Perempuan<br><sup>{teks_program}</sup>",
                    color_discrete_sequence=['#F59E0B', '#10B981']
                )
                fig_dinamis.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
            else:
                st.warning("Dimensi Latar Belakang Pendidikan saat ini hanya mendukung metrik 'Jumlah Debitur (Orang)'.")
                fig_dinamis = None
            
        elif dimensi_analisis == "Pendidikan Ibu Rumah Tangga (Perempuan Menikah)":
            if metrik_analisis == "Jumlah Debitur (Orang)":
                #  Hanya Perempuan KAWIN
                df_pr_kawin = df_pr_aktif[df_pr_aktif['NAMA_MARITAL_STS'] == 'KAWIN']
                
                tot_kawin = df_pr_kawin['TOTAL_DEBITUR'].sum()
                sd_kawin = df_pr_kawin['PENDIDIKAN_SD_SMP'].sum()
                ti_kawin = tot_kawin - sd_kawin
                
                df_chart = pd.DataFrame({
                    'KATEGORI': ['Pendidikan Dasar (SD/SMP)', 'Menengah / Tinggi'],
                    'NILAI': [sd_kawin, ti_kawin]
                })
                
                fig_dinamis = px.pie(
                    df_chart, values='NILAI', names='KATEGORI', hole=0.45,
                    title=f"Pendidikan Ibu Rumah Tangga (Perempuan Menikah)<br><sup>{teks_program}</sup>",
                    color_discrete_sequence=['#4ADE80', '#FBBF24']
                )
                fig_dinamis.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
            else:
                st.warning("Analisis irisan pendidikan ini hanya mendukung perhitungan 'Jumlah Debitur (Orang)'.")
                fig_dinamis = None

        if fig_dinamis is not None:
            fig_dinamis.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=60, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_dinamis, use_container_width=True)
    
    else:
        st.warning(f"Data debitur perempuan tidak tersedia untuk {teks_program} di wilayah ini.")

# TAB 2: TREN & KINERJA SEKTORAL
with tab2:
    st.markdown("#### Analisis Tren Waktu & Proporsi Sektor Usaha")
    
    if not df_master.empty:
        pilihan_program_tab2 = st.radio(
            "🎯 Filter Program:",
            ["SEMUA (Gabungan KUR & UMi)", "Khusus KUR", "Khusus UMi"],
            horizontal=True,
            key="radio_tab2_sektoral"
        )
        
        df_tab2 = df_master.copy()
        
        # Filter Wilayah
        if pilih_prov != "SEMUA":
            df_tab2 = df_tab2[df_tab2['NAMA_PROVINSI'] == pilih_prov]
        if pilih_kabkot != "SEMUA":
            df_tab2 = df_tab2[df_tab2['NAMA_KABKOT'] == pilih_kabkot]

        # Filter Program berdasarkan Radio Button
        if pilihan_program_tab2 == "Khusus KUR":
            df_tab2 = df_tab2[df_tab2['PROGRAM'] == 'KUR']
        elif pilihan_program_tab2 == "Khusus UMi":
            df_tab2 = df_tab2[df_tab2['PROGRAM'] == 'UMI']

        st.markdown("---")
        
        # Validasi data kosong
        if df_tab2.empty:
            st.warning(f"Data tidak tersedia untuk kombinasi wilayah dan program {pilihan_program_tab2}.")
        else:
            col_tren, col_donut = st.columns([2, 1.5])
            
            with col_tren:
                st.markdown("**📈 Tren Penyaluran (Top 5 Sektor)**")
                
                # Agregasi data sektoral
                df_sektor = df_tab2.groupby(['TAHUN', 'SEKTOR_USAHA'])['TOTAL_PENYALURAN'].sum().reset_index()
                
                # Cari 5 sektor terbesar, sisanya jadikan 'LAINNYA'
                top_5_sektor = df_sektor.groupby('SEKTOR_USAHA')['TOTAL_PENYALURAN'].sum().nlargest(5).index
                df_sektor['SEKTOR_FINAL'] = np.where(df_sektor['SEKTOR_USAHA'].isin(top_5_sektor), df_sektor['SEKTOR_USAHA'], 'LAINNYA')
                df_tren_final = df_sektor.groupby(['TAHUN', 'SEKTOR_FINAL'])['TOTAL_PENYALURAN'].sum().reset_index()

                max_penyaluran = df_tren_final['TOTAL_PENYALURAN'].max()
                
                #format trilian / miliar sumbu y
                if max_penyaluran >= 1e12:
                    df_tren_final['PENYALURAN_PLOT'] = df_tren_final['TOTAL_PENYALURAN'] / 1e12
                    satuan_y = " T"
                else:
                    df_tren_final['PENYALURAN_PLOT'] = df_tren_final['TOTAL_PENYALURAN'] / 1e9
                    satuan_y = " M"

                fig_tren = px.line(
                    df_tren_final, x='TAHUN', y='PENYALURAN_PLOT', color='SEKTOR_FINAL',
                    markers=True, title="Tren Sektoral per Tahun",
                    custom_data=['SEKTOR_FINAL', 'TOTAL_PENYALURAN'] 
                )
                
                fig_tren.update_traces(
                    hovertemplate="<b>Tahun %{x}</b><br>Sektor: %{customdata[0]}<br>Nilai: Rp %{customdata[1]:,.0f}<extra></extra>"
                )

                fig_tren.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(dtick=1, title="Tahun"),
                    yaxis=dict(title=f"Total Penyaluran ({satuan_y.strip()})", ticksuffix=satuan_y),
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, title="")
                )
                st.plotly_chart(fig_tren, use_container_width=True)

            with col_donut:
                st.markdown("**🍩 Komposisi Sektor Usaha (Akumulasi)**")
                
                # Agregasi total per sektor untuk Donut Chart
                df_pie = df_tab2.groupby('SEKTOR_USAHA')['TOTAL_PENYALURAN'].sum().reset_index()
                
                # Threshold 3% untuk menggabungkan sektor kecil
                threshold = df_pie['TOTAL_PENYALURAN'].sum() * 0.03
                df_pie['SEKTOR_FINAL'] = np.where(df_pie['TOTAL_PENYALURAN'] < threshold, 'LAINNYA', df_pie['SEKTOR_USAHA'])
                df_pie_final = df_pie.groupby('SEKTOR_FINAL')['TOTAL_PENYALURAN'].sum().reset_index()

                fig_donut = px.pie(
                    df_pie_final, values='TOTAL_PENYALURAN', names='SEKTOR_FINAL', hole=0.45,
                    title="Persentase Distribusi Sektor",
                    custom_data=['SEKTOR_FINAL', 'TOTAL_PENYALURAN']
                )

                fig_donut.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    showlegend=False,
                    hovertemplate="<b>%{label}</b><br>Total Penyaluran: Rp %{value:,.0f}<br>Porsi: %{percent}<extra></extra>"
                    )
                
                fig_donut.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=30, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.warning("Data belum tersedia. Pastikan file data berhasil dimuat.")

    st.markdown("---")
    st.markdown("### 🎓 Analisis Transisi Pembiayaan UMKM (UMi ke KUR)")
    
    df_graduasi_raw = load_graduasi_trend_data()
    
    if not df_graduasi_raw.empty:
        if pilih_prov != "SEMUA":
            df_graduasi_raw = df_graduasi_raw[df_graduasi_raw['NAMA_PROVINSI'] == pilih_prov]
        if pilih_kabkot != "SEMUA":
            df_graduasi_raw = df_graduasi_raw[df_graduasi_raw['NAMA_KABKOT'] == pilih_kabkot]
            
        df_graduasi = df_graduasi_raw.groupby('TAHUN').agg(
            DEBITUR_KUR=('DEBITUR_KUR', 'sum'),
            DEBITUR_UMI=('DEBITUR_UMI', 'sum')
        ).reset_index()

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        fig_grad = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Garis UMi (Area Inkubasi - Primary Y)
        fig_grad.add_trace(
            go.Scatter(x=df_graduasi['TAHUN'], y=df_graduasi['DEBITUR_UMI'], 
                        name="Debitur UMi", 
                        mode='lines+markers',
                        line=dict(color='#10B981', width=4, dash='dot')), # Garis putus-putus hijau
            secondary_y=False,
        )
        
        # Garis KUR (Area Transisi - Secondary Y)
        fig_grad.add_trace(
            go.Scatter(x=df_graduasi['TAHUN'], y=df_graduasi['DEBITUR_KUR'], 
                        name="Debitur KUR", 
                        mode='lines+markers',
                        fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.1)',
                        line=dict(color='#0ea5e9', width=3)), # Garis biru solid
            secondary_y=True,
        )
        
        lokasi_teks = "Seluruh Kalimantan" if pilih_prov == "SEMUA" else (f"Provinsi {pilih_prov.title()}" if pilih_kabkot == "SEMUA" else pilih_kabkot.title())
            
        fig_grad.update_layout(
            title=f"Korelasi Pertumbuhan Debitur: {lokasi_teks} (2017-2023)",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False, type='category', title="Tahun") 
        )
        
        fig_grad.update_yaxes(title_text="Jumlah Debitur UMi (Orang)", secondary_y=False, showgrid=True, gridcolor='#374151', color='#10B981')
        fig_grad.update_yaxes(title_text="Jumlah Debitur KUR (Orang)", secondary_y=True, showgrid=False, color='#0ea5e9')
        
        st.plotly_chart(fig_grad, use_container_width=True)
        
        st.info("💡 Gunakan grafik ini untuk memantau keberhasilan *naik kelas* melalui *Lagging Effect*. Jika pembinaan dan transisi skala usaha berjalan sukses, lonjakan jumlah debitur pada garis hijau putus-putus (UMi) biasanya akan diikuti oleh lonjakan pada garis biru (KUR) **2 hingga 3 tahun kemudian** di wilayah yang sama. Jika garis biru tetap stagnan, hal ini mengindikasikan bahwa UMKM di daerah tersebut gagal melakukan eskalasi kapasitas dan terjebak di plafon pembiayaan ultra-mikro. Data SUPER MIKRO pada KUR telah dihilangkan dari analisis ini.")
    else:
        st.warning("Data tren graduasi tidak tersedia.")
    st.markdown("---")

with tab3:
    st.markdown("#### 🚨 Anomali Inefisiensi Skema Pembiayaan (2018-2021)")
    st.info("Analisis ini memetakan riwayat efisiensi tiap Skema Kredit. Jika terdapat titik anomali di sebelah Kuadran Kiri Atas adalah titik bahwa **Skema tertentu beban subsidi negara sangat besar per debitur, namun nominal pembiayaan yang disalurkan sangat kecil.**\nData merupakan irisan antara data Realisasi KUR dan Subsisi KUR sepanjang tahun 2018-2021")
    
    # Memanggil dataset khusus EWS yang sudah dikoreksi
    df_ews_master = load_ews_skema_data()
    
    if not df_ews_master.empty:
        if pilih_kabkot != "SEMUA":
            st.warning("Grafik inefisiensi skema tidak tersedia untuk wilayah kabupaten/kota. Silakan ubah filter Kab/Kota kembali ke 'SEMUA' untuk melihat persebaran inefisiensi tingkat provinsi.")
        else:
            df_ews_filter = df_ews_master.copy()
            if pilih_prov != "SEMUA":
                df_ews_filter = df_ews_filter[df_ews_filter['NAMA_PROVINSI'] == pilih_prov]
                
            if not df_ews_filter.empty:
                df_ews_filter['PENYALURAN_JUTA'] = df_ews_filter['PENYALURAN_PER_DEBITUR'] / 1e6
                df_ews_filter['SUBSIDI_JUTA'] = df_ews_filter['SUBSIDI_PER_DEBITUR'] / 1e6
                
                fig_ews = px.scatter(
                    df_ews_filter, 
                    x='PENYALURAN_JUTA',
                    y='SUBSIDI_JUTA',    
                    size='TOTAL_DEBITUR', 
                    color='NAMA_SKEMA',
                    color_discrete_map={
                        'MIKRO': '#10B981',         
                        'KECIL': "#35ABE1",         
                        'SUPER MIKRO': "#E89910",   
                        'TKI': "#FF00C3"            
                    },
                    hover_name='NAMA_KABKOT', # Menampilkan nama Kab/Kot di pop-up
                    hover_data={
                        'NAMA_SKEMA': True, 
                        'NAMA_KABKOT': True, # Disembunyikan karena sudah jadi judul
                        'PENYALURAN_JUTA': False, 
                        'SUBSIDI_JUTA': False,    
                        'TOTAL_DEBITUR': ':,',    
                        'PENYALURAN_PER_DEBITUR': ':,.0f', 
                        'SUBSIDI_PER_DEBITUR': ':,.0f'     
                    },
                    title="Peta Inefisiensi Skema (Bubble Size = Jumlah Debitur)",
                    labels={
                        'PENYALURAN_JUTA': 'Rata-rata Penyaluran per Debitur',
                        'SUBSIDI_JUTA': 'Beban Subsidi Bunga per Debitur',
                        'NAMA_SKEMA': 'Skema Kredit',
                        'PENYALURAN_PER_DEBITUR': 'Penyaluran Asli (Rp)',
                        'SUBSIDI_PER_DEBITUR': 'Subsidi Asli (Rp)'
                    }
                )
                
                fig_ews.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='#E5E7EB', ticksuffix=' Juta'),
                    yaxis=dict(showgrid=True, gridcolor='#E5E7EB', ticksuffix=' Juta'),
                    height=550
                )
                
                avg_penyaluran = df_ews_filter['PENYALURAN_JUTA'].median()
                avg_subsidi = df_ews_filter['SUBSIDI_JUTA'].median()
                fig_ews.add_vline(x=avg_penyaluran, line_dash="dash", line_color="red", opacity=0.5)
                fig_ews.add_hline(y=avg_subsidi, line_dash="dash", line_color="red", opacity=0.5)
                
                st.plotly_chart(fig_ews, use_container_width=True)
                
                teks_lokasi_ews = "Seluruh Kalimantan" if pilih_prov == "SEMUA" else f"Provinsi {pilih_prov.title()}"
                st.caption(f"Garis putus-putus merah menandakan median (nilai tengah) efisiensi untuk {teks_lokasi_ews}. Titik yang berada di atas garis horizontal dan di kiri garis vertikal adalah skema dengan potensi inefisiensi anggaran.")
            else:
                st.warning("Data inefisiensi skema tidak tersedia untuk provinsi ini.")
    else:
        st.warning("File historis Subsidi KUR tidak ditemukan atau sedang diproses.")

    # Leverage Subsidi terhadap Kredit KUR
    st.markdown("---")
    st.markdown("### 📈 Tren Tahunan Daya Ungkit (Leverage) Subsidi Bunga terhadap Kredit")
    
    df_trend_raw = load_leverage_trend_data()
    
    if not df_trend_raw.empty:
        if pilih_prov != "SEMUA":
            df_trend_raw = df_trend_raw[df_trend_raw['NAMA_PROVINSI'] == pilih_prov]
        if pilih_kabkot != "SEMUA":
            df_trend_raw = df_trend_raw[df_trend_raw['NAMA_KABKOT'] == pilih_kabkot]
        
        df_trend = df_trend_raw.groupby('TAHUN').agg(
            TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
            TOTAL_SUBSIDI=('TOTAL_SUBSIDI', 'sum')
        ).reset_index()
        
        df_trend['MULTIPLIER'] = np.where(
            df_trend['TOTAL_SUBSIDI'] > 0, 
            df_trend['TOTAL_PENYALURAN'] / df_trend['TOTAL_SUBSIDI'], 0
        )

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_trend.add_trace(
            go.Scatter(x=df_trend['TAHUN'], y=df_trend['TOTAL_PENYALURAN'], 
                        name="Volume Penyaluran (Rp)", 
                        fill='tozeroy', mode='lines+markers',
                        line=dict(color='#1E3A8A', width=2)),
            secondary_y=False,
        )
        
        fig_trend.add_trace(
            go.Scatter(x=df_trend['TAHUN'], y=df_trend['MULTIPLIER'], 
                        name="Daya Ungkit (X)", 
                        mode='lines+markers+text',
                        text=df_trend['MULTIPLIER'].apply(lambda x: f"{x:.1f}x"),
                        textposition="top center",
                        line=dict(color='#F59E0B', width=4)),
            secondary_y=True,
        )
        
        if pilih_prov == "SEMUA":
            lokasi_teks = "Seluruh Kalimantan"
        elif pilih_kabkot == "SEMUA":
            lokasi_teks = f"Provinsi {pilih_prov.title()}"
        else:
            lokasi_teks = f"{pilih_kabkot.title()}"
            
        fig_trend.update_layout(
            title=f"Kinerja Leverage Subsidi: {lokasi_teks} (2017-2021)",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False, type='category', title="Tahun") 
        )
        
        fig_trend.update_yaxes(title_text="Nominal Penyaluran (Rp)", secondary_y=False, showgrid=True, gridcolor='#374151')
        fig_trend.update_yaxes(title_text="Daya Ungkit (Penyaluran/Subsidi)", secondary_y=True, showgrid=False)
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.info("💡 **Analisis Efisiensi Subsidi:** Menggunakan agregasi tahun penuh untuk menghilangkan anomali *lag* penyaluran subsidi. Dengan filter wilayah, dapat ditinjau apakah serapan subsidi di suatu kabupaten/kota memberikan dampak ungkit yang sepadan dibandingkan dengan rata-rata provinsi.")
    else:
        st.warning("Data tren daya ungkit subsidi tidak tersedia untuk rentang atau wilayah terpilih.")

    #Proyeksi ML    
    st.markdown("---")
    st.markdown("#### 🔮 Proyeksi Beban Subsidi")
    st.info("**Holt-Winters** murni menggunakan data historis penyaluran untuk memproyeksikan beban subsidi di masa depan, sedangkan **SARIMAX** menggunakan pendekatan statistik yang lebih kompleks dengan menambahkan variabel eksogen yaitu BI Rate, PDRB, dan Realisasi KUR.")
    
    col_model1, col_model2 = st.columns([3, 1])
    with col_model1:
        pilihan_model = st.selectbox(
            "🧠 Pilih Mesin Prediksi (Algoritma):", 
            ["-- Pilih Algoritma --", "Holt-Winters (Exponential Smoothing)", "SARIMAX (Machine Learning dengan Eksogen)"]
        )
        
    with col_model2:
        st.markdown("<br>", unsafe_allow_html=True) # Spasi agar tombol sejajar dengan selectbox
        tombol_jalankan = st.button("Jalankan Model", use_container_width=True)
    
    if tombol_jalankan:
        if pilihan_model == "-- Pilih Algoritma --":
            st.warning("⚠️ Silakan pilih algoritma prediksi pada menu drop-down di atas terlebih dahulu.")
        else:
            with st.spinner(f"Mesin {pilihan_model} sedang melakukan kalkulasi..."):
                try:
                    if "Holt-Winters" in pilihan_model:
                        df_geo = siapkan_data_geografis()
                        df_hasil, fig_hasil = jalankan_prediksi_wilayah(df_geo, filter_provinsi=pilih_prov, filter_kabkot=pilih_kabkot)
                    elif "SARIMAX" in pilihan_model:
                        df_hasil, fig_hasil = jalankan_sarimax(filter_provinsi=pilih_prov, filter_kabkot=pilih_kabkot)
                        
                    if df_hasil is not None and fig_hasil is not None:
                        st.pyplot(fig_hasil, clear_figure=True)
                        
                        st.markdown("##### 🚨 Tabel Rincian Skenario Proyeksi Subsidi")
                        st.dataframe(df_hasil, use_container_width=True, hide_index=True)
                    else:
                        st.warning("⚠️ Data historis tidak mencukupi atau kosong untuk kombinasi wilayah ini.")
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan sistematik pada mesin prediksi: {e}")

with tab4:
    st.markdown("#### 🧩 Clustering & Feature Importance untuk Profiling Wilayah")
    st.info("Analisis ini mengkombinasikan metode *Unsupervised Learning* (K-Means) dan *Supervised Learning* (Random Forest) untuk mengelompokkan wilayah dan menilai fitur yang paling penting dalam pengambilan keputusan.")

    if not df_master.empty:
        st.markdown("**K-Means Clustering**")
        
        mode_lensa = st.radio(
            "🔎 Pilih Fitur Analisis (Parameter Matriks K-Means):", 
            ["Indikator dan Sektor Penyaluran", "Sosial/GEDSI"], 
            horizontal=True,
            help="Indikator dan Sektor Penyaluran berfokus pada rasio debitur, penyaluran, dan sektor usaha. GEDSI berfokus pada kelompok inklusif (perempuan dan pendidikan dasar)."
        )
        if mode_lensa == "Indikator dan Sektor Penyaluran":
            st.caption("Fitur yang digunakan: Average Penyaluran per Debitur, Rasio Debitur per Penduduk Usia Produktif, % Sektor Pertanian, dan % Sektor Perdagangan.")
        else:
            st.caption("Fitur yang digunakan: Rasio Debitur Perempuan, Rasio Debitur dengan Pendidikan SD/SMP.")

        if pilih_kabkot != "SEMUA":
            st.warning("⚠️ Filter Kabupaten/Kota aktif. Harap ubah filter ke 'SEMUA' pada level provinsi agar mesin dapat memetakan perbandingan antar-wilayah.")
        else:
            fig_km, score_km, k_km = jalankan_kmeans_clustering(df_master, pilih_prov, mode_lensa)
            st.plotly_chart(fig_km, use_container_width=True)
            
            if mode_lensa == "Indikator dan Sektor Penyaluran":
                st.info("💡 **Cara Membaca Indikator dan Sektor Penyaluran:** Sumbu horizontal menunjukkan kekuatan sektor pertanian, sedangkan sumbu vertikal menunjukkan kekuatan modal (Plafon/Orang). Gelembung besar menandakan dominasi sektor Perdagangan.")
            else:
                st.info("💡 **Cara Membaca Sosial/GEDSI:** Kuadran Kanan-Atas (Episentrum kelompok rentan/subsisten). Kuadran Kiri-Bawah (Eksklusif: didominasi pria berpendidikan tinggi). Ukuran gelembung menunjukkan seberapa luas tingkat penetrasi fasilitas kredit.")
                
    
        st.markdown("---")
        st.markdown("**Faktor Pendorong (Random Forest)**")
        fig_rf, r2_rf, _ = jalankan_rf_importance(df_master, pilih_prov)
        st.plotly_chart(fig_rf, use_container_width=True)
        
        st.markdown("""
        <div style='background-color: var(--secondary-background-color); color: var(--text-color); padding: 15px; border-radius: 10px; border-left: 5px solid #8B5CF6; margin-bottom: 15px;'>
            <b>Interpretasi Grafik:</b> Bar chart di atas menunjukkan seberapa kuat intervensi pada suatu variabel berdampak pada total plafon kredit.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Data historis tidak tersedia untuk menjalankan mesin analitik.")

# TAB 5: POLICY SIMULATOR
with tab5:
    st.markdown("#### ⚖️ Deterministik Simulasi Kebijakan Sektoral")
    st.info("Simulator kebijakan ini memproyeksikan **Dampak Inklusi Sosial (GEDSI)** apabila eksekutif mengambil keputusan untuk menaikkan atau menurunkan alokasi pagu penyaluran pada sektor ekonomi tertentu. Basis perhitungan menggunakan *conversion rate* historis.")
    
    if not df_master.empty:
        # Menentukan Baseline (Program -> Tahun)
        col_prog, col_thn = st.columns(2)
        
        with col_prog:
            pilihan_program_sim = st.radio("🎯 Fokus Simulasi Program:", ["SEMUA (KUR & UMi)", "Khusus KUR", "Khusus UMi"], horizontal=True)
            
        # Terapkan filter program untuk mencari daftar tahun yang valid
        df_temp_year = df_master.copy()
        if pilihan_program_sim == "Khusus KUR":
            df_temp_year = df_temp_year[df_temp_year['PROGRAM'] == 'KUR']
        elif pilihan_program_sim == "Khusus UMi":
            df_temp_year = df_temp_year[df_temp_year['PROGRAM'] == 'UMI']

        with col_thn:
            daftar_tahun = sorted(df_temp_year['TAHUN'].unique().tolist(), reverse=True)
            if not daftar_tahun:
                st.warning("Tidak ada data untuk program ini.")
                tahun_terakhir = None
            else:
                tahun_terakhir = st.selectbox("📅 Pilih Tahun Basis Konversi Sektoral:", daftar_tahun)
                
                # Menampilkan peringatan jika memilih "SEMUA" di tahun di mana KUR belum tersedia
                if pilihan_program_sim == "SEMUA (KUR & UMi)" and tahun_terakhir > 2024:
                    st.warning("⚠️ Catatan: Data Realisasi KUR hanya tersedia hingga April 2024. Simulasi untuk tahun 2025 ke atas murni menggunakan konversi data UMi.")

        if tahun_terakhir is not None:
            df_base = df_temp_year[df_temp_year['TAHUN'] == tahun_terakhir].copy()

            # Terapkan Filter Wilayah dari Sidebar
            if pilih_prov != "SEMUA":
                df_base = df_base[df_base['NAMA_PROVINSI'] == pilih_prov]
            if pilih_kabkot != "SEMUA":
                df_base = df_base[df_base['NAMA_KABKOT'] == pilih_kabkot]

            if not df_base.empty:
                df_sektor = df_base.groupby('SEKTOR_USAHA').agg(
                    TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
                    TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum'),
                    PEREMPUAN=('DEBITUR_PEREMPUAN', 'sum'),
                    SD_SMP=('PENDIDIKAN_SD_SMP', 'sum')
                ).reset_index()

                # Top 4 Sektor Terbesar
                top_sektor = df_sektor.nlargest(4, 'TOTAL_PENYALURAN')['SEKTOR_USAHA'].tolist()
                df_sektor['SEKTOR_SIM'] = np.where(df_sektor['SEKTOR_USAHA'].isin(top_sektor), df_sektor['SEKTOR_USAHA'], 'Sektor Lainnya')
                
                df_sim_base = df_sektor.groupby('SEKTOR_SIM').sum().reset_index()

                # Conversion Rate (Daya Ungkit per Rp 1 Miliar)
                df_sim_base['DEBITUR_PER_MILIAR'] = np.where(df_sim_base['TOTAL_PENYALURAN'] > 0, df_sim_base['TOTAL_DEBITUR'] / (df_sim_base['TOTAL_PENYALURAN'] / 1e9), 0)
                df_sim_base['PCT_PEREMPUAN'] = np.where(df_sim_base['TOTAL_DEBITUR'] > 0, df_sim_base['PEREMPUAN'] / df_sim_base['TOTAL_DEBITUR'], 0)
                df_sim_base['PCT_SD_SMP'] = np.where(df_sim_base['TOTAL_DEBITUR'] > 0, df_sim_base['SD_SMP'] / df_sim_base['TOTAL_DEBITUR'], 0)

                st.markdown("---")
                col_slider, col_result = st.columns([1, 2.2])

                with col_slider:
                    st.markdown(f"**🎚️ Panel Rekalibrasi Pagu Sektoral**")
                    st.caption(f"Basis Data Kalibrasi: Tahun {tahun_terakhir}")
                    
                    sliders = {}
                    daftar_sektor_urut = sorted(df_sim_base['SEKTOR_SIM'].unique(), key=lambda x: (x == 'Sektor Lainnya', x))
                    
                    for sektor in daftar_sektor_urut:
                        pagu_awal_miliar = df_sim_base[df_sim_base['SEKTOR_SIM'] == sektor]['TOTAL_PENYALURAN'].values[0] / 1e9
                        sliders[sektor] = st.slider(
                            f"{sektor}", 
                            min_value=-50, max_value=100, value=0, step=5,
                            help=f"Pagu Awal: Rp {pagu_awal_miliar:,.0f} M. Geser untuk intervensi pagu."
                        )
                
                with col_result:
                    st.markdown("**📊 Proyeksi Dampak Intervensi (Simulated Impact)**")
                    
                    df_sim_result = df_sim_base.copy()
                    df_sim_result['SIM_PENYALURAN'] = df_sim_result.apply(lambda row: row['TOTAL_PENYALURAN'] * (1 + (sliders[row['SEKTOR_SIM']] / 100)), axis=1)
                    
                    df_sim_result['SIM_DEBITUR'] = (df_sim_result['SIM_PENYALURAN'] / 1e9) * df_sim_result['DEBITUR_PER_MILIAR']
                    df_sim_result['SIM_PEREMPUAN'] = df_sim_result['SIM_DEBITUR'] * df_sim_result['PCT_PEREMPUAN']
                    df_sim_result['SIM_SD_SMP'] = df_sim_result['SIM_DEBITUR'] * df_sim_result['PCT_SD_SMP']

                    base_peny = df_sim_result['TOTAL_PENYALURAN'].sum()
                    sim_peny = df_sim_result['SIM_PENYALURAN'].sum()
                    base_deb = df_sim_result['TOTAL_DEBITUR'].sum()
                    sim_deb = df_sim_result['SIM_DEBITUR'].sum()
                    base_pr = df_sim_result['PEREMPUAN'].sum()
                    sim_pr = df_sim_result['SIM_PEREMPUAN'].sum()
                    base_sd = df_sim_result['SD_SMP'].sum()
                    sim_sd = df_sim_result['SIM_SD_SMP'].sum()

                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    diff_peny = sim_peny - base_peny
                    diff_deb = sim_deb - base_deb
                    diff_pr = sim_pr - base_pr

                    str_delta_peny = f"{diff_peny/1e9:,.1f} M"
                    str_delta_deb = f"{diff_deb:,.0f} Org"
                    str_delta_pr = f"{diff_pr:,.0f} Org"

                    col_m1.metric("Pagu Penyaluran Total", format_rupiah_dinamis(sim_peny), str_delta_peny if diff_peny != 0 else "Tetap")
                    col_m2.metric("Total Jangkauan UMKM", format_orang_indo(sim_deb), str_delta_deb if diff_deb != 0 else "Tetap")
                    col_m3.metric("Pemberdayaan Perempuan", format_orang_indo(sim_pr), str_delta_pr if diff_pr != 0 else "Tetap")

                    st.markdown("<br>", unsafe_allow_html=True)

                    df_chart_base = pd.DataFrame({'Target Inklusi': ['Total UMKM', 'Perempuan', 'Pend. Dasar (SD/SMP)'], 'Jumlah Target (Orang)': [base_deb, base_pr, base_sd], 'Skenario': 'Riil (Baseline)'})
                    df_chart_sim = pd.DataFrame({'Target Inklusi': ['Total UMKM', 'Perempuan', 'Pend. Dasar (SD/SMP)'], 'Jumlah Target (Orang)': [sim_deb, sim_pr, sim_sd], 'Skenario': 'Simulasi Kebijakan'})
                    df_chart = pd.concat([df_chart_base, df_chart_sim])

                    fig_sim = px.bar(
                        df_chart, x='Target Inklusi', y='Jumlah Target (Orang)', color='Skenario', barmode='group',
                        color_discrete_map={'Riil (Baseline)': '#9CA3AF', 'Simulasi Kebijakan': '#10B981'},
                        title="Komparasi Baseline vs Dampak Simulasi"
                    )
                    fig_sim.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                        margin=dict(l=0, r=0, t=40, b=0), 
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="")
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.warning("Data historis sektoral tidak tersedia untuk kombinasi program dan tahun ini.")
        st.markdown("<hr style='border: 2px solid #374151; margin-top: 40px; margin-bottom: 30px;'>", unsafe_allow_html=True)

        # Prediktif Simulasi Kebijakan (Random Forest)
        st.markdown("### 📇 Simulasi Kebijakan (Dampak Inklusi Sosial)")
        st.markdown("""
        <div style='background-color: var(--secondary-background-color); color: var(--text-color); padding: 15px; border-radius: 10px; border-left: 5px solid #8B5CF6; margin-bottom: 15px;'>
            Simulasikan Skenario: Jika berhasil mendorong peningkatan indikator inklusi demografi di bawah ini, berapa estimasi tambahan uang yang akan beredar di masyarakat berdasarkan model Machine Learning (Random Forest)?
        </div>
        """, unsafe_allow_html=True)
        
        df_simulasi_ml = df_master.copy()
        if pilih_prov != "SEMUA":
            df_simulasi_ml = df_simulasi_ml[df_simulasi_ml['NAMA_PROVINSI'] == pilih_prov]
        
        rata_plafon = df_simulasi_ml['TOTAL_PENYALURAN'].mean() if not df_simulasi_ml.empty else 0
        
        col_ml_sim1, col_ml_sim2 = st.columns([1, 1.5])

        _, _, bobot_rf = jalankan_rf_importance(df_master, pilih_prov)
        
        w_penetrasi = bobot_rf.get('RASIO_PENETRASI', 0)
        w_pendidikan = bobot_rf.get('PCT_PEND_DASAR', 0)
        w_perempuan = bobot_rf.get('PCT_PEREMPUAN', 0)
        
        with col_ml_sim1:
            st.markdown("**Skenario Target Pertumbuhan Demografi:**")
            sim_penetrasi = st.slider("📈 Kenaikan Rasio Penetrasi Kredit (Rasio Debitur per Penduduk Produktif)", min_value=0, max_value=50, value=0, step=5, format="+%d%%")
            sim_pendidikan = st.slider("🎓 Peningkatan Akses Lulusan SD/SMP", min_value=0, max_value=50, value=0, step=5, format="+%d%%")
            sim_perempuan = st.slider("👩 Pemberdayaan Pengusaha Perempuan", min_value=0, max_value=50, value=0, step=5, format="+%d%%")

            st.caption(f"*(Auto-Calibrated Weights: Penetrasi {w_penetrasi*100:.1f}%, Pendidikan {w_pendidikan*100:.1f}%, Perempuan {w_perempuan*100:.1f}%)*")
        with col_ml_sim2:
            st.markdown("**Prediksi Tambahan Dampak Fiskal Regional:**")
            estimasi_dampak = rata_plafon * (
                (sim_penetrasi / 100 * w_penetrasi) + 
                (sim_pendidikan / 100 * w_pendidikan) + 
                (sim_perempuan / 100 * w_perempuan)
            )
            
            if estimasi_dampak > 0:
                st.success(f"💰 **+ Rp {estimasi_dampak / 1e9:,.2f} Miliar** per wilayah/sektor.")
                st.caption(f"Estimasi tambahan perputaran uang di masyarakat dibandingkan kondisi saat ini (Baseline: Rp {rata_plafon / 1e9:,.2f} Miliar). *Kalkulasi ini menggunakan probabilitas kekuatan bobot (Feature Importance) dari algoritma Random Forest.*")
            else:
                st.info("Geser slider di samping untuk melihat prediksi dampak kebijakan inklusi sosial terhadap perputaran ekonomi.")    
    else:
        st.warning("Menunggu sinkronisasi data master...")