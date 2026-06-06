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

# Import Utils
from utils.data_loader import load_and_filter_geojson, load_and_process_raw_data, load_ews_skema_data

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="KALI-TREASURE | DJPb",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 42px !important; font-weight: 900; color: #38BDF8; margin-bottom: 0px; letter-spacing: 1px;}
    .sub-header { font-size: 20px !important; font-weight: 500; color: #FFFFFF; margin-bottom: 30px; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; }
    div[data-testid="metric-container"] { margin-bottom: 10px; }
    hr { margin-top: 15px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# FUNGSI FORMAT ANGKA CERDAS
# ------------------------------------------
def format_rupiah_dinamis(angka):
    if pd.isna(angka) or angka == 0: return "Rp 0"
    if angka >= 1e12:
        triliun = angka / 1e12
        teks = f"{triliun:,.2f}"
        return "Rp " + teks.replace(',', 'X').replace('.', ',').replace('X', '.') + " T"
    else:
        miliar = angka / 1e9
        teks = f"{miliar:,.1f}"
        return "Rp " + teks.replace(',', 'X').replace('.', ',').replace('X', '.') + " M"

def format_orang_indo(angka):
    if pd.isna(angka) or angka == 0: return "0"
    return f"{int(angka):,}".replace(',', '.')


# ==========================================
# 3. SIDEBAR: KENDALI NAVIGASI DINAMIS
# ==========================================
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

# ==========================================
# 4. AREA UTAMA (MAIN DASHBOARD)
# ==========================================
st.markdown('<p class="main-header">KALI-TREASURE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Kalimantan Treasury Radar for SME Financing</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Peta Realisasi & Inklusi", 
    "📈 Tren & Kinerja Sektoral", 
    "🤖 Anomali Inefisiensi Subsidi & Forecasting", 
    "⚖️ Simulasi Kebijakan"
])

# ------------------------------------------
# TAB 1: PETA INKLUSI (FOLIUM CHOROPLETH)
# ------------------------------------------
with tab1:
    st.markdown("#### Sebaran Penyaluran & Inklusi Sosial (GEDSI)")
    
    if not df_master.empty:
        min_tahun = int(df_master['TAHUN'].min())
        max_tahun = int(df_master['TAHUN'].max())
        tahun_pilih = st.slider(
            "📅 Analisis Rentang Waktu (Tahun):", 
            min_value=min_tahun, max_value=max_tahun, 
            value=(min_tahun, max_tahun)
        )
    
    st.markdown("---")
    
    # ------------------------------------------
    # BARIS 1: PETA DAN KINERJA AGREGAT
    # ------------------------------------------
    col_map, col_metric = st.columns([2.5, 1.5])
    
    with col_map:
        # Pilihan Lensa Metrik Peta Interaktif
        lensa_peta = st.radio(
            "Tampilkan Peta Berdasarkan:",
            ["Total Penyaluran", "Porsi Perempuan (%)", "Porsi Pendidikan Dasar (%)"],
            horizontal=True
        )

        if not df_master.empty:
            df_map = df_master[(df_master['TAHUN'] >= tahun_pilih[0]) & (df_master['TAHUN'] <= tahun_pilih[1])]
            
            if pilih_prov != "SEMUA":
                df_map = df_map[df_map['NAMA_PROVINSI'] == pilih_prov]
            if pilih_kabkot != "SEMUA":
                df_map = df_map[df_map['NAMA_KABKOT'] == pilih_kabkot]
                
            # Agregasi Utama untuk Map dan Tooltip
            df_agregat_map = df_map.groupby('NAMA_KABKOT').agg(
                TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
                TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum'),
                DEBITUR_PEREMPUAN=('DEBITUR_PEREMPUAN', 'sum'),
                PENDIDIKAN_SD_SMP=('PENDIDIKAN_SD_SMP', 'sum')
            ).reset_index()
            
            df_agregat_map['MAP_KABKOT'] = df_agregat_map['NAMA_KABKOT'].str.title().str.replace('Kab. ', '', regex=False)
            df_agregat_map['PENYALURAN_MILIAR'] = df_agregat_map['TOTAL_PENYALURAN'] / 1e9
            
            df_agregat_map['PCT_PR'] = (df_agregat_map['DEBITUR_PEREMPUAN'] / df_agregat_map['TOTAL_DEBITUR']) * 100
            df_agregat_map['PCT_PR'] = df_agregat_map['PCT_PR'].fillna(0)
            df_agregat_map['PCT_SD'] = (df_agregat_map['PENDIDIKAN_SD_SMP'] / df_agregat_map['TOTAL_DEBITUR']) * 100
            df_agregat_map['PCT_SD'] = df_agregat_map['PCT_SD'].fillna(0)
            
            df_pivot = pd.pivot_table(df_map, values='TOTAL_PENYALURAN', index='NAMA_KABKOT', columns='PROGRAM', aggfunc='sum', fill_value=0).reset_index()
            if 'KUR' not in df_pivot.columns: df_pivot['KUR'] = 0
            if 'UMI' not in df_pivot.columns: df_pivot['UMI'] = 0
            
            df_agregat_map = df_agregat_map.merge(df_pivot, on='NAMA_KABKOT', how='left')
            
            # Logika Pemilihan Kolom dan Judul Legenda
            if lensa_peta == "Total Penyaluran":
                kolom_aktif = "PENYALURAN_MILIAR"
                judul_legenda = "Total Penyaluran (Miliar Rupiah)"
            elif lensa_peta == "Porsi Perempuan (%)":
                kolom_aktif = "PCT_PR"
                judul_legenda = "Porsi Debitur Perempuan (%)"
            else:
                kolom_aktif = "PCT_SD"
                judul_legenda = "Porsi Pendidikan Dasar (%)"

            file_geojson_kab = os.path.join('data', '38 Provinsi Indonesia - Kabupaten.json')
            geojson_kab = load_and_filter_geojson(file_geojson_kab, level='kabkot')
            
            if geojson_kab and not df_agregat_map.empty:
                for feature in geojson_kab['features']:
                    nama_kab = feature['properties'].get('WADMKK', '')
                    match = df_agregat_map[df_agregat_map['MAP_KABKOT'] == nama_kab]
                    
                    if not match.empty:
                        feature['properties']['TOOLTIP_TOTAL'] = format_rupiah_dinamis(match['TOTAL_PENYALURAN'].values[0])
                        feature['properties']['TOOLTIP_KUR'] = format_rupiah_dinamis(match['KUR'].values[0])
                        feature['properties']['TOOLTIP_UMI'] = format_rupiah_dinamis(match['UMI'].values[0])
                        feature['properties']['TOOLTIP_PCT_PR'] = f"{match['PCT_PR'].values[0]:.1f}%"
                        feature['properties']['TOOLTIP_PCT_SD'] = f"{match['PCT_SD'].values[0]:.1f}%"
                    else:
                        feature['properties']['TOOLTIP_TOTAL'] = "Rp 0"
                        feature['properties']['TOOLTIP_KUR'] = "Rp 0"
                        feature['properties']['TOOLTIP_UMI'] = "Rp 0"
                        feature['properties']['TOOLTIP_PCT_PR'] = "0.0%"
                        feature['properties']['TOOLTIP_PCT_SD'] = "0.0%"

                m = folium.Map(location=[0.5, 114.0], zoom_start=6, tiles='CartoDB positron')
                
                choro = folium.Choropleth(
                    geo_data=geojson_kab,
                    name="Peta Penyaluran",
                    data=df_agregat_map,
                    columns=["MAP_KABKOT", kolom_aktif], # Menggunakan kolom dinamis
                    key_on="feature.properties.WADMKK",
                    fill_color="YlGnBu",
                    fill_opacity=0.8,
                    line_opacity=0.3,
                    legend_name=judul_legenda, # Menggunakan judul legenda dinamis
                    nan_fill_color="white"
                )
                
                tooltip = folium.GeoJsonTooltip(
                    fields=['WADMKK', 'TOOLTIP_TOTAL', 'TOOLTIP_KUR', 'TOOLTIP_UMI', 'TOOLTIP_PCT_PR', 'TOOLTIP_PCT_SD'],
                    aliases=['Wilayah:', 'Total Pembiayaan:', 'Penyaluran KUR:', 'Penyaluran UMi:', '👩 Porsi Perempuan:', '🎓 Porsi Pend. Dasar:'],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 11px; padding: 8px; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);")
                )
                choro.geojson.add_child(tooltip)
                choro.add_to(m)
                
                st_folium(m, width="100%", height=450, returned_objects=[])
            else:
                st.warning("Tidak ada data untuk kombinasi filter ini, atau file GeoJSON tidak tersedia.")

    with col_metric:
        if not df_master.empty and not df_map.empty:
            total_penyaluran = df_map['TOTAL_PENYALURAN'].sum()
            total_debitur = df_map['TOTAL_DEBITUR'].sum()
            
            df_kur = df_map[df_map['PROGRAM'] == 'KUR']
            df_umi = df_map[df_map['PROGRAM'] == 'UMI']
            
            st.markdown(f"##### 📊 Kinerja Agregat ({tahun_pilih[0]}-{tahun_pilih[1]})")
            
            st.metric(label="Total Penyaluran Riil", value=format_rupiah_dinamis(total_penyaluran))
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("KUR", format_rupiah_dinamis(df_kur['TOTAL_PENYALURAN'].sum()))
            col_p2.metric("UMi", format_rupiah_dinamis(df_umi['TOTAL_PENYALURAN'].sum()))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.metric(label="Total Debitur", value=f"{format_orang_indo(total_debitur)} Orang")
            col_d1, col_d2 = st.columns(2)
            col_d1.metric("KUR", f"{format_orang_indo(df_kur['TOTAL_DEBITUR'].sum())}")
            col_d2.metric("UMi", f"{format_orang_indo(df_umi['TOTAL_DEBITUR'].sum())}")

    # ------------------------------------------
    # BARIS 2: PERSPEKTIF INKLUSI (GEDSI) DI BAWAH PETA
    # ------------------------------------------
    st.markdown("---")
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

        # ------------------------------------------
        # BARIS 3: GRAFIK TREN GEDSI (PLOTLY LINE CHART)
        # ------------------------------------------
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


    # ------------------------------------------
    # BARIS 4: TOP 5 KABUPATEN/KOTA (Penyaluran & Debitur)
    # ------------------------------------------
    st.markdown("<br><hr style='border:1px solid #E5E7EB; margin-top:10px; margin-bottom:10px'>", unsafe_allow_html=True)
    st.markdown(f"##### 🏆 Top 5 Kabupaten/Kota ({tahun_pilih[0]}-{tahun_pilih[1]})")

    # Data Frame khusus Top 5: Hanya patuh pada Tahun dan Provinsi
    df_top5 = df_master[(df_master['TAHUN'] >= tahun_pilih[0]) & (df_master['TAHUN'] <= tahun_pilih[1])]
    
    if pilih_prov != "SEMUA":
        df_top5 = df_top5[df_top5['NAMA_PROVINSI'] == pilih_prov]
        
    if not df_top5.empty:
        # Agregasi data agregat total tanpa memisahkan KUR/UMi
        df_top5_agg = df_top5.groupby('NAMA_KABKOT').agg(
            TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
            TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum')
        ).reset_index()
        
        # Ekstraksi Top 5 (Diurutkan ascending agar batang terpanjang di atas pada chart horizontal)
        df_top5_peny = df_top5_agg.nlargest(5, 'TOTAL_PENYALURAN').sort_values('TOTAL_PENYALURAN', ascending=True)
        df_top5_deb = df_top5_agg.nlargest(5, 'TOTAL_DEBITUR').sort_values('TOTAL_DEBITUR', ascending=True)
        
        # Label cerdas untuk mencetak angka di dalam batang grafik
        df_top5_peny['LABEL'] = df_top5_peny['TOTAL_PENYALURAN'].apply(format_rupiah_dinamis)
        df_top5_deb['LABEL'] = df_top5_deb['TOTAL_DEBITUR'].apply(format_orang_indo)
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            fig_t1 = px.bar(
                df_top5_peny, x='TOTAL_PENYALURAN', y='NAMA_KABKOT', orientation='h',
                text='LABEL',
                title="Top 5 Penyaluran (Rupiah)",
                labels={'TOTAL_PENYALURAN': '', 'NAMA_KABKOT': ''},
                color_discrete_sequence=['#1E3A8A'] # Warna Biru DJPb
            )
            fig_t1.update_traces(textposition='inside', textfont=dict(color='white', size=13))
            fig_t1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(showticklabels=False, showgrid=False) # Sembunyikan axis bawah agar bersih
            )
            st.plotly_chart(fig_t1, use_container_width=True)
            
        with col_t2:
            fig_t2 = px.bar(
                df_top5_deb, x='TOTAL_DEBITUR', y='NAMA_KABKOT', orientation='h',
                text='LABEL',
                title="Top 5 Jumlah Debitur Terbanyak",
                labels={'TOTAL_DEBITUR': '', 'NAMA_KABKOT': ''},
                color_discrete_sequence=['#F59E0B'] # Warna Oranye UMKM
            )
            fig_t2.update_traces(textposition='inside', textfont=dict(color='white', size=13))
            fig_t2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(showticklabels=False, showgrid=False)
            )
            st.plotly_chart(fig_t2, use_container_width=True)
    else:
        st.warning("Data kewilayahan tidak tersedia untuk rentang tahun ini.")
# ------------------------------------------
# TAB 2: TREN & KINERJA SEKTORAL
# ------------------------------------------
with tab2:
    st.markdown("#### Analisis Tren Waktu & Proporsi Sektor Usaha")
    
    if not df_master.empty:
        # Terapkan Filter Sidebar (Tanpa filter tahun agar tren terekam penuh)
        df_tab2 = df_master.copy()
        if pilih_prov != "SEMUA":
            df_tab2 = df_tab2[df_tab2['NAMA_PROVINSI'] == pilih_prov]
        if pilih_kabkot != "SEMUA":
            df_tab2 = df_tab2[df_tab2['NAMA_KABKOT'] == pilih_kabkot]

        st.markdown("---")
        col_tren, col_donut = st.columns([2, 1.5])
        
        with col_tren:
            st.markdown("**📈 Dinamika Penyaluran (Top 5 Sektor)**")
            
            # Agregasi data sektoral
            df_sektor = df_tab2.groupby(['TAHUN', 'SEKTOR_USAHA'])['TOTAL_PENYALURAN'].sum().reset_index()
            
            # Cari 5 sektor terbesar, sisanya jadikan 'LAINNYA' agar grafik rapi
            top_5_sektor = df_sektor.groupby('SEKTOR_USAHA')['TOTAL_PENYALURAN'].sum().nlargest(5).index
            df_sektor['SEKTOR_FINAL'] = np.where(df_sektor['SEKTOR_USAHA'].isin(top_5_sektor), df_sektor['SEKTOR_USAHA'], 'LAINNYA')
            df_tren_final = df_sektor.groupby(['TAHUN', 'SEKTOR_FINAL'])['TOTAL_PENYALURAN'].sum().reset_index()

            fig_tren = px.line(
                df_tren_final, x='TAHUN', y='TOTAL_PENYALURAN', color='SEKTOR_FINAL',
                markers=True, title="Tren Sektoral per Tahun (Rupiah)"
            )
            fig_tren.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(dtick=1, title="Tahun"),
                yaxis=dict(title="Total Penyaluran"),
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
                title="Persentase Distribusi Sektor"
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
            fig_donut.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.warning("Data belum tersedia. Pastikan file data berhasil dimuat.")

with tab3:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("#### 🚨 Anomali Inefisiensi Skema Pembiayaan (2018-2021)")
    st.info("Analisis ini memetakan riwayat efisiensi tiap Skema Kredit. Titik anomali (Kuadran Kiri Atas) adalah indikator: **Skema tertentu beban subsidi negara sangat besar per debitur, namun nominal pembiayaan yang disalurkan sangat kecil.**\nData merupakan irisan antara data Realisasi KUR dan Subsisi KUR sepanjang tahun 2018-2021")
    
    # Memanggil dataset khusus EWS yang sudah dikoreksi
    df_ews_master = load_ews_skema_data()
    
    if not df_ews_master.empty:
        # Filter berdasarkan navigasi Sidebar
        df_ews_filter = df_ews_master.copy()
        if pilih_prov != "SEMUA":
            df_ews_filter = df_ews_filter[df_ews_filter['NAMA_PROVINSI'] == pilih_prov]
        if pilih_kabkot != "SEMUA":
            df_ews_filter = df_ews_filter[df_ews_filter['NAMA_KABKOT'] == pilih_kabkot]
            
        if not df_ews_filter.empty:
            # --- PERBAIKAN: KONVERSI KE JUTAAN UNTUK SUMBU X & Y ---
            df_ews_filter['PENYALURAN_JUTA'] = df_ews_filter['PENYALURAN_PER_DEBITUR'] / 1e6
            df_ews_filter['SUBSIDI_JUTA'] = df_ews_filter['SUBSIDI_PER_DEBITUR'] / 1e6
            
            # Membuat Bubble Chart
            fig_ews = px.scatter(
                df_ews_filter, 
                x='PENYALURAN_JUTA', # Gunakan kolom skala jutaan
                y='SUBSIDI_JUTA',    # Gunakan kolom skala jutaan
                size='TOTAL_DEBITUR', 
                color='NAMA_SKEMA',
                color_discrete_map={
                    'MIKRO': '#10B981',         # Hijau
                    'KECIL': "#35ABE1",         # Biru
                    'SUPER MIKRO': "#E89910",   # Oranye
                    'TKI': "#FF0D00"            # Merah
                },
                hover_name='NAMA_SKEMA',
                hover_data={
                    'NAMA_SKEMA': False, 
                    'PENYALURAN_JUTA': False, # Sembunyikan angka jutaan dari pop-up
                    'SUBSIDI_JUTA': False,    # Sembunyikan angka jutaan dari pop-up
                    'TOTAL_DEBITUR': ':,',    # Format ribuan untuk orang
                    'PENYALURAN_PER_DEBITUR': ':,.0f', # Tampilkan nominal asli
                    'SUBSIDI_PER_DEBITUR': ':,.0f'     # Tampilkan nominal asli
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
                # Tambahkan suffix ' Juta' agar terbaca eksplisit tanpa singkatan ambigu
                xaxis=dict(showgrid=True, gridcolor='#E5E7EB', ticksuffix=' Juta'),
                yaxis=dict(showgrid=True, gridcolor='#E5E7EB', ticksuffix=' Juta'),
                height=550
            )
            
            # Menarik garis bantu (Rata-rata wilayah) dengan data skala jutaan
            avg_penyaluran = df_ews_filter['PENYALURAN_JUTA'].median()
            avg_subsidi = df_ews_filter['SUBSIDI_JUTA'].median()
            fig_ews.add_vline(x=avg_penyaluran, line_dash="dash", line_color="red", opacity=0.5)
            fig_ews.add_hline(y=avg_subsidi, line_dash="dash", line_color="red", opacity=0.5)
            
            st.plotly_chart(fig_ews, use_container_width=True)
            st.caption(f"Garis putus-putus merah menandakan median (nilai tengah) efisiensi wilayah {pilih_kabkot}. Titik yang berada di atas garis horizontal dan di kiri garis vertikal adalah skema dengan potensi inefisiensi anggaran.")
        else:
            st.warning("Data inefisiensi skema tidak tersedia untuk wilayah ini.")
    else:
        st.warning("File historis Subsidi KUR tidak ditemukan atau sedang diproses.")
    
    st.markdown("---")
    st.markdown("#### Proyeksi Beban Subsidi")
    st.info("**Holt-Winters** murni menggunakan data historis penyaluran untuk memproyeksikan beban subsidi di masa depan, sedangkan **SARIMAX** menggunakan pendekatan statistik yang lebih kompleks dengan menambahkan variabel exogenous yaitu BI Rate, PDRB, dan Realisasi KUR. Pilih mesin prediksi untuk melihat proyeksi beban subsidi berdasarkan data historis wilayah yang dipilih.")
    pilihan_model = st.radio(
        "🧠 Pilih Mesin Prediksi (Algoritma):", 
        ["Holt-Winters (Exponential Smoothing)", "SARIMAX"], 
        horizontal=True
    )
    
    # Indikator loading saat ML sedang bekerja memproses data historis
    with st.spinner("Mesin ML sedang melakukan kalkulasi multi-skenario in-memory..."):
        try:
            if "Holt-Winters" in pilihan_model:
                # Memanggil fungsi Holt-Winters dari modul eksternal
                df_geo = siapkan_data_geografis()
                df_hasil, fig_hasil = jalankan_prediksi_wilayah(df_geo, filter_provinsi=pilih_prov, filter_kabkot=pilih_kabkot)
            else:
                # Memanggil fungsi SARIMAX dari modul eksternal
                df_hasil, fig_hasil = jalankan_sarimax(filter_provinsi=pilih_prov, filter_kabkot=pilih_kabkot)
                
            # Jika mesin berhasil mengembalikan data dan grafik
            if df_hasil is not None and fig_hasil is not None:
                # Render grafik Matplotlib bawaan modul
                st.pyplot(fig_hasil)
                
                # Tampilkan tabel proyeksi di bawah grafik
                st.markdown("##### 🚨 Tabel Rincian Skenario Proyeksi Subsidi")
                st.dataframe(df_hasil, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Data historis tidak mencukupi atau kosong untuk kombinasi wilayah ini.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan sistematik pada mesin prediksi: {e}")

# ------------------------------------------
# TAB 4: POLICY SIMULATOR
# ------------------------------------------
# ------------------------------------------
# TAB 4: POLICY SIMULATOR
# ------------------------------------------
with tab4:
    st.markdown("#### ⚖️ Deterministik What-If Policy Simulator")
    st.info("Simulator kebijakan ini memproyeksikan **Dampak Inklusi Sosial (GEDSI)** apabila eksekutif mengambil keputusan untuk menaikkan atau menurunkan alokasi pagu penyaluran pada sektor ekonomi tertentu. Basis perhitungan menggunakan *conversion rate* historis.")
    
    if not df_master.empty:
        # 1. Menentukan Baseline Dinamis (Program -> Tahun)
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
                    st.warning("⚠️ Catatan: Data Realisasi KUR hanya tersedia hingga 2024. Simulasi untuk tahun 2025 ke atas murni menggunakan konversi data UMi.")

        if tahun_terakhir is not None:
            df_base = df_temp_year[df_temp_year['TAHUN'] == tahun_terakhir].copy()

            # Terapkan Filter Wilayah dari Sidebar
            if pilih_prov != "SEMUA":
                df_base = df_base[df_base['NAMA_PROVINSI'] == pilih_prov]
            if pilih_kabkot != "SEMUA":
                df_base = df_base[df_base['NAMA_KABKOT'] == pilih_kabkot]

            if not df_base.empty:
                # 2. Pemrosesan Sektor & Rasio Historis
                df_sektor = df_base.groupby('SEKTOR_USAHA').agg(
                    TOTAL_PENYALURAN=('TOTAL_PENYALURAN', 'sum'),
                    TOTAL_DEBITUR=('TOTAL_DEBITUR', 'sum'),
                    PEREMPUAN=('DEBITUR_PEREMPUAN', 'sum'),
                    SD_SMP=('PENDIDIKAN_SD_SMP', 'sum')
                ).reset_index()

                # Mengambil Top 4 Sektor Terbesar
                top_sektor = df_sektor.nlargest(4, 'TOTAL_PENYALURAN')['SEKTOR_USAHA'].tolist()
                df_sektor['SEKTOR_SIM'] = np.where(df_sektor['SEKTOR_USAHA'].isin(top_sektor), df_sektor['SEKTOR_USAHA'], 'Sektor Lainnya')
                
                df_sim_base = df_sektor.groupby('SEKTOR_SIM').sum().reset_index()

                # Menghitung Conversion Rate (Daya Ungkit per Rp 1 Miliar)
                df_sim_base['DEBITUR_PER_MILIAR'] = np.where(df_sim_base['TOTAL_PENYALURAN'] > 0, df_sim_base['TOTAL_DEBITUR'] / (df_sim_base['TOTAL_PENYALURAN'] / 1e9), 0)
                df_sim_base['PCT_PEREMPUAN'] = np.where(df_sim_base['TOTAL_DEBITUR'] > 0, df_sim_base['PEREMPUAN'] / df_sim_base['TOTAL_DEBITUR'], 0)
                df_sim_base['PCT_SD_SMP'] = np.where(df_sim_base['TOTAL_DEBITUR'] > 0, df_sim_base['SD_SMP'] / df_sim_base['TOTAL_DEBITUR'], 0)

                st.markdown("---")
                col_slider, col_result = st.columns([1, 2.2])

                # 3. Panel Interaktif Slider Sektoral
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

                # 4. Kalkulasi Mesin Simulasi & Visualisasi
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
    else:
        st.warning("Menunggu sinkronisasi data master...")