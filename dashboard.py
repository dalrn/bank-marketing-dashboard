import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from functions import segment_age2, segment_balance, segment_campaign2, segment_duration, segment_pdays2, segment_previous2,\
plot_segmented_variable, new_features, segment_age, segment_campaign, segment_previous, segment_pdays,\
preprocess_dataframe, replace_outliers_series, apply_transform_series, outlier_process
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Dashboard Academya',
                   page_icon=':bank:', layout='wide')

st.markdown(
    """
    <style>
        html, body, [class*="st-"] {
            font-family: "Futura", sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
df = pd.read_csv('bank-full.csv')
df = df.rename(columns={'y': 'Subscribed'})

# Capitalize the first letter of each column name
df.columns = [col.capitalize() for col in df.columns]

# Capitalize all values in the dataframe
df = df.applymap(lambda x: x.capitalize() if isinstance(x, str) else x)

st.markdown("""<h1 style='text-align: center; font-size: 35px; font-weight: bold; 
            color: #F56060; font-family: Futura, sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); margin-top: -50px;'>
            Analisis Performa Pemasaran Bank</h1>
            <h3 style="text-align: center; font-weight: normal; font-size: 20px; color: gray;">Siapa yang Berlangganan dan Mengapa?</h3>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True) 

# ------------------------------ SIDEBAR -----------------------------

st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebarContent"] {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 10px;
        align-items: center;
        text-align: center;
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 40px; /* Increased padding */
    }

    .sidebar-button {
        display: block;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        color: #CCCCCC;
        background-color: transparent;
        border: none;
        text-align: center;
        padding: 12px 0; /* Increased height */
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 10px auto;
    }

    .sidebar-button:hover {
        background-color: #333333;
        color: #F56060;
    }

    .sidebar-button.active {
        background-color: #F56060;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-title">Navigasi</div>', unsafe_allow_html=True)

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Dataset Explorer"  # Default active page

# Sidebar buttons for navigation
if st.sidebar.button("📊 Dataset Explorer", key="explorer"):
    st.session_state.page = "Dataset Explorer"

if st.sidebar.button("💡 Di Balik Data", key="story"):
    st.session_state.page = "Di Balik Data"

if st.sidebar.button("🤖 Model Prediksi", key="predictive"):
    st.session_state.page = "Model Prediksi"

st.sidebar.markdown(
    f"""
    <script>
        var buttons = window.parent.document.querySelectorAll('.sidebar-button');
        buttons.forEach(btn => btn.classList.remove('active'));
        buttons[{["Dataset Explorer", "Di Balik Data", "Model Prediksi"].index(st.session_state.page)}].classList.add('active');
    </script>
    """,
    unsafe_allow_html=True
)

# "About the Dataset" Dropdown
with st.sidebar.expander("Tentang Dataset"):

    st.markdown("""<div style='text-align: justify;'>
        Dataset ini berisi catatan kampanye pemasaran langsung dari sebuah institusi perbankan di Portugal dari Mei 2008 hingga November 2010.
        Produk yang ditawarkan adalah deposito berjangka, dan kampanye pemasaran dilakukan melalui panggilan telepon.
    </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("""<div style='text-align: justify;'>
        Deskripsi lebih lanjut mengenai dataset dan setiap variabelnya dapat ditemukan di 
        <a href="https://pastebin.com/B6b8qRgB" target="_blank">sini</a>.
    </div>""", unsafe_allow_html=True)


# ------------------------------ PAGE 1: DATASET EXPLORER ------------------------------

if st.session_state.page == "Dataset Explorer":

    col1, _, col2 = st.columns([0.8, 0.05, 2])
    
    with col1.container(height=820):
        st.markdown("<div style='color: #FF4B4B;font-size: 27px;font-weight:bold;margin-bottom: 15px;'>Filter</div>", unsafe_allow_html=True)

        # Toggle to show/hide missing values
        show_missing = st.checkbox("Sembunyikan nilai hilang", value=False)
        
        # Categorical Filters
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        selected_cat_var = st.multiselect("Filter berdasarkan (kategori): ", categorical_cols, default=None, placeholder='Pilih opsi')
        
        selected_values = {}

        # Generate checkboxes for each selected categorical variable
        for cat in selected_cat_var:
            unique_vals = df[cat].dropna().unique().tolist()
            
            if cat == 'Month':
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                unique_vals = sorted(unique_vals, key=lambda x: month_order.index(x))

            st.markdown(f"Select **{cat}**:") 

            col1_1, col1_2 = st.columns(2)
            selected_values[cat] = []

            # Loop through categories and assign checkboxes to columns
            for i, val in enumerate(unique_vals):
                with [col1_1, col1_2][i % 2]:  # Rotate through columns
                    if st.checkbox(val, key=f"{cat}_{val}", value=False):
                        selected_values[cat].append(val)

            # Apply filtering
            for cat, selected_vals in selected_values.items():
                if selected_vals:  # Only apply filter if values are selected
                    df = df[df[cat].isin(selected_vals)]
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add space between filters and columns to display
        
        # Numerical Filters
        numerical_cols = df.select_dtypes(include='number').columns.tolist()
        selected_num_vars = st.multiselect("Filter berdasarkan (numerik):", numerical_cols, default=None, placeholder='Pilih opsi')

        for selected_num_var in selected_num_vars:
            col1_3, col2_3 = st.columns(2)

            # Default min/max values
            min_val, max_val = float(df[selected_num_var].min()), float(df[selected_num_var].max())

            # Session state keys
            min_key, max_key = f"min_{selected_num_var}", f"max_{selected_num_var}"

            # Initialize session state if not exists
            if min_key not in st.session_state:
                st.session_state[min_key] = min_val
            if max_key not in st.session_state:
                st.session_state[max_key] = max_val

            # Manual text input
            with col1_3:
                manual_min = st.text_input(f"Min {selected_num_var}", value=str(st.session_state[min_key]))
            with col2_3:
                manual_max = st.text_input(f"Max {selected_num_var}", value=str(st.session_state[max_key]))

            # Convert text inputs to float
            try:
                selected_min = float(manual_min)
                selected_max = float(manual_max)
                if selected_min > selected_max:
                    raise ValueError("Min cannot be greater than Max")
            except ValueError:
                st.warning(f"Invalid input for {selected_num_var}. Using previous values.")
                selected_min = st.session_state[min_key]
                selected_max = st.session_state[max_key]

            # Slider with session state values
            selected_min, selected_max = st.slider(
                f"Select Range for {selected_num_var}",
                min_val, max_val,
                (selected_min, selected_max)
            )

            # Update session state
            st.session_state[min_key] = selected_min
            st.session_state[max_key] = selected_max

            # Apply filter
            df = df[(df[selected_num_var] >= selected_min) & (df[selected_num_var] <= selected_max)]
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add space between filters and columns to display

        # Sort by
        st.markdown("<div style='color: #FF4B4B;font-size: 27px;font-weight:bold;margin-bottom: 15px;'>Sort</div>", unsafe_allow_html=True)

        selected_sort_vars = st.multiselect("Urutkan berdasarkan:", df.columns, default=None, placeholder='Pilih opsi')

        sort_order = st.radio("Urutan", ["Menaik", "Menurun"], horizontal=True)

        if selected_sort_vars:
            df = df.sort_values(by=selected_sort_vars, ascending=(sort_order == "Menaik"))

    with col2:

        # Display Dataset
        if show_missing:
            df = df.dropna()
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.markdown(f"<div style='text-align: right; font-size: 14px'>Menunjukkan <strong style='color: #F56060;'>{len(df):,}</strong> catatan pemasaran</div>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        sub_col1, _, sub_col2 = st.columns([1.1, 0.01, 0.8])
        
        with sub_col1:
            st.markdown("### Tingkat Konversi")
            st.markdown("<br>", unsafe_allow_html=True)

            # Calculate the percentage of Y = "Yes"
            yes_count = df['Subscribed'].value_counts().get('Yes', 0)
            no_count = df['Subscribed'].value_counts().get('No', 0)
            yes_percentage = round((yes_count/(yes_count+no_count))*100, 1)

            if yes_percentage <= 30:
                yes_color = 'red'
            elif yes_percentage >= 70:
                yes_color = '#52F257'
            else:
                yes_color = '#FFD700'

            # Create a donut chart using Altair
            source = pd.DataFrame({'Berlangganan': ['Ya', 'Tidak'], 'Jumlah': [yes_count, no_count], '%': [yes_percentage, 100-yes_percentage]})  
            source['Legend_Label'] = source.apply(lambda x: f"{x['Berlangganan']} ({x['Jumlah']})", axis=1)

            chart = alt.Chart(source).mark_arc(innerRadius=70, outerRadius=90).encode(
                theta=alt.Theta(field="Jumlah", type="quantitative"),
                color=alt.Color(field="Legend_Label", type="nominal",
                                scale=alt.Scale(domain=source["Legend_Label"].tolist(), range=[yes_color, '#D3D3D3']),
                                legend=alt.Legend(title="Berlangganan (Jumlah)")),
                tooltip=["Berlangganan", "Jumlah", "%"]
            ).properties(
                width='container',
                height=250
            )

            text = alt.Chart(pd.DataFrame({'text': [f"{yes_percentage}%"], 'tooltip': [f"Persentase: {yes_percentage}%"]})).mark_text(
                size=30, fontWeight='bold', color=yes_color
            ).encode(
                text='text:N',
                tooltip='tooltip:N'
            ).properties(
                width='container',
                height=250
            )

            st.altair_chart(chart+text)
        
        # Donut Chart Description
        with sub_col2:
            st.markdown(f"""  
                #### Dari <span style='color:{yes_color};'>{yes_percentage}%</span> yang berlangganan...  
            """, unsafe_allow_html=True)     

            df_yes = df[df['Subscribed'] == 'Yes']  

            def display_percentage(df, column):
                most_common = df[column].value_counts().idxmax().lower()
                count = df[column].value_counts().max()
                percentage = round((count / len(df)) * 100)

                if column == "Education":
                    if most_common == "primary":
                        description = "sedang menduduki <span style='color: #F56060;'>SD/sederajat</span>"
                    elif most_common == "secondary":
                        description = "sedang menduduki <span style='color: #F56060;'>SMP/SMA</span>"
                    elif most_common == "tertiary":
                        description = "sedang menempuh <span style='color: #F56060;'>pendidikan lanjut</span>"
                    elif most_common == "unknown":
                        description = "<span style='color: #F56060;'>tidak diketahui </span>jenjang pendidikannya"

                if column == "Contact":
                    if most_common == "cellular":
                        description = "dihubungi melalui <span style='color: #F56060;'>telepon genggam</span>"
                    elif most_common == "telephone":
                        description = "dihubungi melalui <span style='color: #F56060;'>telepon kabel</span>"
                    elif most_common == "unknown":
                        description = "<span style='color: #F56060;'>tidak diketahui</span> metode kontaknya"

                if column == "Marital":
                    if most_common == "married":
                        description = "sudah <span style='color: #F56060;'>menikah</span>"
                    elif most_common == "single":
                        description = "adalah <span style='color: #F56060;'>lajang</span>"
                    elif most_common == "divorced":
                        description = "sudah <span style='color: #F56060;'>cerai</span> dan belum menikah lagi"

                else:
                    description = description.format(f"<span style='color: #F56060;'>{most_common}</span>")

                st.markdown(
                    f"""
                    <div style="font-size: 35px; font-weight: bold; color: #F56060; display: inline;">{percentage}%</div>
                    <div style="font-size: 15px; font-weight: bold; display: inline;"> {description}</div>
                    <div style="font-size: 11px;">({count} orang)</div>
                    """,
                    unsafe_allow_html=True
                )

            display_percentage(df_yes, "Contact")
            display_percentage(df_yes, "Marital")
            display_percentage(df_yes, "Education")

    # Data Visualizations
    st.markdown("<br>", unsafe_allow_html=True)
    col5_1, _, col5_2 = st.columns([0.3, 0.01, 1])
    with col5_1:
        st.markdown("<div style='color: #FF4B4B;font-size: 27px;font-weight:bold;margin-bottom: 10px;'>Distribusi Data</div>", unsafe_allow_html=True)
    with col5_2:
        show_successful = st.checkbox("Berlangganan")

    col_left, col_right = st.columns(2)

    with col_left.container(height=650):
        numeric_vars = ["Age", "Balance", "Duration", "Day", "Campaign", "Pdays", "Previous"]
        figs = {}
        for col in numeric_vars:
            fig = px.histogram(df, x=col, nbins=30, opacity=0.7, color_discrete_sequence=["#FF4500"], barmode='overlay')
            if show_successful:
                df_successful = df[df['Subscribed'] == 'Yes']
                fig.add_trace(px.histogram(df_successful, x=col, nbins=30, opacity=0.7, color_discrete_sequence=[yes_color], barmode='overlay').data[0])
            fig.update_traces(marker_line_width=0.5, marker_line_color='black')
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',  
                paper_bgcolor='rgba(0,0,0,0)',
                title=f"{col}",
                title_font=dict(size=16, color='#FAFAFA'),
                xaxis_title=col,
                yaxis_title="Frequency",
                xaxis=dict(title_font=dict(size=14, color="gray")),
                yaxis=dict(title_font=dict(size=14, color="gray"))
            )
            figs[col] = fig

        for i, col in enumerate(numeric_vars):
            if i % 2 == 0:
                col_left_1, col_left_2 = st.columns(2)
            with col_left_1 if i % 2 == 0 else col_left_2:
                st.plotly_chart(figs[col], use_container_width=True)

    with col_right.container(height=650):
        categoric_vars = [
            ("Job",), 
            ("Month",), 
            ("Contact", "Marital"), 
            ("Education", "Poutcome"), 
            ("Housing", "Loan"), 
            ("Default", "Subscribed")
        ]
        # month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for row in categoric_vars:
            cols = st.columns(len(row))
            for col, category in zip(cols, row):
                counts = df[category].value_counts().reset_index()
                counts.columns = [category, 'Count']
                # if category == "Month":
                #     counts[category] = pd.Categorical(counts[category], categories=month_order, ordered=True)
                #     counts = counts.sort_values(category)
                fig = px.bar(counts, x=category, y='Count', opacity=0.7, color_discrete_sequence=["#B71C1C"], barmode='group')
                if show_successful:
                    counts_successful = df[df['Subscribed'] == 'Yes'][category].value_counts().reset_index()
                    counts_successful.columns = [category, 'Count']
                    # if category == "Month":
                    #     counts_successful[category] = pd.Categorical(counts_successful[category], categories=month_order, ordered=True)
                    #     counts_successful = counts_successful.sort_values(category)
                    fig.add_trace(px.bar(counts_successful, x=category, y='Count', opacity=0.7, color_discrete_sequence=[yes_color], barmode='group').data[0])
                fig.update_traces(marker_line_width=0.5, marker_line_color='black')
                fig.update_layout(
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',  
                    paper_bgcolor='rgba(0,0,0,0)',
                    title=category,
                    title_font=dict(size=16, color='#FAFAFA'),
                    xaxis_title=category,
                    yaxis_title="Count",
                    xaxis=dict(title_font=dict(size=14, color="gray")),
                    yaxis=dict(title_font=dict(size=14, color="gray"), tickformat=',d')
                )
                col.plotly_chart(fig, use_container_width=True)


# ------------------------------ PAGE 2: DI BALIK DATA ------------------------------

elif st.session_state.page == "Di Balik Data":
    df2 = pd.read_csv('bank-full.csv')
    df2 = df2.rename(columns={'y': 'Subscribed'})
    
    categorical_cols = df2.select_dtypes(include='object')
    numerical_cols = df2.select_dtypes(include='number')

    total_calls = len(df2)
    success_rate = df2['Subscribed'].value_counts(normalize=True).get(1, 0) * 100
    avg_duration = df2['duration'].mean()

    # Title
    st.markdown('<p style="margin-bottom:10px; font-size:18px; text-align:center;">📊 Data historis menunjukkan bahwa...</p>', unsafe_allow_html=True)

    st.markdown(f"""
        <div style="border-radius:10px; padding:15px; background-color:#1E1E1E; text-align:center;">
            <p style="font-size:30px; color:white; font-weight:bold;">
                Dari <span style="color:#FF4B4B; font-size:42px;">{total_calls}</span> usaha pemasaran deposito,
                hanya <span style="color:#FF4B4B; font-size:42px;">{success_rate:.2f}%</span> berhasil memenangkan klien baru.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style="font-size:18px; text-align:center; margin-top:15px;">
            Untuk meningkatkan keberhasilan pemasaran berikutnya, perlu diidentifikasi 
            <span style="color:#FF4B4B; font-weight:bold;">karakteristik klien yang paling mungkin berlangganan</span> 
            serta diterapkan 
            <span style="color:#FF4B4B; font-weight:bold;">strategi pemasaran yang tepat</span>.
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border: 2px solid #4B9EFF; padding: 10px; border-radius: 10px; background-color: #E8F4FF; text-align: left;">
        <h4 style="color: black;font-size: 20px;">🎯 1. Klien seperti apa yang paling mungkin berlangganan?</h4>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1_2, _, col2_2 = st.columns([1.2,0.1,1])

    # Section 1
    with col1_2.container(height=650):

        st.markdown("<div style='color: #fafafa;font-size: 27px;font-weight:bold;margin-bottom: 15px;'>Segmentasi Klien:<br>Distribusi dan Tingkat Kesuksesannya</div>", unsafe_allow_html=True)
        duration_median = df2['duration'].median()
        balance_median = df2['balance'].median()

        df2['Kelompok Umur'] = df2['age'].apply(segment_age2)
        df2['Durasi Telepon'] = df2['duration'].apply(lambda x: segment_duration(x, duration_median))
        df2['Saldo Bank'] = df2['balance'].apply(lambda x: segment_balance(x, balance_median))
        df2['Berapa Kali Dihubungi (kampanye ini)'] = df2['campaign'].apply(segment_campaign2)
        df2['Berapa Kali Dihubungi (kampanye lalu)'] = df2['previous'].apply(segment_previous2)
        df2['Berapa Lama Sejak Terakhir Dihubungi'] = df2['pdays'].apply(segment_pdays2)

        # List of segmented variables
        segment_vars = [
            "Kelompok Umur", "Durasi Telepon", "Saldo Bank",
            "Berapa Kali Dihubungi (kampanye ini)", "Berapa Kali Dihubungi (kampanye lalu)", 
            "Berapa Lama Sejak Terakhir Dihubungi"
        ]

        # Generate visualizations for each segmented variable
        for segment in segment_vars:
            plot_segmented_variable(df2, segment, 420)

    with col2_2:
        st.markdown(
            """
            <style>
                .highlight-box {
                    background-color: rgba(255, 255, 255, 0.1);
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 2px 2px 10px rgba(255, 255, 255, 0.1);
                    margin-bottom: 20px;
                }
                .highlight-text {
                    color: #00C8FF; /* Warna skyblue */
                    font-weight: bold;
                }
                .header-title {
                    color: #fafafa;
                    font-size: 34px;
                    font-weight: bold;
                    text-align: left;
                    font-family: 'Poppins', sans-serif;
                    margin-bottom: 15px;
                }
                .content-text {
                    text-align: justify;
                    font-size: 16px;
                    line-height: 1.6;
                }
            </style>
            
            <div class="header-title">Terungkap bahwa...</div>
            
            <div class="highlight-box">
                <p class="content-text">
                    Kebanyakan klien <span class="highlight-text">tidak akan setuju berlangganan pada kontak pertama</span>. 
                    Kemungkinan berhasil memenangkan klien yang belum pernah dihubungi sebelumnya sangatlah rendah. Namun, jangan menyerah! 
                    Karena biasanya klien yang sudah dihubungi <span class="highlight-text">1-4 kali</span> di kampanye ini (apa pun hasilnya), 
                    pada kampanye-kampanye berikutnya <span class="highlight-text">jauh lebih mungkin</span> untuk menerima tawaran, bahkan langsung pada kontak pertamanya.
                </p>
                <p class="content-text">
                    Meski demikian, jika penjual tidak kunjung menghubungi kembali seorang klien hingga <span class="highlight-text">melebihi 3 bulan</span>, 
                    ketertarikan klien tetap dapat menurun seiring waktu. Seandainya pada penghubungan kembali masih saja ditolak, 
                    <span class="highlight-text">jangan menghubungi secara berlebihan</span>. Menghubungi klien hingga <span class="highlight-text">5 kali</span> dalam satu kampanye atau lebih terbukti 
                    malah <span class="highlight-text">menurunkan ketertarikan</span>.
                </p>
                <p class="content-text">
                    Dalam memilih klien, <span class="highlight-text">usia</span> dan <span class="highlight-text">saldo bank</span> layak dipertimbangkan. 
                    Klien berusia <span class="highlight-text">lebih muda</span> (di bawah 33 tahun) cenderung lebih tertarik dengan tawaran deposito. 
                    Hal ini sepertinya disebabkan masih tingginya motivasi untuk menabung dan mencari stabilitas finansial serta pengalaman investasi. 
                    Klien-klien dengan <span class="highlight-text">uang lebih banyak</span> memiliki dana yang cukup untuk diinvestasikan tanpa mengganggu kebutuhannya, 
                    membuat mereka lebih tertarik pada produk yang ditawarkan.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Section 2 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border: 2px solid #ff8000; padding: 10px; border-radius: 10px; background-color: #FFD8B0; text-align: left;">
        <h4 style="color: black;font-size: 20px;">⌚ 2. Bagaimana cara meningkatkan durasi telepon?</h4>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col3_1, _, col3_2 = st.columns([0.6, 0.01, 1])

    with col3_1:
        plot_segmented_variable(df2, "Durasi Telepon", 320, key='plot2')

    with col3_2:
        st.markdown(
            """
            <div style='color: #fafafa; font-size: 31px; font-weight: bold; text-align: right; display: flex; align-items: center; margin-bottom:20px;'>
                Kenapa perlu meningkatkan durasi?
            </div>
            """,
            unsafe_allow_html=True
        )        
        st.markdown(
            """
            <div style='text-align: justify'> 
                Terlihat di visualisasi sebelumnya bahwa <span style="color:#ffb266;">semakin lama durasi telepon, semakin mungkin klien berlangganan</span>. Ini masuk akal, karena keputusan untuk berlangganan
                pasti diawali dengan ketertarikan lebih lanjut terhadap produk, yang cenderung membuat sesi telepon lebih lama.
                <br><br>
                Namun dalam situasi nyata,
                <span style="color:#ffb266;">informasi durasi kontak tidak akan tersedia</span> sebelum kontak itu dilaksanakan, dan sulit memastikan percakapan akan berjalan berapa lama.
                Karena itu, pada bagian selanjutnya diselidiki apa saja <span style="color:#ffb266;">hal yang dapat dilakukan untuk meningkatkan durasi</span>, sehingga penjual dapat
                menghubungi klien pada kondisi yang tepat.
            </div>
            """,
            unsafe_allow_html=True
        )

    box1, _, box2 = st.columns([1,0.1,1])
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    box3, _, box4 = st.columns([1,0.1,1])

    # First row
    with box1:
        st.markdown("""
            <div style="padding:10px; border-radius:10px; background-color:#1E1E1E; text-align:center;">
                <span style="color:#FF9933; font-size:24px; font-weight:bold;">Durasi Kontak: Awal vs. Akhir Bulan</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style="font-size:18px; color:#FAFAFA; text-align:center;margin-top:8px;">
                ⏳ <b>Kontak di awal bulan berlangsung lebih lama!</b>
            </p>
        """, unsafe_allow_html=True)

        # Metric Boxes
        box1_1, box1_2 = st.columns(2)

        with box1_1:
            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#222222; text-align:center;">
                    <p style="font-size:16px; color:#FFCC66;"><b>Durasi Rata-Rata (Tanggal 1-15)</b></p>
                    <p style="font-size:32px; color:#FAFAFA; font-weight:bold;">359.92 detik</p>
                    <p style="color:#66FF66; font-size:14px;">⬆ +13.21 detik</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#222222; text-align:center;">
                    <p style="font-size:16px; color:#FFCC66;"><b>Durasi Rata-Rata (Tanggal 16-31)</b></p>
                    <p style="font-size:32px; color:#FAFAFA; font-weight:bold;">346.71 detik</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        with box1_2:
            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#333333;">
                    <p style="font-size:14px; color:#FAFAFA;">
                        <b>📌 Note:</b> Perbedaan bukan kebetulan! <span style="color:#66FF66;">(signifikan secara statistik ✅)</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("📝 Detail uji statistik"):
                st.markdown("""
                    <p style="font-size:13px;">
                    - <b>Nama Uji:</b> Mann-Whitney U<br>
                    - <b>H<sub>0</sub>:</b> Durasi rata-rata awal bulan = durasi rata-rata akhir bulan<br>
                    - <b>P-Value:</b> 0.0000<br>
                    - <b>Tingkat Kepercayaan:</b> 95%<br>
                    - <b>Kesimpulan:</b> H<sub>0</sub> ditolak ✅<br>
                    </p>
                """, unsafe_allow_html=True)

        st.markdown("""
            <p style="font-size:14px; color:#FAFAFA; text-align:center;">
                Klien lebih terbuka untuk percakapan lebih lama di awal bulan. Orang-orang cenderung memiliki lebih banyak fleksibilitas
                    finansial dan mental karena <b>gaji baru saja diterima</b>, serta <b>beban pekerjaan mungkin masih lebih ringan</b> dibanding akhir bulan.
        """, unsafe_allow_html=True)
        
        # Call-to-Action Box
        st.markdown("""
            <div style="border: 2px solid #FFB266; padding: 15px; border-radius: 10px; background-color: #1E1E1E; text-align: center;">
                <p style="color:#FAFAFA; font-weight:bold; font-size:16px; margin:0;">
                    💡 <b>Optimalkan strategi pemasaran dan penjualan di</b> <span style="color:#FFB266;">awal bulan</span> 
                    <b>untuk meningkatkan konversi</b> 💡
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with box2:
        st.markdown("""
            <div style="padding:10px; border-radius:10px; background-color:#1E1E1E; text-align:center;">
                <span style="color:#FF9933; font-size:24px; font-weight:bold;">Durasi Kontak: Awal vs. Akhir Tahun</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style="font-size:18px; color:#FAFAFA; text-align:center;margin-top:8px;">
                📅 <b> Kontak di akhir tahun berlangsung lebih lama!</b>
            </p>
        """, unsafe_allow_html=True)

        # Metric Boxes
        box2_1, box2_2 = st.columns(2)

        with box2_1:
            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#222222; text-align:center;">
                    <p style="font-size:16px; color:#FFCC66;"><b>Durasi Rata-Rata (Bulan Januari-Juni)</b></p>
                    <p style="font-size:32px; color:#FAFAFA; font-weight:bold;">345.31 detik</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#222222; text-align:center;">
                    <p style="font-size:16px; color:#FFCC66;"><b>Durasi Rata-Rata (Bulan Juli-Desember)</b></p>
                    <p style="font-size:32px; color:#FAFAFA; font-weight:bold;">364.34 detik</p>
                    <p style="color:#66FF66; font-size:14px;">⬆ +19.03 detik</p>

                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        with box2_2:
            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#333333;">
                    <p style="font-size:14px; color:#FAFAFA;">
                        <b>📌 Note:</b> Perbedaan bukan kebetulan! <span style="color:#66FF66;">(signifikan secara statistik ✅)</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("📝 Detail uji statistik"):
                st.markdown("""
                    <p style="font-size:13px;">
                    - <b>Nama Uji:</b> Mann-Whitney U<br>
                    - <b>H<sub>0</sub>:</b> Durasi rata-rata awal tahun = durasi rata-rata akhir tahun<br>
                    - <b>P-Value:</b> 3.14 × 10⁻⁶<br>
                    - <b>Tingkat Kepercayaan:</b> 95%<br>
                    - <b>Kesimpulan:</b> H<sub>0</sub> ditolak ✅<br>
                    </p>
                """, unsafe_allow_html=True)

        st.markdown("""
            <p style="font-size:14px; color:#FAFAFA; text-align:center;">
                Klien mungkin lebih reflektif terhadap <b>perencanaan ke depan</b>, terutama jika terkait dengan resolusi tahun baru
                    atau <b>anggaran yang masih tersedia</b> di akhir tahun.
        """, unsafe_allow_html=True)
        
        # Call-to-Action Box
        st.markdown("""
            <div style="border: 2px solid #FFB266; padding: 15px; border-radius: 10px; background-color: #1E1E1E; text-align: center;">
                <p style="color:#FAFAFA; font-weight:bold; font-size:16px; margin:0;">
                    💡 <b>Tingkatkan intensitas kampanye dan follow-up di</b> <span style="color:#FFB266;">akhir tahun</span> 
                    <b>untuk meningkatkan konversi</b> 💡
                </p>
            </div>
        """, unsafe_allow_html=True)


    # Second row
    with box3:
        st.markdown("""
            <div style="padding:10px; border-radius:10px; background-color:#1E1E1E; text-align:center;">
                <span style="color:#FF9933; font-size:24px; font-weight:bold;">Durasi Kontak: Hasil Kampanye Sebelum Ini</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style="font-size:18px; color:#FAFAFA; text-align:center;margin-top:8px;">
                🤷 <b> Kontak dengan klien yang keputusan terakhirnya belum jelas berlangsung lebih lama!</b>
            </p>
        """, unsafe_allow_html=True)

        box3_1, box3_2 = st.columns(2)

        with box3_1:
            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#222222; text-align:center;">
                    <p style="font-size:16px; color:#FFCC66;"><b>Durasi Rata-Rata (Menerima)</b></p>
                    <p style="font-size:32px; color:#FAFAFA; font-weight:bold;">305.01 detik</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#222222; text-align:center;">
                    <p style="font-size:16px; color:#FFCC66;"><b>Durasi Rata-Rata (Menolak)</b></p>
                    <p style="font-size:32px; color:#FAFAFA; font-weight:bold;">326.22 detik</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
                <p style="font-size:14px; color:#FAFAFA; text-align:center;">
                        Karena promosi sebelumnya tidak berujung sukses maupun gagal, ada kemungkinan klien masih memiliki <b>ketertarikan, pengetahuan, atau kebingungan</b> yang
                        membuat mereka lebih terbuka untuk diskusi lebih lama dan mendalam. Ini bisa menjadi <b>peluang untuk memberikan informasi tambahan</b> atau strategi
                        pendekatan yang lebih meyakinkan.    
                    """, unsafe_allow_html=True)


        with box3_2:
            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#222222; text-align:center;">
                    <p style="font-size:16px; color:#FFCC66;"><b>Durasi Rata-Rata (Lainnya)</b></p>
                    <p style="font-size:32px; color:#FAFAFA; font-weight:bold;">349.72 detik</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#333333;">
                    <p style="font-size:14px; color:#FAFAFA;">
                        <b>📌 Note:</b> Perbedaan bukan kebetulan! <span style="color:#66FF66;">(signifikan secara statistik ✅)</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("📝 Detail uji statistik"):
                st.markdown("""
                    <p style="font-size:13px;">
                    - <b>Nama Uji:</b> Kruskal-Wallis<br>
                    - <b>H<sub>0</sub>:</b> Tidak ada perbedaan durasi rata-rata untuk klien yang pada kampanye sebelumnya menerima, menolak, maupun lainnya.<br>
                    - <b>P-Value:</b> 2.39 × 10⁻¹⁷<br>
                    - <b>Tingkat Kepercayaan:</b> 95%<br>
                    - <b>Kesimpulan:</b> H<sub>0</sub> ditolak ✅<br>
                    </p>
                """, unsafe_allow_html=True)
        
        # Call-to-Action Box
        st.markdown("""
            <div style="border: 2px solid #FFB266; padding: 15px; border-radius: 10px; background-color: #1E1E1E; text-align: center;">
                <p style="color:#FAFAFA; font-weight:bold; font-size:16px; margin:0;">
                    💡 <b>Usahakan follow-up kembali klien yang </b> <span style="color:#FFB266;">belum jelas keputusannya</span> 
                    <b> sebelum kampanye ini</b> 💡
                </p>
            </div>
        """, unsafe_allow_html=True)


    with box4:
        st.markdown("""
            <div style="padding:10px; border-radius:10px; background-color:#1E1E1E; text-align:center;">
                <span style="color:#FF9933; font-size:24px; font-weight:bold;">Durasi Kontak: Sudah vs. Belum Pernah Dihubungi</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <p style="font-size:18px; color:#FAFAFA; text-align:center;margin-top:8px;">
                🤝 <b>Kontak dengan klien baru berlangsung lebih lama!</b>
            </p>
        """, unsafe_allow_html=True)

        # Metric Boxes
        box4_1, box4_2 = st.columns(2)

        with box4_1:
            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#222222; text-align:center;">
                    <p style="font-size:16px; color:#FFCC66;"><b>Durasi Rata-Rata (Klien Baru)</b></p>
                    <p style="font-size:32px; color:#FAFAFA; font-weight:bold;">359.74 detik</p>
                    <p style="color:#66FF66; font-size:14px;">⬆ +26.93 detik</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#222222; text-align:center;">
                    <p style="font-size:16px; color:#FFCC66;"><b>Durasi Rata-Rata (Klien Lama)</b></p>
                    <p style="font-size:32px; color:#FAFAFA; font-weight:bold;">332.81 detik</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        with box4_2:
            st.markdown("""
                <div style="border-radius:10px; padding:10px; background-color:#333333;">
                    <p style="font-size:14px; color:#FAFAFA;">
                        <b>📌 Note:</b> Perbedaan bukan kebetulan! <span style="color:#66FF66;">(signifikan secara statistik ✅)</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("📝 Detail uji statistik"):
                st.markdown("""
                    <p style="font-size:13px;">
                    - <b>Nama Uji:</b> Mann-Whitney U<br>
                    - <b>H<sub>0</sub>:</b> Durasi rata-rata dengan klien baru = durasi rata-rata dengan klien lama<br>
                    - <b>P-Value:</b> 1.07 × 10⁻¹⁴<br>
                    - <b>Tingkat Kepercayaan:</b> 95%<br>
                    - <b>Kesimpulan:</b> H<sub>0</sub> ditolak ✅<br>
                    </p>
                """, unsafe_allow_html=True)

        st.markdown("""
            <p style="font-size:14px; color:#FAFAFA; text-align:center;">
                Klien baru mungkin <b>belum memiliki pengalaman dikontak sebelumnya</b> dan perlu memahami lebih lanjut, sehingga lebih bersedia meluangkan waktu.
        """, unsafe_allow_html=True)

        # Call-to-Action Box
        st.markdown("""
            <div style="border: 2px solid #FFB266; padding: 15px; border-radius: 10px; background-color: #1E1E1E; text-align: center;">
                <p style="color:#FAFAFA; font-weight:bold; font-size:16px; margin:0;">
                    💡 <b>Perbanyak menghubungi </b> <span style="color:#FFB266;">klien baru</span> 
                    <b>dan usahakan panggilan pertama efektif</b> 💡
                </p>
            </div>
        """, unsafe_allow_html=True)


# ------------------------------ PAGE 3: MODEL PREDIKSI ------------------------------

elif st.session_state.page == "Model Prediksi":

    pipeline_prep = joblib.load('pipeline_prep.pkl')
    pipeline_trans = joblib.load('pipeline_trans.pkl')
    final_model = joblib.load('final_model.pkl')

    st.title("Akankah Klien Berlangganan Deposito Berjangka?")

    st.subheader("Masukkan informasi klien dan kontak:")

    col1, col2 = st.columns(2)

    # Pemetaan dari bahasa Indonesia ke bahasa Inggris
    job_mapping = {
        "Administrator": "admin.", "Pekerja Kasar": "blue-collar", "Pengusaha": "entrepreneur",
        "Pembantu Rumah Tangga": "housemaid", "Manajemen": "management", "Pensiunan": "retired",
        "Wirausaha": "self-employed", "Pelayanan": "services", "Mahasiswa": "student",
        "Teknisi": "technician", "Pengangguran": "unemployed", "Tidak Diketahui": "unknown"
    }

    marital_mapping = {
        "Menikah": "married", "Lajang": "single", "Cerai": "divorced"
    }

    education_mapping = {
        "SD/Sederajat": "primary", "SMP/SMA": "secondary", "Perguruan Tinggi": "tertiary", "Tidak Diketahui": "unknown"
    }

    default_mapping = {"Ya": "yes", "Tidak": "no"}
    housing_mapping = {"Ya": "yes", "Tidak": "no"}
    loan_mapping = {"Ya": "yes", "Tidak": "no"}
    contact_mapping = {"Seluler": "cellular", "Telepon": "telephone", "Tidak Diketahui": "unknown"}
    month_mapping = {
        "Januari": "jan", "Februari": "feb", "Maret": "mar", "April": "apr", "Mei": "may", "Juni": "jun",
        "Juli": "jul", "Agustus": "aug", "September": "sep", "Oktober": "oct", "November": "nov", "Desember": "dec"
    }
    poutcome_mapping = {"Gagal": "failure", "Tidak Diketahui": "unknown", "Sukses": "success"}

    with col1:
        age = st.number_input("Umur", min_value=18, max_value=100, step=1)
        job = st.selectbox("Pekerjaan", list(job_mapping.keys()))
        marital = st.selectbox("Status Pernikahan", list(marital_mapping.keys()))
        education = st.selectbox("Pendidikan", list(education_mapping.keys()))
        default = st.selectbox("Sedang Gagal Bayar Kredit?", list(default_mapping.keys()))
        balance = st.number_input("Saldo Bank (negatif jika utang)", min_value=-10000, max_value=100000, step=100)
        housing = st.selectbox("Sedang Kredit Rumah?", list(housing_mapping.keys()))
        loan = st.selectbox("Punya Pinjaman Pribadi?", list(loan_mapping.keys()))

    with col2:
        contact = st.selectbox("Metode Kontak", list(contact_mapping.keys()))
        day = st.number_input("Dihubungi Pada Tanggal", min_value=1, max_value=31, step=1)
        month = st.selectbox("Dihubungi Pada Bulan", list(month_mapping.keys()))
        campaign = st.number_input("Sudah Berapa Kali Dihubungi (Kampanye Ini)", min_value=1, max_value=50, step=1)
        previous = st.number_input("Sudah Berapa Kali Dihubungi (Kampanye Lalu)", min_value=0, max_value=50, step=1)
        pdays = st.number_input("Jumlah Hari Sejak Kontak Terakhir (-1 jika belum pernah dihubungi)", min_value=-1, max_value=1000, step=1)
        poutcome = st.selectbox("Hasil Kontak Terakhir", list(poutcome_mapping.keys()))

    col1_2, col2_2 = st.columns(2)

    with col1_2:
        col1_2_1, col1_2_2 = st.columns([0.3, 1])
        with col1_2_1:
            predict_button = st.button("Prediksi!")
        with col1_2_2:
            st.markdown("<p style='margin-top:10px;font-size:13px;'>Estimasi Akurasi: 88% | Model Prediksi: Extreme Gradient Boosting</p>", unsafe_allow_html=True)

        if predict_button:
            # Konversi ke bahasa Inggris sebelum diproses
            input_data = pd.DataFrame([{ 
                "age": age, "job": job_mapping[job], "marital": marital_mapping[marital], 
                "education": education_mapping[education], "default": default_mapping[default],
                "balance": balance, "housing": housing_mapping[housing], "loan": loan_mapping[loan],
                "contact": contact_mapping[contact], "day": day, "month": month_mapping[month],
                "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome_mapping[poutcome]
            }])

            input_preprocessed = pipeline_prep.transform(input_data)
            input_transformed = pipeline_trans.transform(input_preprocessed)

            prediction = final_model.predict(input_transformed)
            result = "✅ Klien akan berlangganan!💰" if prediction[0] == 1 else "❌ Klien belum berlangganan. Mungkin di kontak berikutnya!⏳"
            st.subheader(f"{result}")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 20px;
        font-size: 14px;
        color: gray;
    }
    </style>
    <div class="footer">
        © 2025 maba sibuk
    </div>
    """,
    unsafe_allow_html=True
)
