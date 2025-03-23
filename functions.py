import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# segmentasi usia
def segment_age2(age):
    if age < 33: 
        return '<33 tahun'
    elif 34 <= age <= 39: # usia produktif (mayoritas)
        return '34-39 tahun'
    elif 40 <= age <= 48:
        return '40-48 tahun'
    else: 
        return '49+ tahun'

# segmentasi durasi telpon menjadi singkat dan lama berdasarkan median
# median digunakan karena lebih rentan terhadap outlier
def segment_duration(duration, median):
    if duration < 120: 
        return '<2 menit'
    elif 121 <= duration <= 240:
        return '2-4 menit'
    elif 241 <= duration <= 480:
        return '4-8 menit'
    else: 
        return '9+ menit'

# segementasi saldo menjadi rendah dan tinggi berdasarkan median
def segment_balance(balance, median):
    if balance < 0: 
        return '<€0'
    elif 1 <= balance <= 150:
        return '€0-150'
    elif 151 <= balance <= 550:
        return '€151-550'
    elif 551 <= balance <= 1500:
        return '€551-1500'
    else: 
        return '€1500+'
        
# segmentasi seberapa sering dikontak di kampanye ini
def segment_campaign2(campaign):
    if campaign == 1:
        return 'Sekali'
    elif 2 <= campaign <= 4:
        return '2-4 kontak'
    else:
        return '5+ kontak'

# segmentasi seberapa sering dikontak di kampanye sebelumnya
def segment_previous2(previous):
    if previous == 0:
        return 'Tidak pernah'
    elif 1 <= previous <= 3:
        return '1-3 kontak'
    else:
        return '4+ kontak'
    
# segmentasi sudah berapa lama sejak terakhir dikontak di kampanye sebelumnya
def segment_pdays2(pdays):
    if pdays == -1:
        return 'Tidak pernah'
    elif 0 <= pdays <= 90:
        return '3 bulan terakhir'
    elif 91 <= pdays <= 180:
        return '6 bulan terakhir'
    else:
        return '6+ bulan terakhir'

def plot_segmented_variable(df, segment_col, height, target_col='Subscribed', key=None):

    # Aggregate counts per segment
    segment_counts = df[segment_col].value_counts().sort_index()

    # Calculate subscription rate per segment
    subscription_rate = df.groupby(segment_col)[target_col].apply(lambda x: (x == 'yes').mean() * 100)

    # Create figure
    fig = go.Figure()

    category_order = {
            "Kelompok Umur": ['<33 tahun', '34-39 tahun', '40-48 tahun', '49+ tahun'],
            "Durasi Telepon": ['<2 menit', '2-4 menit', '4-8 menit', '9+ menit'],
            "Saldo Bank": ['<€0', '€0-150', '€151-550', '€551-1500', '€1500+'],
            "Berapa Kali Dihubungi (kampanye ini)": ['Sekali', '2-4 kontak', '5+ kontak'],
            "Berapa Kali Dihubungi (kampanye lalu)": ['Tidak pernah', '1-3 kontak', '4+ kontak'], 
            "Berapa Lama Sejak Terakhir Dihubungi": ['Tidak pernah', '3 bulan terakhir', '6 bulan terakhir', '6+ bulan terakhir']
    }
    subtitles = {
            "Kelompok Umur": "Kelompok tua paling sering dihubungi. Kelompok muda paling mungkin berlangganan.",
            "Durasi Telepon": "Semakin lama durasi telepon, semakin mungkin klien berlangganan.",
            "Saldo Bank": "Kelompok saldo tinggi paling sering dihubungi dan paling mungkin berlangganan.",
            "Berapa Kali Dihubungi (kampanye ini)": "Kebanyakan klien langsung setuju berlangganan di kontak pertama kampanye.",
            "Berapa Kali Dihubungi (kampanye lalu)": "Klien yang belum pernah dihubungi di kampanye sebelumnya jarang langsung ingin berlangganan.",
            "Berapa Lama Sejak Terakhir Dihubungi": "Dalam kampanye yang sama, jangan tunggu terlalu lama sebelum menghubungi klien kembali."
        }


    # Bar chart: Count per segment
    fig.add_trace(go.Bar(
        x=segment_counts.index, 
        y=segment_counts.values, 
        name="Jumlah Data", 
        marker_color='skyblue',
        hovertemplate='%{y}'
    ))

    sorted_index = category_order[segment_col]
    subscription_rate = subscription_rate.reindex(sorted_index)

    # Line chart: Subscription rate
    fig.add_trace(go.Scatter(
        x=subscription_rate.index, 
        y=subscription_rate.values, 
        mode='lines+markers', 
        name="Persentase Sukses (%)",
        yaxis='y2',
        marker=dict(color='#ff4b4b', size=8),
        line=dict(width=2),
        hovertemplate='%{y:.2f}'
    ))

    # Layout settings
    fig.update_layout(
        title=dict(text=f"<b style='font-size:17px;'><span style='color:skyblue;'>{segment_col}</span></b><br><span style='font-size:12px;font-weight:normal;'>{subtitles.get(segment_col, '')}</span>"),
        xaxis=dict(title="Kategori", categoryorder='array', categoryarray=category_order.get(segment_col,[])),
        yaxis=dict(title="Jumlah Data"),
        yaxis2=dict(title="Persentase Sukses (%)", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.8, y=1.1),
        template="plotly_dark",
        height=height
    )

    st.plotly_chart(fig, use_container_width=True, key=key)

def new_features(df):
    df = df.copy()
    df['balance_square'] = df['balance'] ** 2
    df['age_square'] = df['age'] ** 2
    df['age_balance'] = df['age'] * df['balance']
    df['balance_per_age'] = df['balance'] / df['age']
    df['pdays_previous_diff'] = df['pdays'] - df['previous']
    df['contacted'] = df['pdays'].apply(lambda x: 'no' if x == -1 else 'yes')

    return df

def segment_age(age):
    if age < 25: 
        return '<25'
    elif 25 <= age <= 50: # usia produktif (mayoritas)
        return '21-50'
    else: 
        return '51+'
    
def segment_campaign(campaign):
    if campaign == 1:
        return 'Sekali'
    elif 2 <= campaign <= 4:
        return '2 - 4 Kontak'
    else:
        return '5+ Kontak'
    
def segment_previous(previous):
    if previous == 0:
        return 'Tidak pernah'
    elif 1 <= previous <= 3:
        return '1-3 Kontak'
    else:
        return '4+ Kontak'
    
def segment_pdays(pdays):
    if pdays == -1:
        return 'Tidak pernah'
    elif 0 <= pdays <= 7:
        return 'Dalam minggu terakhir'
    elif 8 <= pdays <= 180:
        return 'Dalam 6 bulan terakhir'
    else:
        return 'Lebih dari 6 bulan yang lalu'


# preprocessing feature 
def preprocess_dataframe(df):
    df = df.copy()
    # Feature Engineering
    df['Tanggal Telepon'] = df['day'].apply(lambda x: 'Early' if x <= 15 else 'Late')
    df['Kelompok Umur'] = df['age'].apply(segment_age)
    df['Berapa Kali Dihubungi (kampanye ini)'] = df['campaign'].apply(segment_campaign) #discretization campaign
    df['Berapa Kali Dihubungi (kampanye lalu)'] = df['previous'].apply(segment_previous) #discretization previous
    df['Berapa Lama Sejak Terakhir Dihubungi'] = df['pdays'].apply(segment_pdays)

    return df

# clipping untuk variabel age
def replace_outliers_series(series, method='mean', lower_bound=None, upper_bound=None):    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR if lower_bound is None else lower_bound
    upper = Q3 + 1.5 * IQR if upper_bound is None else upper_bound

    if method == 'bound':
        replacement = None
    if method == 'bound':
        return series.clip(lower, upper)
    else:
        return np.where((series < lower) | (series > upper), replacement, series)

def apply_transform_series(series, transform_type='log'):
    series_copy = series.copy()
    
    if transform_type == 'log':
        return np.log1p(series_copy) 
    elif transform_type == 'sqrt':
        return np.sqrt(series_copy)

def outlier_process(df):
    # Replace outliers pada 'age'
    df['age'] = replace_outliers_series(df['age'], method='bound', lower_bound=-100, upper_bound=100)
    
    # Transformasi 'balance'
    min_value_balance = df['balance'].min()
    if min_value_balance < 0:
        df['balance'] = apply_transform_series(df['balance'] - min_value_balance + 1, transform_type='log')
    else:
        df['balance'] = apply_transform_series(df['balance'], transform_type='log')

    # Transformasi 'pdays'
    min_value_pdays = df['pdays'].min()
    if min_value_pdays < 0:
        df['pdays'] = apply_transform_series(df['pdays'] - min_value_pdays + 1, transform_type='sqrt')
    else:
        df['pdays'] = apply_transform_series(df['pdays'], transform_type='sqrt')

    return df
