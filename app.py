# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:58:10 2025

@author: MUSTAFA
"""

import streamlit as st
import pandas as pd
import pickle
import os 
from PIL import Image
import random as random

def get_random_images():
    images_folder = 'images'
    valid_ext = ('.jpg', '.jpeg', '.png', '.gif')
    try:
        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(valid_ext)]
    except FileNotFoundError():
       image_files = []
    if not image_files:
        return None
    return random.choice(image_files)

def preprocess_user_input(user_inputs, train_cols):
    user_df = pd.DataFrame([user_inputs])
    user_df = pd.get_dummies(user_df, columns=['Brand', 'Processor', 'GPU', 'Operating System'], drop_first=True)
    for col in train_cols:
        if col in user_df:
            continue
        else:
            user_df[col] = 0
            
    if f"Brand_{user_inputs['Brand']}" in train_cols:
        user_df[f"Brand_{user_inputs['Brand']}"] = 1
    if f"Processor_{user_inputs['Processor']}" in train_cols:
        user_df[f"Processor_{user_inputs['Processor']}"] = 1
    if f"GPU_{user_inputs['GPU']}" in train_cols:
        user_df[f"GPU_{user_inputs['GPU']}"] = 1
    if f"Operating System_{user_inputs['Operating System']}" in train_cols:
        user_df[f"Operating System_{user_inputs['Operating System']}"] = 1
    return user_df[train_cols]

# Modelleri yükle
lnr_model = pickle.load(open('linear_regression_model.pkl', 'rb'))
xgb_model = pickle.load(open('xgboost_model.pkl', 'rb'))
lgb_model = pickle.load(open('lightgbm_model.pkl', 'rb'))

# Veri setini yükle (özellikleri almak için)
df = pd.read_csv('laptop_prices.csv')

# Sayfa yapılandırması ve tema
st.set_page_config(page_title='Laptop Price Predictor', page_icon='💻', layout='wide')
st.markdown(
    """
    <style>
    body {
         background-color : #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color : #e8f5e9;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

# Streamlit arayüzü
st.title('💻 Leptop Price Predictor')

# Kullanıcıdan girdi al
st.sidebar.header('Leptop Features')
screen_size = st.sidebar.slider('Screen Size (inch)', min_value=10.0, max_value=20.0, value=15.6)
ram = st.sidebar.slider('RAM (GB)', min_value=4, max_value=64, value=8)
storage = st.sidebar.slider('Storage (GB)', min_value=128, max_value=2048, value=512)
ppi = st.sidebar.slider('PPI', min_value=100, max_value=400, value=220)
weight = st.sidebar.slider('Weight (kg)', min_value=1.0, max_value=5.0, value=2.0)
brand = st.sidebar.selectbox('Brand', df['Brand'].unique())
processor = st.sidebar.selectbox('Processor', df['Processor'].unique())
gpu = st.sidebar.selectbox('GPU', df['GPU'].unique())
ost = st.sidebar.selectbox('Operating System', df['Operating System'].unique())
currency = st.sidebar.selectbox('Choose your currency', ('USD', 'EUR', 'TRY'))
selected_model = st.sidebar.selectbox("Select Model", ("Linear Regression", "XGBoost", "LightGBM"))
eur_usd = 1.10
tr_usd = 36.55
# Tahmin yap
if st.button('Predict Pc Price'):
    # Kullanıcı girdilerini DataFrame'e dönüştür
    user_inputs = {
        'Screen Size (inch)': screen_size,
        'RAM (GB)': ram,
        'Storage': storage,
        'PPI': ppi,
        'Weight (kg)': weight,
        'Brand': brand,
        'Processor': processor,
        'GPU': gpu,
        'Operating System': ost
    }

    # Kategorik özellikleri one-hot encode et
    train_cols = lnr_model.feature_names_in_
    user_data = preprocess_user_input(user_inputs, train_cols)

    # Tahminleri yap
    if selected_model == "Linear Regression":
        prediction = lnr_model.predict(user_data)[0]
    elif selected_model == "XGBoost":
        prediction = xgb_model.predict(user_data)[0]
    elif selected_model == "LightGBM":
        prediction = lgb_model.predict(user_data)[0]
    else:
        prediction = lgb_model.predict(user_data)[0]

    # Tahmin sonuçlarını göster
    st.subheader("Prediction:")
    if currency == 'USD':
        st.markdown(f"<p style='font-size: 20px; font-weight: bold;'>${prediction:.2f}</p>", unsafe_allow_html=True)
    elif currency == 'EUR':
        st.markdown(f"<p style='font-size: 20px; font-weight: bold;'>€{prediction * eur_usd:.2f}</p>", unsafe_allow_html=True)
    elif currency == 'TRY':
        st.markdown(f"<p style='font-size: 20px; font-weight: bold;'>₺{prediction * tr_usd:.2f}</p>", unsafe_allow_html=True)
        
    image_path = get_random_images()
    if image_path:
        image_path = os.path.abspath(image_path)
    else:
        image_path = ""
        st.write('Resim dosyası yüklenirken bir hata oluştu.')
    
    image = Image.open(image_path)
    st.image(image,width=300)
        
    
    
    

