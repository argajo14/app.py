import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    # Pastikan file ini sudah di-upload ke GitHub
    return joblib.load('iris_model.pkl')

model = load_model()
iris = load_iris()

st.title("🌸 Iris Species Predictor")

# --- SIDEBAR INPUT ---
st.sidebar.header("Input Parameter")

def user_input_features():
    sepal_l = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
    sepal_w = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
    petal_l = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
    petal_w = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    # PENYESUAIAN NAMA KOLOM (Sesuai Pesan Error)
    # Model Anda meminta 'petal_width (cm)' bukan 'petal width (cm)'
    data = {
        'sepal length (cm)': sepal_l,
        'sepal width (cm)': sepal_w,
        'petal length (cm)': petal_l,
        'petal_width (cm)': petal_w  # Menggunakan underscore (_)
    }
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# --- TAMPILKAN PARAMETER ---
st.subheader('Parameter yang Anda Masukkan:')
st.write(df)

# --- PROSES PREDIKSI ---
try:
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    st.subheader('Hasil Prediksi:')
    spesies = iris.target_names[prediction][0]
    st.success(f"Spesies: **{spesies.upper()}**")

    st.subheader('Probabilitas Prediksi:')
    df_proba = pd.DataFrame(prediction_proba, columns=iris.target_names)
    st.bar_chart(df_proba.T)
    
except Exception as e:
    st.error(f"Terjadi kesalahan saat prediksi: {e}")
