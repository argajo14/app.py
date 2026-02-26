import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

# 1. Load Model
# Pastikan file iris_model.pkl sudah di-upload ke GitHub di folder yang sama
try:
    with open('iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("File 'iris_model.pkl' tidak ditemukan. Pastikan sudah di-upload ke GitHub.")

# 2. Load Data untuk Label
iris = load_iris()

st.title("🌸 Iris Species Predictor")

# 3. Sidebar Input
st.sidebar.header("Input Parameter")
def user_input_features():
    sepal_l = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_w = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_l = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_w = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_l, 'sepal_width': sepal_w,
            'petal_length': petal_l, 'petal_width': petal_w}
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# 4. Prediksi
if 'model' in locals():
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    st.subheader('Hasil Prediksi')
    st.success(f"Spesies: **{iris.target_names[prediction][0]}**")

    st.subheader('Probabilitas')
    st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))
