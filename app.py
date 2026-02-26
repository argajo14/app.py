import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

# Load model yang sudah dibuat di Colab tadi
model = pickle.load(open('iris_model.pkl', 'rb'))
iris = load_iris()

st.title("Iris Flower Prediction App")
st.write("Aplikasi ini memprediksi jenis bunga Iris berdasarkan input pengguna.")

# Sidebar untuk input parameter
st.sidebar.header("Input Parameter")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input parameters')
st.write(df_input)

# Prediksi
prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction][0])

st.subheader('Prediction Probability')
st.write(prediction_proba)
