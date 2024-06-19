import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris Flower** type.      
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("Petal length", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("Petal wdith", 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader("User Input Parameters")
st.write(df)

Iris = datasets.load_iris()
X = Iris.data
Y = Iris.target

model = RandomForestClassifier()
model.fit(X, Y)

prediction = model.predict(df)
prediction_prob = model.predict_proba(df)

st.subheader("Class labels and their corresponding index number")
st.write(Iris.target_names)

st.subheader("Prediction")
st.write(Iris.target_names[prediction])

st.subheader("Prediction Probability")
st.write(prediction_prob)