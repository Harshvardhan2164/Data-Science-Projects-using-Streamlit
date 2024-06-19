import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Classification App

This app predicts the **Palmer Penguin** species    

Data obtained from the [palmerpenguins](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV Input File](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)                 
""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV File", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill Length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
        data = {
            'island': island,
            'sex': sex,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()  

penguin_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguin_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

st.subheader("User Input Features")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV File to be uploaded. Currently using the example input features (shown below).")
    st.write(df)

load_model = pickle.load(open('penguins_model.pkl', 'rb'))

prediction = load_model.predict(df)
prediction_prob = load_model.predict_proba(df)

st.subheader('Prediction')
penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguin_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_prob)