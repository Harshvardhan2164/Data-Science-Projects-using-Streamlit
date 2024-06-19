import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# House Price Prediction App

This app predicts the **House Prices**.      
""")
st.write('---')

# data_url = "http://lib.stat.cmu.edu/datasets/boston" raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None) data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) target = raw_df.values[1::2, 2]

data = datasets.fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
Y = pd.DataFrame(data.target, columns=['MedHouseVal'])

st.sidebar.header("Specify Input Features")

def user_input_features():
    MedInc = st.sidebar.slider('MedInc', X.MedInc.min(), X.MedInc.max(), X.MedInc.mean())
    HouseAge = st.sidebar.slider('HouseAge', X.HouseAge.min(), X.HouseAge.max(), X.HouseAge.mean())
    AveRooms = st.sidebar.slider('AveRooms', X.AveRooms.min(), X.AveRooms.max(), X.AveRooms.mean())
    AveBedrms = st.sidebar.slider('AveBedrms', X.AveBedrms.min(), X.AveBedrms.max(), X.AveBedrms.mean())
    Population = st.sidebar.slider('Population', X.Population.min(), X.Population.max(), X.Population.mean())
    AveOccup = st.sidebar.slider('AveOccup', X.AveOccup.min(), X.AveOccup.max(), X.AveOccup.mean())
    Latitude = st.sidebar.slider('Latitude', X.Latitude.min(), X.Latitude.max(), X.Latitude.mean())
    Longitude = st.sidebar.slider('Longitude', X.Longitude.min(), X.Longitude.max(), X.Longitude.mean())
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.header("Specified Input Parameters")
st.write(df)
st.write('---')

model = RandomForestRegressor()
model.fit(X, Y)
prediction = model.predict(df)

st.header("Prediction of MedHouseVal")
st.write(prediction)
st.write('---')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header("Feature Importance")
plt.title("Feature Importance based on SHAP values")
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches = 'tight')
st.write('---')

plt.title("Feature Importance based on SHAP values (Bar)")
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches = 'tight')