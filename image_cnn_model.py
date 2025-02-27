import streamlit as st
import pandas as pd
import numpy as np
import folium
import tensorflow as tf
import streamlit_folium
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from streamlit_folium import folium_static

# Load trained model
model = keras.models.load_model("poverty_prediction_model.h5")

# Load poverty dataset
data = pd.read_csv("kenya_poverty_data.csv")

# Normalize the input features
scaler = StandardScaler()
data[["night_lights", "ndvi"]] = scaler.fit_transform(data[["night_lights", "ndvi"]])

# Predict poverty levels
data["poverty_prediction"] = model.predict(data[["night_lights", "ndvi"]]).flatten()

# Create Streamlit UI
st.title("Kenya Poverty Prediction Map 🌍")
st.write("This interactive map visualizes poverty predictions using satellite data.")

# Create Folium map centered around Kenya
m = folium.Map(location=[0.0236, 37.9062], zoom_start=6)

# Add poverty predictions as colored circles
for _, row in data.iterrows():
    color = "red" if row["poverty_prediction"] > 0.7 else "orange" if row["poverty_prediction"] > 0.4 else "green"
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"Poverty Score: {row['poverty_prediction']:.2f}"
    ).add_to(m)

# Display map in Streamlit
folium_static(m)
# streamlit_folium(m)
