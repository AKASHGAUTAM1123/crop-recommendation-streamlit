import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

@st.cache_resource
def train_model():
    # load data
    df = pd.read_csv("crop_recommendation.csv")
    X = df.drop("label", axis=1)
    y = df["label"]   # keep as crop names (rice, maize, etc.)

    # model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = train_model()

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and weather values to get a suitable crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

if st.button("Recommend Crop"):
    input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Recommended crop: **{prediction}**")
