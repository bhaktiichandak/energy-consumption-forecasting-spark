import streamlit as st
import json
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Energy Forecast",
    page_icon="⚡",
    layout="centered"
)

# Title and description
st.title("⚡ Energy Consumption Forecast")
st.markdown("Predict Global Active Power using Spark MLlib Linear Regression")

# Load model parameters
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "spark_model", "model_params.json")
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, "r") as f:
        model_data = json.load(f)
    return model_data

model_data = load_model()

if model_data is None:
    st.error("❌ Model not found! Please run `python train_model.py` first to train the model.")
    st.stop()

# Extract model parameters
coefficients = np.array(model_data["coefficients"])
intercept = model_data["intercept"]
feature_names = model_data["features"]
training_rmse = model_data["rmse"]

# Display model info
with st.sidebar:
    st.subheader("📊 Model Information")
    st.metric("Training RMSE", f"{training_rmse:.4f}")
    st.metric("Features", len(feature_names))
    st.markdown("**Features used:**")
    for feat in feature_names:
        st.write(f"• {feat}")

# Create input form
st.header("Input Parameters")

# Input fields for the features used in training
col1, col2 = st.columns(2)

with col1:
    voltage = st.number_input(
        "Voltage (V)",
        min_value=0.0,
        value=240.0,
        step=0.1,
        help="Electrical voltage in volts"
    )

    global_intensity = st.number_input(
        "Global Intensity (A)",
        min_value=0.0,
        value=10.0,
        step=0.1,
        help="Global electrical intensity in amperes"
    )

    sub_metering_1 = st.number_input(
        "Sub Metering 1 (Wh)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="Energy consumption from sub-metering 1 in watt-hours"
    )

with col2:
    hour = st.number_input(
        "Hour of Day (0-23)",
        min_value=0,
        max_value=23,
        value=12,
        step=1,
        help="Hour extracted from time (0-23)"
    )

    day = st.number_input(
        "Day of Month (1-31)",
        min_value=1,
        max_value=31,
        value=15,
        step=1,
        help="Day extracted from date (1-31)"
    )

# Predict button
if st.button("🔮 Predict Global Active Power", type="primary", use_container_width=True):
    # Prepare feature vector (same order as training)
    features = np.array([voltage, global_intensity, sub_metering_1, hour, day])
    
    # Linear regression prediction: y = coefficients · features + intercept
    prediction = np.dot(coefficients, features) + intercept
    
    # Display result
    st.success("✅ Prediction completed!")
    st.metric(
        label="Predicted Global Active Power (kW)",
        value=f"{prediction:.4f}",
        delta="kW"
    )
    
    # Display feature breakdown
    st.subheader("📈 Feature Contribution to Prediction")
    contributions = coefficients * features
    
    breakdown_data = {
        "Feature": feature_names,
        "Value": [f"{v:.2f}" for v in features],
        "Coefficient": [f"{c:.6f}" for c in coefficients],
        "Contribution": [f"{co:.4f}" for co in contributions]
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(breakdown_data, use_container_width=True, hide_index=True)
    
    with col2:
        st.write(f"**Base (Intercept):** {intercept:.4f}")
        st.write(f"**Total Prediction:** {prediction:.4f} kW")
        st.info(f"📝 This prediction is based on the Linear Regression model trained with Spark MLlib on the UCI energy consumption dataset.")

# Footer
st.markdown("---")
st.markdown(
    "**Project:** Energy Consumption Forecasting using Spark MLlib | "
    "**Algorithm:** Linear Regression | "
    "**Framework:** Spark MLlib"
)