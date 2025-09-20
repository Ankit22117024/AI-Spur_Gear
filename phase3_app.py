import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# --- Load the Trained Model and Scalers ---
# Use st.cache_resource to load these only once, making the app faster.
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('gear_predictor_model.h5')
    scaler_X = joblib.load('scaler_X.joblib')
    scaler_y = joblib.load('scaler_y.joblib')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_assets()

# --- Build the User Interface ---

st.set_page_config(page_title="ML Gear Design Assistant", layout="wide")
st.title("⚙️ ML Spur Gear Design Assistant")
st.write("Enter your operating conditions, and the AI will predict the optimal gear geometry.")

st.divider()

# Create columns for a clean layout
col1, col2 = st.columns(2)

with col1:
    st.header("Input Parameters")
    p_in = st.number_input('Input Power (kW)', min_value=1.0, max_value=200.0, value=10.0, step=1.0)
    n_in = st.number_input('Input Speed (RPM)', min_value=500.0, max_value=3000.0, value=1200.0, step=50.0)

with col2:
    st.header("Output Parameters")
    p_out = st.number_input('Desired Output Power (kW)', min_value=1.0, max_value=200.0, value=9.8, step=1.0)
    n_out = st.number_input('Desired Output Speed (RPM)', min_value=100.0, max_value=3000.0, value=400.0, step=50.0)

st.divider()

# --- Backend Logic and Prediction ---

if st.button('**Design My Gear**', use_container_width=True):
    # Input validation
    if n_out <= 0 or n_in <= 0:
        st.error("Error: Speeds must be greater than zero.")
    else:
        # 1. Process the inputs
        ratio = n_in / n_out
        efficiency = (p_out / p_in) * 100

        st.info(f"Calculated Velocity Ratio: **{ratio:.2f}** | Calculated Efficiency: **{efficiency:.1f}%**")
        
        # 2. Prepare data for the model
        # The model expects a 2D array, so we wrap our input in another list
        input_data = np.array([[p_in, n_in, ratio]])
        input_data_scaled = scaler_X.transform(input_data)

        # 3. Make a prediction
        prediction_scaled = model.predict(input_data_scaled)
        
        # 4. Inverse transform the result to get human-readable values
        predicted_geometry = scaler_y.inverse_transform(prediction_scaled)

        # 5. Display the final result
        st.success('**Recommended Gear Design:**')
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Module (m)", f"{predicted_geometry[0][0]:.2f} mm")
        res_col2.metric("Pinion Teeth (Zp)", f"{int(predicted_geometry[0][1])}")
        res_col3.metric("Face Width (b)", f"{predicted_geometry[0][2]:.2f} mm")