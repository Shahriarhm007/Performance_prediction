import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

# Load the pre-trained model and scaler
try:
    model = joblib.load('performance_best_xgboost_model.pkl')
    scaler = joblib.load('performance_feature_scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'performance_best_xgboost_model.pkl' and 'performance_feature_scaler.pkl' are in the correct directory.")
    st.stop()

def predict_resonance_and_loss(analyte_ri, num_layers, materials):
    """
    Predicts Resonance Wavelength (µm) and Peak Loss (dB/m) for a given Analyte RI
    and layer configuration.

    Parameters:
    - analyte_ri (float): Refractive index of the analyte.
    - num_layers (int): Number of layers (1 to 5).
    - materials (list of str): Material names for 5 layers
                              (e.g., ["Au", "None", "Graphene (C)", "Ag", "Cu"]).

    Returns:
    - tuple: (resonance_wavelength, peak_loss) in µm and dB/m
    """
    material_codes = {"N/A": 0, "Gold (Au)": 1, "Silver (Ag)": 2, "Copper (Cu)": 3, "Graphene (C)": 4}
    thickness_map = {"Gold (Au)": 0.035, "Silver (Ag)": 0.035, "Copper (Cu)": 0.035, "Graphene (C)": 0.00034, "N/A": 0.0}

    material_codes_int = [material_codes[mat] for mat in materials]
    thicknesses = [thickness_map[mat] for mat in materials]

    distances = [0.0] * 4
    if num_layers >= 2: distances[0] = 1.05 + thicknesses[0]
    if num_layers >= 3: distances[1] = 1.05 + thicknesses[0] + thicknesses[1]
    if num_layers >= 4: distances[2] = 1.05 + thicknesses[0] + thicknesses[1] + thicknesses[2]
    if num_layers >= 5: distances[3] = 1.05 + thicknesses[0] + thicknesses[1] + thicknesses[2] + thicknesses[3]

    if len(material_codes_int) != 5 or len(thicknesses) != 5 or len(distances) != 4:
        raise ValueError("Material codes and thicknesses must have 5 elements, distances must have 4 elements.")

    input_array = [analyte_ri, num_layers] + material_codes_int + thicknesses + distances
    input_array = np.array(input_array).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    predictions = model.predict(scaled_input)
    resonance_wavelength = np.exp(predictions[0][0]) - 1
    peak_loss = np.exp(predictions[0][1])
    return resonance_wavelength, peak_loss

# Streamlit GUI
st.title("SPR Sensor Performance Prediction")

st.markdown(
    """
    <style>
    .stApp {background-color: #1a1a1a; color: white;}
    .stTextInput > div > div > input {background-color: #2a2a2a; color: white;}
    .stNumberInput > div > div > input {background-color: #2a2a2a; color: white;}
    .stSelectbox > div > div > select {background-color: #2a2a2a; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

st.header("Input Parameters")

# Analyte RI Range
st.subheader("Analyte RI")
col1, col2, col3 = st.columns(3)
with col1: ri_start = st.number_input("Start", min_value=1.33, max_value=1.43, value=1.33, step=0.01, format="%.2f")
with col2: ri_step = st.number_input("Step", min_value=0.01, max_value=0.1, value=0.02, step=0.01, format="%.2f")
with col3: ri_end = st.number_input("End", min_value=1.33, max_value=1.43, value=1.41, step=0.01, format="%.2f")

# Number of Layers
num_layers = st.selectbox("Number of Layers", options=[1, 2, 3, 4, 5], index=0)

# Materials
st.subheader("Plasmonic Metal Layers")
material_options = ["N/A", "Gold (Au)", "Silver (Ag)", "Graphene (C)", "Copper (Cu)"]
materials = [st.selectbox(f"Plasmonic Metal {i+1}st Layer", options=material_options, index=0 if i > 0 else 1) for i in range(5)]

# Calculate Button
if st.button("Calculate"):
    if ri_start >= ri_end or ri_step <= 0:
        st.error("Invalid Analyte RI range or step size.")
    else:
        results_data = []
        ri_values = np.arange(ri_start, ri_end + ri_step, ri_step)
        for analyte_ri in ri_values:
            try:
                resonance, loss = predict_resonance_and_loss(analyte_ri, num_layers, materials)
                results_data.append([analyte_ri, resonance, loss])
            except Exception as e:
                st.error(f"Error during prediction for RI {analyte_ri:.2f}: {str(e)}")
                break

        if results_data:
            results_df = pd.DataFrame(results_data, columns=["Analyte RI", "Resonance Wavelength (µm)", "Peak Loss (dB/m)"])
            
            # Calculate Sensitivities
            wavelength_sensitivity = []
            amplitude_sensitivity = []
            for i in range(len(results_df) - 1):
                delta_ri = results_df["Analyte RI"][i + 1] - results_df["Analyte RI"][i]
                delta_wavelength = results_df["Resonance Wavelength (µm)"][i + 1] - results_df["Resonance Wavelength (µm)"][i]
                delta_loss = results_df["Peak Loss (dB/m)"][i + 1] - results_df["Peak Loss (dB/m)"][i]
                if delta_ri != 0 and results_df["Peak Loss (dB/m)"][i] != 0:
                    s_lambda = delta_wavelength / delta_ri if delta_ri != 0 else 0
                    s_amplitude = -(1 / results_df["Peak Loss (dB/m)"][i]) * (delta_loss / delta_ri) if delta_ri != 0 else 0
                    wavelength_sensitivity.append(s_lambda)
                    amplitude_sensitivity.append(s_amplitude)
            # Assign first row as 0 (no previous data for sensitivity)
            wavelength_sensitivity.insert(0, 0)
            amplitude_sensitivity.insert(0, 0)

            # Add sensitivity columns
            results_df["Wavelength Sensitivity"] = wavelength_sensitivity
            results_df["Amplitude Sensitivity"] = amplitude_sensitivity

            st.dataframe(results_df)

            # Export Options
            export_path = st.text_input("Export generated data (excel):", value="D:\\MLDataset")
            if st.button("Export"):
                results_df.to_excel(f"{export_path}\\spr_results.xlsx", index=False)
                st.success("Data exported successfully!")

            # CSV Download Option
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="spr_prediction_results.csv",
                mime="text/csv",
            )
