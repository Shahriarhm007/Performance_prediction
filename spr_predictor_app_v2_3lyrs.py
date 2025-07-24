import streamlit as st
import numpy as np
import joblib
import os
from pyngrok import ngrok
import pandas as pd

# Load the pre-trained 3-layer model and scaler
try:
    # Changed to load the 3-layer model and scaler
    model = joblib.load('best_xgboost_model_3lyrs_new.pkl')
    scaler = joblib.load('scaler_3lyrs_new.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'best_xgboost_model_3lyrs_new.pkl' and 'scaler_3lyrs.pkl' are in the correct directory.")
    st.stop() # Stop the app if files are missing

def predict_resonance_and_loss(analyte_ri, num_layers, materials):
    """
    Predicts Resonance Wavelength (µm) and Peak Loss (dB/m) for a given Analyte RI
    and a 3-layer configuration.

    Parameters:
    - analyte_ri (float): Refractive index of the analyte.
    - num_layers (int): Number of layers (1 to 3).
    - materials (list of str): Material names for 3 layers
                              (e.g., ["Au", "Graphene (C)", "None"]).

    Returns:
    - tuple: (resonance_wavelength, peak_loss) in µm and dB/m
    """
    material_codes = {"None": 0, "Au": 1, "Ag": 2, "Cu": 3, "Graphene (C)": 4}
    thickness_map = {"Au": 0.035, "Ag": 0.035, "Cu": 0.035, "Graphene (C)": 0.00034, "None": 0.0}

    # --- MODIFIED LOGIC FOR 3 LAYERS ---
    material_codes_int = [material_codes[mat] for mat in materials]
    thicknesses = [thickness_map[mat] for mat in materials]

    # Calculate 2 distance features for a 3-layer system
    distances = [0.0] * 2
    if num_layers >= 2:
        distances[0] = 1.05 + thicknesses[0]
    if num_layers >= 3:
        distances[1] = 1.05 + thicknesses[0] + thicknesses[1]

    # Ensure input lists have the correct length for the 10-feature model
    if len(material_codes_int) != 3 or len(thicknesses) != 3 or len(distances) != 2:
        raise ValueError("Material codes and thicknesses must have 3 elements, and distances must have 2 elements for the 3-layer model.")

    # Assemble the 10-feature input array
    # [analyte_ri, num_layers] + [mat1, mat2, mat3] + [thick1, thick2, thick3] + [dist1, dist2]
    input_array = [analyte_ri, num_layers] + material_codes_int + thicknesses + distances

    # Convert to numpy array and reshape for scaling (1 sample, 10 features)
    input_array = np.array(input_array).reshape(1, -1)

    # Scale the input using the loaded scaler
    scaled_input = scaler.transform(input_array)

    # Predict using the model (returns log-transformed values)
    predictions = model.predict(scaled_input)

    # Inverse transform predictions using exp(x) - 1 for log1p
    resonance_wavelength = np.expm1(predictions[0][0])
    peak_loss = np.expm1(predictions[0][1]) # Corrected inverse transform

    return resonance_wavelength, peak_loss

# --- MODIFIED STREAMLIT GUI ---
st.title("SPR Sensor Performance Prediction (3-Layer System)")

st.header("Input Parameters")

# Analyte RI Range
st.subheader("Analyte Refractive Index (RI) Range")
ri_start = st.number_input("Start RI", min_value=1.33, max_value=1.43, value=1.33, step=0.001, format="%.3f")
ri_end = st.number_input("End RI", min_value=1.33, max_value=1.43, value=1.41, step=0.001, format="%.3f")
ri_step = st.number_input("Step Size", min_value=0.001, max_value=0.1, value=0.005, step=0.001, format="%.3f")

# Number of Layers (changed options to 1, 2, 3)
num_layers = st.selectbox("Number of Layers", options=[1, 2, 3], index=2) # Default to 3 layers

# Materials (changed loop to range(3))
st.subheader("Material of Each Layer")
material_options = ["None", "Au", "Ag", "Cu", "Graphene (C)"]
materials = []
for i in range(3):
    label = f"Material of Layer {i+1}"
    # Set different defaults for a common 3-layer setup
    default_material = "None"
    if i == 0:
        default_material = "Au"
    elif i == 1:
        default_material = "Graphene (C)"
        
    material = st.selectbox(label, options=material_options, index=material_options.index(default_material))
    materials.append(material)

# Predict Button
if st.button("Predict for RI Range"):
    if ri_start >= ri_end or ri_step <= 0:
        st.error("Invalid Analyte RI range or step size.")
    else:
        results_data = []
        # Generate RI values and predict for each
        for analyte_ri in np.arange(ri_start, ri_end + ri_step, ri_step):
            try:
                resonance, loss = predict_resonance_and_loss(analyte_ri, num_layers, materials)
                results_data.append([analyte_ri, resonance, loss])
            except Exception as e:
                st.error(f"Error during prediction for RI {analyte_ri:.3f}: {str(e)}")
                break # Stop on error

        if results_data:
            results_df = pd.DataFrame(results_data, columns=["Analyte RI", "Resonance Wavelength (µm)", "Peak Loss (dB/m)"])
            st.header("Prediction Results for Analyte RI Range")
            st.dataframe(results_df)

            # Option to download results as CSV
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="spr_prediction_results_3_layer.csv",
                mime="text/csv",
            )
