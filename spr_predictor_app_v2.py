import streamlit as st
import numpy as np
import joblib
import os
from pyngrok import ngrok
import pandas as pd

# Load the pre-trained model and scaler
try:
    model = joblib.load('performance_best_xgboost_model.pkl')
    scaler = joblib.load('performance_feature_scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'performance_best_xgboost_model.pkl' and 'performance_feature_scaler.pkl' are in the correct directory.")
    st.stop() # Stop the app if files are missing

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
    material_codes = {"None": 0, "Au": 1, "Ag": 2, "Cu": 3, "Graphene (C)": 4}
    thickness_map = {"Au": 0.035, "Ag": 0.035, "Cu": 0.035, "Graphene (C)": 0.00034, "None": 0.0}

    material_codes_int = [material_codes[mat] for mat in materials]
    thicknesses = [thickness_map[mat] for mat in materials]

    distances = [0.0] * 4
    if num_layers >= 2:
        distances[0] = 1.05 + thicknesses[0]
    if num_layers >= 3:
        distances[1] = 1.05 + thicknesses[0] + thicknesses[1]
    if num_layers >= 4:
        distances[2] = 1.05 + thicknesses[0] + thicknesses[1] + thicknesses[2]
    if num_layers >= 5:
        distances[3] = 1.05 + thicknesses[0] + thicknesses[1] + thicknesses[2] + thicknesses[3]

    # Ensure input lists are the correct length
    if len(material_codes_int) != 5 or len(thicknesses) != 5 or len(distances) != 4:
        raise ValueError("Material codes and thicknesses must have 5 elements, distances must have 4 elements.")

    # Assemble the 16-feature input array
    input_array = [analyte_ri, num_layers] + material_codes_int + thicknesses + distances

    # Convert to numpy array and reshape for scaling (1 sample, 16 features)
    input_array = np.array(input_array).reshape(1, -1)

    # Scale the input using the loaded scaler
    scaled_input = scaler.transform(input_array)

    # Predict using the model (returns log-transformed values)
    predictions = model.predict(scaled_input)

    # Exponentiate predictions and adjust for resonance wavelength
    resonance_wavelength = np.exp(predictions[0][0]) - 1  # Adjusted for ln(y + 1) training
    peak_loss = np.exp(predictions[0][1])  # No adjustment needed for peak loss (assuming ln(y))

    return resonance_wavelength, peak_loss

# Streamlit GUI
st.title("SPR Sensor Performance Prediction")

st.header("Input Parameters")

# Analyte RI Range
st.subheader("Analyte Refractive Index (RI) Range")
ri_start = st.number_input("Start RI", min_value=1.33, max_value=1.43, value=1.33, step=0.001, format="%.3f")
ri_end = st.number_input("End RI", min_value=1.33, max_value=1.43, value=1.41, step=0.001, format="%.3f")
ri_step = st.number_input("Step Size", min_value=0.001, max_value=0.1, value=0.005, step=0.001, format="%.3f")

# Number of Layers
num_layers = st.selectbox("Number of Layers", options=[1, 2, 3, 4, 5], index=0)

# Materials
st.subheader("Material of Each Layer")
material_options = ["None", "Au", "Ag", "Cu", "Graphene (C)"]
materials = []
for i in range(5):
    label = f"Material of Layer {i+1}"
    default_material = "None"
    if i == 0:
        default_material = "Au"
    material = st.selectbox(label, options=material_options, index=material_options.index(default_material))
    materials.append(material)

# Predict Button
if st.button("Predict for RI Range"):
    if ri_start >= ri_end or ri_step <= 0:
        st.error("Invalid Analyte RI range or step size.")
    else:
        results_data = []
        for analyte_ri in np.arange(ri_start, ri_end + ri_step, ri_step):
            try:
                resonance, loss = predict_resonance_and_loss(analyte_ri, num_layers, materials)
                results_data.append([analyte_ri, resonance, loss])
            except Exception as e:
                st.error(f"Error during prediction for RI {analyte_ri:.3f}: {str(e)}")
                break

        if results_data:
            results_df = pd.DataFrame(results_data, columns=["Analyte RI", "Resonance Wavelength (µm)", "Peak Loss (dB/m)"])
            st.header("Prediction Results for Analyte RI Range")
            st.dataframe(results_df)

            # Option to download results as CSV
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="spr_prediction_results.csv",
                mime="text/csv",
            )

# %%
# Run the Streamlit app in the background and expose it with ngrok
# You need to replace 'YOUR_NGROK_AUTH_TOKEN' with your actual ngrok token
# You can get a token from https://dashboard.ngrok.com/get-started/your-authtoken
# If you don't have an ngrok account, you'll need to sign up.

try:
    # Authenticate ngrok (replace with your actual token)
    # You only need to run this once per Colab session
    ngrok.set_auth_token("2xEP2anuSbgqGu4xc3OgYU3TB4F_25Lp6xgVapAkdZSHJ1ig1")

    # Start the Streamlit app in the background
    # The &> /dev/null & redirects output and runs in the background
    os.system(f"streamlit run {streamlit_app_file} &")

    # Wait a few seconds for the app to start (optional but recommended)
    import time
    time.sleep(5)

    # Create a public URL with ngrok for port 8501 (Streamlit's default)
    public_url = ngrok.connect(8501)

    print(f"Streamlit app public URL: {public_url}")

except Exception as e:
    print(f"An error occurred while trying to start ngrok or Streamlit: {e}")
    print("Please ensure you have replaced 'YOUR_NGROK_AUTH_TOKEN' with your actual token.")
    print("Also, check the Colab output for any errors from Streamlit.")
