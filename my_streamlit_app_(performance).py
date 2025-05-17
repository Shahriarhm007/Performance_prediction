import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model and scaler
model = joblib.load('performance_best_xgboost_model.pkl')
scaler = joblib.load('performance_feature_scaler.pkl')

def predict_resonance_and_loss(analyte_ri, num_layers, materials, thicknesses, distances):
    """
    Predicts Resonance Wavelength (µm) and Peak Loss (dB/m) using the pre-trained XGBoost model.

    Parameters:
    - analyte_ri (float): Refractive index of the analyte (e.g., 1.33 to 1.42)
    - num_layers (int): Number of layers (1 to 5)
    - materials (list of int): Material codes for 5 layers (e.g., [1, 0, 0, 0, 0], where 1=Au, 2=Ag, 3=Cu, 4=C, 0=none)
    - thicknesses (list of float): Thicknesses for 5 layers in µm (e.g., [0.035, 0, 0, 0, 0])
    - distances (list of float): Distances from core to layers 2-5 in µm (e.g., [0, 0, 0, 0])

    Returns:
    - tuple: (resonance_wavelength, peak_loss) in µm and dB/m
    """
    # Ensure input lists are the correct length
    if len(materials) != 5 or len(thicknesses) != 5 or len(distances) != 4:
        raise ValueError("Materials and thicknesses must have 5 elements, distances must have 4 elements.")

    # Assemble the 16-feature input array
    input_array = [analyte_ri, num_layers] + materials + thicknesses + distances

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
st.title("SPR Sensor Prediction System")

st.header("Input Parameters")

# Analyte RI
analyte_ri = st.number_input("Analyte Refractive Index (RI)", min_value=1.33, max_value=1.42, value=1.33, step=0.01)

# Number of Layers
num_layers = st.selectbox("Number of Layers", options=[1, 2, 3, 4, 5], index=0)

# Materials
st.subheader("Material of Each Layer")
material_options = {"None": 0, "Au": 1, "Ag": 2, "Cu": 3, "Graphene (C)": 4}
materials = []
for i in range(5):
    label = f"Material of Layer {i+1}"
    default = "None" if i >= num_layers else "Au"
    material = st.selectbox(label, options=list(material_options.keys()), index=list(material_options.keys()).index(default))
    materials.append(material_options[material])

# Thicknesses
st.subheader("Thickness of Each Layer (µm)")
thicknesses = []
for i in range(5):
    label = f"Thickness of Layer {i+1} (µm)"
    default = 0.035 if i == 0 else 0.0
    thickness = st.number_input(label, min_value=0.0, max_value=1.0, value=default if i < num_layers else 0.0, step=0.0001)
    thicknesses.append(thickness)

# Distances
st.subheader("Distance from Core to Layers 2–5 (µm)")
distances = []
for i in range(4):
    label = f"Distance to Layer {i+2} (µm)"
    default = 1.085 if i == 0 and num_layers >= 2 else 0.0
    distance = st.number_input(label, min_value=0.0, max_value=2.0, value=default if i < num_layers-1 else 0.0, step=0.001)
    distances.append(distance)

# Predict Button
if st.button("Predict"):
    try:
        resonance, loss = predict_resonance_and_loss(analyte_ri, num_layers, materials, thicknesses, distances)
        st.header("Prediction Results")
        st.write(f"**Predicted Resonance Wavelength**: {resonance:.2f} µm")
        st.write(f"**Predicted Peak Loss**: {loss:.2f} dB/m")
    except Exception as e:
        st.error(f"Error: {str(e)}")