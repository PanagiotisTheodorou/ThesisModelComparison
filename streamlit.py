import streamlit as st
import joblib
import pandas as pd

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    model_data = joblib.load('RandomForestModel.pkl')
    return model_data['model'], model_data['label_mappings']

model, label_mappings = load_model()

# Streamlit app
st.title("Thyroid Disease Prediction")
st.write("""
This app predicts thyroid disease based on input features.
""")

# Input fields for user
st.sidebar.header("Input Features")

# Define input fields for the features the user can provide
user_inputs = {
    'age': st.sidebar.number_input("Age", min_value=0, max_value=120, value=30),
    'TSH': st.sidebar.number_input("TSH", min_value=0.0, value=2.0),
    'T3': st.sidebar.number_input("T3", min_value=0.0, value=1.2),
    'TT4': st.sidebar.number_input("TT4", min_value=0.0, value=100.0),
    'T4U': st.sidebar.number_input("T4U", min_value=0.0, value=1.0),
    'FTI': st.sidebar.number_input("FTI", min_value=0.0, value=100.0),
}

# Predict button
if st.sidebar.button("Predict"):
    # Create a DataFrame with all features expected by the model
    input_data = pd.DataFrame(columns=model.feature_names_in_)

    # Fill in the user-provided features
    for feature in user_inputs:
        input_data[feature] = [user_inputs[feature]]

    # Fill missing features with default values (e.g., 0)
    input_data = input_data.fillna(0)

    # Make prediction
    prediction = model.predict(input_data)
    decoded_prediction = label_mappings['class'][prediction[0]]

    # Display prediction
    st.success(f"Prediction: **{decoded_prediction}**")
    st.write("### Input Data:")
    st.write(input_data)