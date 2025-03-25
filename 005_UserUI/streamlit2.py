import streamlit as st
import joblib
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Thyroid Disease Prediction",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Function to load a selected model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model(model_path):
    model_data = joblib.load(model_path)
    return model_data['model'], model_data['label_mappings']

# Streamlit app
st.title("Thyroid Disease Prediction")
st.write("""
This app predicts thyroid disease based on input features. Select a model from the sidebar to get started.
""")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_options = {
    "Random Forest": "RandomForestModel.pkl",
    "Support Vector Machine": "trained_svm_model.pkl",
    "Logistic Regression": "trained_logistic_model.pkl",
    "Linear Regression": "trained_linear_regression_model",
    "Naive Bayes": "trained_naive_bayes_model"
}
selected_model = st.sidebar.selectbox("Choose a model", list(model_options.keys()))

# Load the selected model
model_path = model_options[selected_model]
model, label_mappings = load_model(model_path)

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
    input_data = input_data.fillna(0).astype(float)

    # Make prediction
    prediction = model.predict(input_data)
    decoded_prediction = label_mappings['class'][prediction[0]]

    # Display prediction
    st.success(f"**Prediction:** {decoded_prediction}")
    st.write("### Input Data:")
    st.write(input_data)

# Add a footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #0078D7;">
        <p>Developed with by Your Name</p>
        <p>For medical use only. Consult a healthcare professional for accurate diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True
)