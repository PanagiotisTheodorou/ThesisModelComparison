import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Thyroid Disease Prediction",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Function to load a selected model
@st.cache_resource
def load_model(model_path):
    model_data = joblib.load(model_path)
    return model_data['model'], model_data['label_mappings']


# Load and preprocess the dataset for visualzations
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv('../000_Data/raw_with_general_classes.csv')
        df = df[(df['age'] >= 0) & (df['age'] <= 120)]
        numeric_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numeric_cols)
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Visualizations will be disabled.")
        return None


# Function to get default values for features
def get_feature_defaults():
    return {
        # numerical columns
        'age': {'value': 30, 'min': 0, 'max': 120, 'step': 1, 'type': 'int'},
        'TSH': {'value': 2.0, 'min': 0.0, 'max': 500.0, 'step': 0.1, 'type': 'float'},
        'T3': {'value': 1.2, 'min': 0.0, 'max': 10.0, 'step': 0.1, 'type': 'float'},
        'TT4': {'value': 100.0, 'min': 0.0, 'max': 300.0, 'step': 1.0, 'type': 'float'},
        'T4U': {'value': 1.0, 'min': 0.0, 'max': 2.0, 'step': 0.1, 'type': 'float'},
        'FTI': {'value': 100.0, 'min': 0.0, 'max': 300.0, 'step': 1.0, 'type': 'float'},
        'TBG': {'value': 25.0, 'min': 0.0, 'max': 100.0, 'step': 1.0, 'type': 'float'},

        # categorical columns
        'sex': {'type': 'categorical', 'options': ['F', 'M'], 'mapping': {'F': 0, 'M': 1}},
        'on_thyroxine': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'query_on_thyroxine': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'on_antithyroid_meds': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'sick': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'pregnant': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'thyroid_surgery': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'I131_treatment': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'query_hypothyroid': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'query_hyperthyroid': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'lithium': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'goitre': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'tumor': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'hypopituitary': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
        'psych': {'type': 'categorical', 'options': ['f', 't'], 'mapping': {'f': 0, 't': 1}},
    }


st.title("SETA")
st.write("This app predicts thyroid diseases. Select a model from the sidebar to get started.")

# load and Preprocess dataset
dataset = load_and_preprocess_data()

# model selection
st.sidebar.header("Model Selection")
model_options = {
    "Random Forest": "trained_random_forest_model.pkl",
    "Decision Tree": "trained_decision_tree_model.pkl",
    "Support Vector Machine": "trained_svm_model.pkl",
    "Logistic Regression": "trained_logistic_regression_model.pkl",
    "Linear Regression": "trained_linear_regression_model.pkl",
    "Naive Bayes": "trained_naive_bayes_model.pkl",
    "K-Nearest Neighbor": "trained_knn_model.pkl",
}
selected_model = st.sidebar.selectbox("Choose a model", list(model_options.keys()))

# Load model
model_path = model_options[selected_model]
model, label_mappings = load_model(model_path)

# Input fields for user
st.sidebar.header("Input Features")
user_inputs = {}

# Get feature defaults
feature_defaults = get_feature_defaults()

# generete input fields for each feature the model expects
for feature in model.feature_names_in_:
    if feature == 'class':
        continue

    # Handle Ohe features
    if '_' in feature and feature.split('_')[0] in label_mappings:
        original_feature = feature.split('_')[0]
        if original_feature not in user_inputs:
            options = list(label_mappings[original_feature].keys())
            selected = st.sidebar.selectbox(
                original_feature,
                options,
                key=original_feature
            )
            for opt in options:
                user_inputs[f"{original_feature}_{opt}"] = 1 if opt == selected else 0

    # Handle categorical features
    elif feature in feature_defaults and feature_defaults[feature]['type'] == 'categorical':
        params = feature_defaults[feature]
        selected = st.sidebar.selectbox(
            feature.replace('_', ' ').title(),
            params['options'],
            key=feature
        )
        user_inputs[feature] = params['mapping'][selected]

    # Handle numeric columns
    elif feature in feature_defaults:
        params = feature_defaults[feature]
        if params['type'] == 'int':
            user_inputs[feature] = st.sidebar.number_input(
                feature.replace('_', ' ').title(),
                min_value=params['min'],
                max_value=params['max'],
                value=params['value'],
                step=params['step'],
                key=feature
            )
        else:
            user_inputs[feature] = st.sidebar.number_input(
                feature.replace('_', ' ').title(),
                min_value=params['min'],
                max_value=params['max'],
                value=params['value'],
                step=params['step'],
                key=feature
            )
    else:
        # Default for unexpected features
        user_inputs[feature] = st.sidebar.number_input(
            feature.replace('_', ' ').title(),
            value=0.0,
            step=0.1,
            key=feature
        )

#predict button
if st.sidebar.button("Predict"):
    # dataFrame with alll features in correct order
    input_data = pd.DataFrame([user_inputs])[model.feature_names_in_]

    # Make prediction
    try:
        prediction = model.predict(input_data)
        decoded_prediction = label_mappings['class'][prediction[0]]

        # dsplay prediction
        st.success(f"**Prediction:** {decoded_prediction}")
        st.write("### Input Data:")
        st.write(input_data)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Data Visualization Section
if dataset is not None:
    st.header("Data Insights")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Class Distribution",
        "Feature Importance",
        "Feature Distributions",
        "Correlation"
    ])

    with tab1:
        st.subheader("Thyroid Condition Distribution (Original Data)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=dataset, x='class', ax=ax,
                      order=dataset['class'].value_counts().index)
        ax.set_title("Distribution of Thyroid Conditions (Before Transformation)")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write("This shows the original distribution of thyroid conditions in the dataset.")

    with tab2:
        # show the feature importance
        st.subheader("Feature Importance from Random Forest")

        categorical_columns = dataset.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        df_encoded = dataset.copy()

        for col in categorical_columns:
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

        X = df_encoded.drop(columns=['class'])
        y = df_encoded['class']

        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X, y)

        feature_importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
        ax.set_title('Feature Importance from Random Forest')
        st.pyplot(fig)
        st.write("This shows which features are most important for predicting thyroid conditions.")

    with tab3:
        # show a feature distribution  figure
        st.subheader("Distribution of Numerical Features")

        numerical_features = dataset.select_dtypes(include=['float64', 'int64']).columns
        numerical_features = [f for f in numerical_features if f != 'age']

        selected_feature = st.selectbox("Select a feature to view distribution", numerical_features)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(dataset[selected_feature], kde=True, bins=20, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {selected_feature}')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

        fig_age, ax_age = plt.subplots(figsize=(8, 6))
        sns.histplot(dataset['age'], kde=True, bins=20, color='skyblue', ax=ax_age)
        ax_age.set_title('Distribution of Age (Outliers Removed)')
        ax_age.set_xlabel('Age')
        ax_age.set_ylabel('Frequency')
        st.pyplot(fig_age)

    with tab4:
        # craete a feature coorelation plot
        st.subheader("Feature Correlation Heatmap")

        categorical_columns = dataset.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        df_encoded = dataset.copy()

        for col in categorical_columns:
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

        numeric_cols = df_encoded.select_dtypes(include=['float64', 'int64']).columns
        corr = df_encoded[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title("Feature Correlation Heatmap (Numeric and Encoded Features)")
        st.pyplot(fig)
        st.write("This heatmap shows how different features correlate with each other.")

# footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #0078D7;">
        <p>For medical use only. Consult a healthcare professional for accurate diagnosis.</p>
    </div>
    """,
    unsafe_allow_html=True
)