import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('credit_risk_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('credit_toy_dataset.csv')

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

st.title("🏦 Credit Risk Prediction App")
st.markdown("""
This application predicts whether a credit applicant is a **Good** or **Bad** credit risk 
based on their profile. It uses a Machine Learning model trained on the German Credit dataset.
""")

st.header("Applicant Information")

# Separate features from target
X = df.drop('target', axis=1)

# Create input fields dynamically based on the dataset
input_data = {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial & Loan Details")
    for col in ['checking_status', 'credit_amount', 'duration', 'purpose', 'credit_history', 'savings_status', 'installment_commitment', 'other_payment_plans', 'existing_credits']:
        if col in X.columns:
            if X[col].dtype == 'object':
                input_data[col] = st.selectbox(f"{col.replace('_', ' ').capitalize()}", options=X[col].unique())
            else:
                input_data[col] = st.number_input(f"{col.replace('_', ' ').capitalize()}", min_value=int(X[col].min()), max_value=int(X[col].max()), value=int(X[col].median()))

with col2:
    st.subheader("Personal Details")
    for col in ['age', 'personal_status', 'employment', 'job', 'housing', 'property_magnitude', 'residence_since', 'num_dependents', 'other_parties', 'own_telephone', 'foreign_worker']:
        if col in X.columns:
            if X[col].dtype == 'object':
                input_data[col] = st.selectbox(f"{col.replace('_', ' ').capitalize()}", options=X[col].unique())
            else:
                input_data[col] = st.number_input(f"{col.replace('_', ' ').capitalize()}", min_value=int(X[col].min()), max_value=int(X[col].max()), value=int(X[col].median()))

# Prediction Button
if st.button("Predict Credit Risk", type="primary", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    
    with st.spinner("Analyzing profile..."):
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            st.divider()
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.success(f"✅ **Good Credit Risk**")
                st.write(f"Confidence: **{probability[1]*100:.2f}%**")
            else:
                st.error(f"❌ **Bad Credit Risk**")
                st.write(f"Confidence: **{probability[0]*100:.2f}%**")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
