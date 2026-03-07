# Credit Risk Predictor 🏦

This directory contains the deployment files for **Assignment 3: NLP & ML Project Development**. 
A machine learning model trained to predict whether a credit applicant is a **Good** or **Bad** credit risk, based on a dataset of representative instances from the German Credit dataset.

## Files Included

- `app.py`: The Streamlit web application script providing the UI for real-time predictions.
- `credit_risk_model.pkl`: The trained Random Forest machine learning pipeline (includes preprocessor + model).
- `credit_toy_dataset.csv`: The toy dataset of 100 representative instances extracted from the UCI German Credit dataset.
- `Untitled13.ipynb`: The Google Colab Notebook with the execution workflow covering data extraction, model training, and exporting the trained model.
- `requirements.txt`: Required Python dependencies for cloud deployment.

## Technical Implementation & Model Details

This project implements an end-to-end Machine Learning pipeline utilizing `scikit-learn`.

**1. The Dataset (German Credit Data):**
- **Source:** UCI Machine Learning Repository (via `fetch_openml`).
- **Target Variable:** The `class` feature (converted to binary `0 = Bad Risk`, `1 = Good Risk`).
- **Features Used:** 20 distinct applicant features consisting of both numerical data (e.g., `duration`, `credit_amount`, `age`) and categorical data (e.g., `checking_status`, `personal_status`, `housing`).
- **Sampling:** A "toy dataset" comprising 100 representative instances was extracted for rapid deployment testing.

**2. The Machine Learning Pipeline:**
To prevent data leakage and ensure seamless inference during deployment, a robust `Pipeline` was constructed:
- **Preprocessing (`ColumnTransformer`):**
  - **Numerical Features:** Scaled seamlessly using `StandardScaler`.
  - **Categorical Features:** Encoded using `OneHotEncoder` (with `handle_unknown='ignore'` to gracefully handle unexpected categories in production).
- **Classifier:** 
  - A `RandomForestClassifier` initialized with `100` estimators was utilized to learn the complex, non-linear dependencies determining credit risk.
- **Performance:** 
  - The model achieved **75.00% Accuracy** on the Hold-Out test set (20% split) of the sampled toy dataset.

**3. Deployment Environment:**
- The model and preprocessor are bundled together into `credit_risk_model.pkl` via `joblib`, making it instantly consumable by the Streamlit frontend.
- Strict dependency pinning (e.g., `scikit-learn==1.6.1`) ensures the cloud inference environment perfectly mirrors the training environment.

## Running Locally

To run the application on your local machine, follow these steps:

1. Setup the environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Cloud Deployment (Streamlit Community Cloud)


## Live Deployment Link

*Replace this section with your active URL once deployed*

**Deployment URL:** `https://984xb7jkaevbdqsuramefb.streamlit.app`
