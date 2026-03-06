# Credit Risk Predictor 🏦

This repository contains the deployment files for **Assignment 3: NLP & ML Project Development**. 
A machine learning model trained to predict whether a credit applicant is a **Good** or **Bad** credit risk, based on a dataset of representative instances from the German Credit dataset.

## Files Included

- `app.py`: The Streamlit web application script providing the UI for real-time predictions.
- `credit_risk_model.pkl`: The trained Random Forest machine learning pipeline (includes preprocessor + model).
- `credit_toy_dataset.csv`: The toy dataset of 100 representative instances extracted from the UCI German Credit dataset.
- `Untitled13.ipynb`: The Google Colab Notebook with the execution workflow covering data extraction, model training, and exporting the trained model.
- `requirements.txt`: Required Python dependencies for cloud deployment.

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

To deploy your finalized ML model so it can be accessed live on the web:

1. **Push to GitHub**: Upload all the files in this directory (`app.py`, `credit_risk_model.pkl`, `credit_toy_dataset.csv`, `requirements.txt`) to a public GitHub repository.
2. **Sign up / Log in to Streamlit**: Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. **Deploy the App**:
   - Click **New app**.
   - Select the GitHub repository and branch you just created.
   - Set the Main file path to `app.py`.
   - Click **Deploy!**
4. Your application will be live in a few minutes.

## Live Deployment Link

*Replace this section with your active URL once deployed*

**Deployment URL:** `[Insert your PythonAnywhere or Streamlit Cloud URL here]`
