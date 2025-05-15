import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk Prediction")

st.title("🩺 Diabetes Risk Prediction")
st.markdown("Upload patient data in CSV format to predict the risk of diabetes.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(input_data.head())

    model_name = st.selectbox("Choose Prediction Model", ["logistic", "random_forest", "xgboost", "svm"])
    model = joblib.load(f"models/{model_name}.pkl")

    predictions = model.predict_proba(input_data)[:, 1]
    st.write("### Predicted Diabetes Risk Scores")
    st.dataframe(pd.DataFrame(predictions, columns=["Risk Score"]))