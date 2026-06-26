# app.py
import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from customer_churn.pipeline.prediction import PredictionPipeline

# Set page layout configuration
st.set_page_config(
    page_title="Telco Churn Analytics Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("🔮 Telco Customer Churn Prediction Engine")
st.markdown("""
This production-grade analytics application leverages optimized machine learning pipelines 
to evaluate customer profiles and flag high-risk churn signals before they happen.
---
""")

# -------------------------------------------------------------------------
# SIDEBAR CONTROL INTERFACES
# -------------------------------------------------------------------------
st.sidebar.header("🛠️ Pipeline Configurations")

# Model Strategy Selector
available_models = ["xgboost", "random_forest", "logistic_regression", "svm"]
selected_model = st.sidebar.selectbox(
    "Choose Inference Framework Model:",
    options=available_models,
    index=0
)

# Display Current Evaluation Performance Summary Metrics if available
metrics_path = "artifacts/model_evaluation/metrics.json"
if os.path.exists(metrics_path):
    st.sidebar.markdown("### 📊 Active Model Performance")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
        if selected_model in all_metrics:
            m = all_metrics[selected_model]
            st.sidebar.metric(label="Test Accuracy", value=f"{m.get('accuracy', 0)*100:.2f}%")
            st.sidebar.metric(label="Balanced F1-Score", value=f"{m.get('f1_score', 0):.4f}")
            st.sidebar.metric(label="Model Recall (Sensitivity)", value=f"{m.get('recall', 0):.4f}")
    except Exception:
        st.sidebar.warning("Unable to fetch performance logs matrix.")

# -------------------------------------------------------------------------
# TAB CONTROL LAYOUT
# -------------------------------------------------------------------------
tab1, tab2 = st.tabs(["👤 Live Customer Evaluation", "📁 Bulk Batch Processing"])

# =========================================================================
# TAB 1: LIVE INDIVIDUAL CUSTOMER SCORING
# =========================================================================
with tab1:
    st.subheader("Simulate a Live Customer Profile")
    st.write("Modify the demographic and account dimensions below to compute real-time risk probabilities.")

    # Form configuration grid
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### **Demographics & Core Identity**")
        gender = st.selectbox("Gender:", ["Male", "Female"])
        senior_citizen = st.selectbox("Is Senior Citizen? (1=Yes, 0=No):", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure Months (Loyalty Window):", min_value=1, max_value=72, value=12)

    with col2:
        st.markdown("##### **Core Infrastructure Services**")
        phone_service = st.selectbox("Phone Service Subscription:", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines Setup:", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Network Class Type:", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security Feature:", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Cloud Backup Setup:", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("##### **Extended Services & Billing Metrics**")
        device_protection = st.selectbox("Device Protection Plan:", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Premium Tech Support Add-on:", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV Service package:", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movie Add-on Package:", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Terms Agreement Strategy:", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Electronic Invoicing:", ["Yes", "No"])
        payment_method = st.selectbox("Payment Gateway Method:", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        monthly_charges = st.number_input("Monthly Charge Vector Value ($):", min_value=10.0, max_value=150.0, value=65.0, step=0.5)
        
        # Approximate baseline total calculation dynamically
        calculated_total = round(monthly_charges * tenure, 2)
        total_charges = st.text_input("Total Historical Charges ($):", value=str(calculated_total))

    st.markdown("---")
    
    # Execution Button Trigger
    if st.button("🔮 Calculate Risk Exposure", type="primary"):
        # Explicit column tracking setup matching original source column labels exactly
        input_data = pd.DataFrame({
            "customerID": ["LIVE-SIM-01"],
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [int(tenure)],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [float(monthly_charges)],
            "TotalCharges": [str(total_charges)]
        })

        try:
            # Route inputs into prediction pipeline instance
            pipeline = PredictionPipeline(model_name=selected_model)
            prediction_output = pipeline.predict(input_data)
            
            # Display results via clear contrast UI notification layouts
            if prediction_output[0] == 1:
                st.error("🚨 **High Attrition Danger Flagged!** This subscriber matches established high-churn behavioral profiles.")
            else:
                st.success("✅ **Stable Account Status.** Subscriber exhibits steady retention parameters.")
                
        except Exception as e:
            st.error(f"Inference computation execution error: {str(e)}")

# =========================================================================
# TAB 2: BULK BATCH FILE PROCESSING MODE
# =========================================================================
with tab2:
    st.subheader("Batch Evaluation Pipeline Process Engine")
    st.write("Upload a raw un-preprocessed customer CSV file below to perform bulk evaluation predictions instantly.")

    uploaded_file = st.file_uploader("Upload raw customer data matrix CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_batch_df = pd.read_csv(uploaded_file)
            st.info(f"Loaded batch data matrix successfully! Found {raw_batch_df.shape[0]} customer records.")
            
            st.dataframe(raw_batch_df.head(5))

            if st.button("🚀 Execute Bulk Inference Suite", type="secondary"):
                with st.spinner("Processing data transformation features and running evaluations..."):
                    
                    pipeline = PredictionPipeline(model_name=selected_model)
                    predictions = pipeline.predict(raw_batch_df)
                    
                    output_df = raw_batch_df.copy()
                    output_df["Churn_Prediction_Output"] = predictions
                    output_df["Churn_Risk_Status"] = output_df["Churn_Prediction_Output"].map({1: "High Risk Churn", 0: "Loyal"})
                    
                    st.success("Bulk evaluations compiled successfully!")
                    
                    # Target showcase layout configuration
                    display_cols = [c for c in ["customerID", "tenure", "MonthlyCharges", "Churn_Risk_Status"] if c in output_df.columns]
                    st.dataframe(output_df[display_cols].head(10))
                    
                    # Convert processed frame array output into download stream artifact safely
                    csv_export = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Exported Target Predictions CSV",
                        data=csv_export,
                        file_name=f"batch_churn_predictions_{selected_model}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Batch transformation process crashed: {str(e)}")