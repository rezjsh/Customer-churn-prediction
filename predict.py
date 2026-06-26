import pandas as pd
import logging
from src.customer_churn.pipeline.prediction import PredictionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

if __name__ == "__main__":
    # Simulate a raw real-time incoming customer observation payload
    mock_customer_payload = pd.DataFrame({
        "customerID": ["7016-ZZZZZ"],
        "gender": ["Female"],
        "SeniorCitizen": [1],
        "Partner": ["No"],
        "Dependents": ["No"],
        "tenure": [3],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [94.85],
        "TotalCharges": ["284.55"]
    })

    try:
        # Request inference using your desired model backend ('xgboost', 'random_forest', 'logistic_regression', 'svm')
        predictor = PredictionPipeline(model_name="xgboost")
        predictions = predictor.predict(mock_customer_payload)
        
        status_label = "Churn Risk (Yes)" if predictions[0] == 1 else "Loyal Customer (No)"
        print(f"\n Execution Result: {predictions} -> Mapped Class Outcome: {status_label}\n")
        
    except Exception as e:
        logging.error(f"Prediction run execution pipeline crashed: {e}")