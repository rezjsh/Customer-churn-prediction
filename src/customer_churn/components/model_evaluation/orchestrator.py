# src/customer_churn/components/model_evaluation/orchestrator.py
import os
import json
import logging
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from customer_churn.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_all_models(self):
        logging.info("Model Evaluation: Loading processed testing matrix splits.")
        if not os.path.exists(self.config.test_data_path):
            raise FileNotFoundError(f"Missing required test data matrix at {self.config.test_data_path}")
            
        test_df = pd.read_csv(self.config.test_data_path)
        
        # Isolate target variable
        target_col = self.config.target_column
        y_test = test_df[target_col]
        
        evaluation_summary = {}

        # Scan directory for trained serialized model binaries
        if not os.path.exists(self.config.model_dir):
            raise FileNotFoundError(f"Model storage directory missing: {self.config.model_dir}")
            
        model_files = [f for f in os.listdir(self.config.model_dir) if f.endswith(".joblib")]
        
        if not model_files:
            logging.warning(f"No trained models found in {self.config.model_dir}. Skipping evaluation metrics extraction.")
            return

        for model_file in model_files:
            model_name = model_file.replace("_model.joblib", "")
            logging.info(f"--- Unpacking and Evaluating Candidate Model: {model_name} ---")
            
            # Load the complete artifact bundle
            artifact_bundle = joblib.load(os.path.join(self.config.model_dir, model_file))
            
            # Check if artifact is bundled or bare (legacy protection)
            if isinstance(artifact_bundle, dict) and "estimator" in artifact_bundle:
                estimator = artifact_bundle["estimator"]
                expected_features = artifact_bundle["features_schema"]
            else:
                logging.warning(f"Legacy unbundled model artifact detected for {model_name}. Proceeding without strict feature alignment validation.")
                estimator = artifact_bundle
                expected_features = [c for c in test_df.columns if c != target_col]

            # Re-isolate and strictly align the test features to match the exact training matrix layout
            X_test = test_df.drop(columns=[target_col], errors='ignore')
            
            # Align features order and fill any missing dummy features with 0
            X_test = X_test.reindex(columns=expected_features, fill_value=0)

            # Ensure all boolean data indicators are integers to satisfy strict framework parsers (e.g., XGBoost)
            for col in X_test.columns:
                if X_test[col].dtype == 'bool':
                    X_test[col] = X_test[col].astype(int)

            # Generate evaluation predictions
            predictions = estimator.predict(X_test)
            
            # Handle probabilistic tracking securely across model archetypes
            try:
                probs = estimator.predict_proba(X_test)[:, 1]
                auc_score = round(float(roc_auc_score(y_test, probs)), 4)
            except (AttributeError, IndexError):
                logging.warning(f"Model {model_name} does not natively support class probabilities output. Setting ROC-AUC to N/A.")
                auc_score = "N/A"

            # Populate evaluation summaries map
            evaluation_summary[model_name] = {
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
                "roc_auc": auc_score
            }

        # Ensure output metrics directory structure exists
        os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)

        # Serialize complete metrics report map directly to disk
        with open(self.config.metric_file_name, "w", encoding="utf-8") as f:
            json.dump(evaluation_summary, f, indent=4)
            
        logging.info(f"Model Evaluation complete. Metrics successfully logged to: {self.config.metric_file_name}")
        logging.info(f"Final Evaluation Report Summary Summary:\n{json.dumps(evaluation_summary, indent=2)}")