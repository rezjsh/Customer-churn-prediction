# src/customer_churn/pipeline/prediction.py
import os
import joblib
import pandas as pd
import numpy as np
from customer_churn.utils.logging_setup import logger
from customer_churn.config.configuration import ConfigurationManager

class PredictionPipeline:
    def __init__(self, model_name: str = "xgboost"):
        """
        Production-grade inference execution pipeline engine.
        Ensures perfect reconstruction of training topologies during real-time batch/stream inference.
        """
        self.model_name = model_name.lower()
        self.config_manager = ConfigurationManager()
        self.transform_config = self.config_manager.get_data_transformation_config()
        self.eval_config = self.config_manager.get_model_evaluation_config()
        
        # Define paths to saved historical training matrices state layers
        self.prep_artifacts_path = os.path.join(self.transform_config.root_dir, "preprocessing_artifacts.joblib")
        self.model_path = os.path.join(self.eval_config.model_dir, f"{self.model_name}_model.joblib")
        
        # Load all static operational parameter transformations upfront
        self._load_pipeline_components()

    def _load_pipeline_components(self):
        """Loads fitted state parameters safely to prevent downstream runtime failures."""
        if not os.path.exists(self.prep_artifacts_path):
            raise FileNotFoundError(f"Stateful preprocessing parameters absent at {self.prep_artifacts_path}. Run Data Transformation first.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model checkpoint binary absent at {self.model_path}. Run Model Training first.")
            
        self.artifacts = joblib.load(self.prep_artifacts_path)
        self.model_bundle = joblib.load(self.model_path)

    def _preprocess_raw_input(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms un-preprocessed payloads using stateful parameters from training
        to eliminate data distribution bias and input scale errors.
        """
        df = raw_df.copy()
        
        # 1. Clean identity indicators safely
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
            
        # 2. Stateful cleaning and alignment via saved imputation metrics
        num_features = self.artifacts["numerical_features"]
        impute_values = self.artifacts["cleaner_impute_values"]
        
        for col in num_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Guarantee imputation matches training distributions, even for single rows
                fill_val = impute_values.get(col, 0)
                df[col] = df[col].fillna(fill_val)
                
        # 3. Inject missing domain logic columns identically
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            df["Charges_Per_Tenure"] = df["MonthlyCharges"] / (df["tenure"] + 0.1)
        else:
            df["Charges_Per_Tenure"] = 0.0
            
        if "Contract" in df.columns:
            df["Is_LongTerm_Contract"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)
        else:
            df["Is_LongTerm_Contract"] = 0

        # 4. Stateful Numeric Scaling (using fitted training instance weights)
        scaler = self.artifacts["scaler"]
        num_targets = [c for c in num_features if c in df.columns] + ["Charges_Per_Tenure"]
        df[num_targets] = scaler.transform(df[num_targets])
        
        # 5. Safe Categorical One-Hot Generation
        cat_features = self.artifacts["categorical_features"]
        cat_targets = [c for c in cat_features if c in df.columns]
        if cat_targets:
            df = pd.get_dummies(df, columns=cat_targets, drop_first=True)
            
        return df

    def _align_features_to_model_spec(self, processed_df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
        """Forces testing matrices to mirror training topology schemas explicitly."""
        # Align column indexes and structural placement layout perfectly
        aligned_df = processed_df.reindex(columns=expected_features, fill_value=0)
        
        # Coerce boolean types to clean integer formats to protect models like XGBoost
        for col in aligned_df.columns:
            if aligned_df[col].dtype == 'bool':
                aligned_df[col] = aligned_df[col].astype(int)
                
        return aligned_df

    def predict(self, raw_features_df: pd.DataFrame) -> np.ndarray:
        """
        Main entry point for web APIs and streaming instances. 
        Transforms incoming payloads and yields clean inference predictions.
        """
        logger.info(f"Inference: Input prediction batch signature received: {raw_features_df.shape}")
        
        # 1. Safely unpack our model bundle structure
        if isinstance(self.model_bundle, dict) and "estimator" in self.model_bundle:
            estimator = self.model_bundle["estimator"]
            expected_features = self.model_bundle["features_schema"]
        else:
            estimator = self.model_bundle
            expected_features = self.artifacts["final_columns"]

        # 2. Run data transformations matching original configurations
        processed_data = self._preprocess_raw_input(raw_features_df)
        
        # 3. Align input feature matrix order with the model's expected layout
        final_inference_matrix = self._align_features_to_model_spec(processed_data, expected_features)
        
        # 4. Generate clean, predictable model predictions
        predictions = estimator.predict(final_inference_matrix)
        logger.info(f"Inference complete. Successfully generated predictions array for {len(predictions)} cases.")
        
        return predictions