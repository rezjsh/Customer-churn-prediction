# src/customer_churn/components/data_transformation/operations.py
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from customer_churn.components.data_transformation.interface import PreprocessingCommand

class DataCleaningCommand(PreprocessingCommand):
    """Handles irrelevant column dropping, type casting, and strategic missing value imputation."""
    def __init__(self, numerical_cols: list, strategy: str = "median"):
        self.numerical_cols = numerical_cols
        self.strategy = strategy.lower()
        self.impute_values = {}  # Save stateful values to reuse on test data

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Preprocessing: Executing data cleaning with '{self.strategy}' imputation strategy.")
        df = df.copy()
        
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
            
        for col in self.numerical_cols:
            if col in df.columns:
                # Force blank spaces (like in TotalCharges) or bad text to real NaNs
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate impute value if running for the first time (Training Mode)
                if col not in self.impute_values:
                    if self.strategy == "median":
                        self.impute_values[col] = df[col].median()
                    elif self.strategy == "mean":
                        self.impute_values[col] = df[col].mean()
                    else:
                        mode_series = df[col].mode()
                        self.impute_values[col] = mode_series.iloc[0] if not mode_series.empty else 0
                
                # Apply stored imputation value (prevents leakage on test/inference data)
                df[col] = df[col].fillna(self.impute_values[col])
                
        return df


class FeatureEngineeringCommand(PreprocessingCommand):
    """Synthesizes engineered business indicators to uncover nonlinear predictive trends."""
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Preprocessing: Synthesizing engineered domain features.")
        df = df.copy()
        
        # Avoid division-by-zero anomalies via safe floor values
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            df["Charges_Per_Tenure"] = df["MonthlyCharges"] / (df["tenure"] + 0.1)
            
        if "Contract" in df.columns:
            df["Is_LongTerm_Contract"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)
            
        return df


class EncodingAndScalingCommand(PreprocessingCommand):
    """Applies standard scaling to numerical vectors and one-hot tracking to categorical groups."""
    def __init__(self, numerical_cols: list, categorical_cols: list):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.final_columns = []  # Tracks exact categorical schema layout across splits

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Preprocessing: Aligning Categorical Encodings and Scaling Numeric Continuums.")
        df = df.copy()
        
        num_targets = [c for c in self.numerical_cols if c in df.columns]
        if "Charges_Per_Tenure" in df.columns:
            num_targets.append("Charges_Per_Tenure")
            
        # 1. Stateful Numeric Scaling
        if num_targets:
            if not self.is_fitted:
                df[num_targets] = self.scaler.fit_transform(df[num_targets])
            else:
                df[num_targets] = self.scaler.transform(df[num_targets])
                
        # 2. Schema-Insulated One-Hot Encoding
        cat_targets = [c for c in self.categorical_cols if c in df.columns]
        if cat_targets:
            # Perform standard encoding
            df_encoded = pd.get_dummies(df, columns=cat_targets, drop_first=True)
            
            if not self.is_fitted:
                # Capture the explicit structural layout during training
                self.final_columns = df_encoded.columns.tolist()
                self.is_fitted = True
                return df_encoded
            else:
                # Reindex test/inference rows to ensure it perfectly matches the training vector columns
                df_encoded = df_encoded.reindex(columns=self.final_columns, fill_value=0)
                return df_encoded
                
        return df