# src/customer_churn/components/data_transformation/orchestrator.py
import os
from customer_churn.utils.logging_setup import logger
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from customer_churn.entity.config_entity import DataTransformationConfig
from customer_churn.components.data_transformation.operations import (
    DataCleaningCommand, FeatureEngineeringCommand, EncodingAndScalingCommand
)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
        # Instantiate commands at initialization to persist internal state parameters safely
        self.cleaner = DataCleaningCommand(self.config.numerical_features, strategy=self.config.imputation_strategy)
        self.engineer = FeatureEngineeringCommand()
        self.transformer = EncodingAndScalingCommand(self.config.numerical_features, self.config.categorical_features)

    def execute_preprocessing_and_split(self, df: pd.DataFrame):
        """Processes raw matrices securely, eliminating potential distribution validation leakages."""
        target = self.config.target_column
        df = df.copy()

        # 1. Preliminary Cleaning & Generation (Row-independent)
        df = self.cleaner.execute(df)
        df = self.engineer.execute(df)

        # 2. Binary target variable layout standardization
        if target in df.columns:
            df[target] = df[target].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        else:
            raise KeyError(f"Target variable tracking column '{target}' absent from processing payload.")

        # 3. Partition Data Frame matrices BEFORE scaling or encoding
        train_df, test_df = train_test_split(
            df, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=df[target]
        )

        # 4. Apply Stateful Encoder/Scaler Operations
        logger.info("Applying stateful scaling and encoding transformations to Train Split...")
        train_processed = self.transformer.execute(train_df)
        
        logger.info("Applying identical transformation parameters to Test Split...")
        test_processed = self.transformer.execute(test_df)

        # 5. Persist splits safely to artifacts storage
        os.makedirs(self.config.root_dir, exist_ok=True)
        train_path = os.path.join(self.config.root_dir, self.config.train_file_name)
        test_path = os.path.join(self.config.root_dir, self.config.test_file_name)
        
        train_processed.to_csv(train_path, index=False)
        test_processed.to_csv(test_path, index=False)
        
        logger.info(f"Data Transformation completed successfully. Datasets saved to: {self.config.root_dir}")
        logger.info(f"Train dataset shapes: {train_processed.shape} | Test dataset shapes: {test_processed.shape}")

        preprocessing_artifacts = {
            "cleaner_impute_values": self.cleaner.impute_values,
            "scaler": self.transformer.scaler,
            "final_columns": self.transformer.final_columns,
            "numerical_features": self.config.numerical_features,
            "categorical_features": self.config.categorical_features
        }
        artifacts_path = os.path.join(self.config.root_dir, "preprocessing_artifacts.joblib")
        joblib.dump(preprocessing_artifacts, artifacts_path)
        logger.info(f"Stateful preprocessing metadata saved to: {artifacts_path}")