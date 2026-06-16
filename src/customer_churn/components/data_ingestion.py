import os
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split

# Adjust these imports based on your exact project structure
from src.entity.config_entity import DataIngestionConfig
from src.utils.logging_setup import logger
from src.utils.helpers import download_file, save_dataframe

class DataIngestion:
    """
    Handles fetching the raw Telco Customer Churn data and creating reproducible splits.
    Supports automatic downloading from a remote source if the local file is missing.
    """
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """
        Entry point to load the Telco dataset. Downloads the raw CSV if it 
        doesn't exist locally at the configured path.
        """
        logger.info("Initiating data loading process...")
        
        # Check if file exists locally, if not, download it
        if not os.path.exists(self.config.local_data_path):
            logger.warning(f"Data not found at {self.config.local_data_path}. Attempting to download...")
            os.makedirs(os.path.dirname(self.config.local_data_path), exist_ok=True)
            success = download_file(self.config.source_url, self.config.local_data_path)
            
            if not success:
                raise Exception("Failed to download the dataset. Please check your source_url.")
        
        # Load the CSV
        try:
            df = pd.read_csv(self.config.local_data_path)
            logger.info(f"Telco Churn dataset loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading the dataset: {e}")
            raise e

    def create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Creates Train, Validation, and Test splits.
        Calculates relative sizes to ensure final proportions match config.
        """
        train_ratio = 1 - self.config.test_size - self.config.val_size
        logger.info(f"Splitting data into Train/Val/Test ({train_ratio:.2f}/{self.config.val_size}/{self.config.test_size})")

        # 1. Separate Test set
        # Stratify on 'Churn' target variable to maintain class imbalance proportions
        target_col = 'Churn'
        stratify_param = df[target_col] if target_col in df.columns else None
        
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
            stratify=stratify_param
        )

        # 2. Separate Validation set from the remaining training data
        # Calculate relative val_size
        relative_val_size = self.config.val_size / (1 - self.config.test_size)
        
        # Stratify again for the Val split
        stratify_val_param = train_val_df[target_col] if target_col in train_val_df.columns else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
            stratify=stratify_val_param
        )

        logger.info(f"Split results: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True)
        }
        
    def initiate_data_ingestion(self) -> Dict[str, str]:
        """
        Master function to orchestrate loading, splitting, and saving the splits to disk.
        Returns the paths of the saved splits.
        """
        df = self.load_data()
        splits = self.create_splits(df)
        
        # Save splits using the helper function we built earlier
        save_dataframe(splits['train'], self.config.train_data_path, file_format="csv")
        save_dataframe(splits['val'], self.config.val_data_path, file_format="csv")
        save_dataframe(splits['test'], self.config.test_data_path, file_format="csv")
        
        logger.info("Data Ingestion phase completed successfully.")
        
        return {
            "train_path": self.config.train_data_path,
            "val_path": self.config.val_data_path,
            "test_path": self.config.test_data_path
        }