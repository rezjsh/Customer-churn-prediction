import os
from customer_churn.components.data_ingestion.orchestrator import DataIngestion
from customer_churn.utils.logging_setup import logger

class DataIngestionTrainingPipeline:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def run_pipeline(self):
        logger.info("Initializing configuration manager properties for Data Ingestion Stage...")
        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        
        # Initialize component orchestrator
        data_ingestion = DataIngestion(config=data_ingestion_config)
        raw_dataframe = data_ingestion.download_data()
        
        # Save the dataset to the configured local workspace to create a persistent audit trail
        output_path = data_ingestion_config.file_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        raw_dataframe.to_csv(output_path, index=False)
        logger.info(f"Raw data cached locally to target path: {output_path}")
        
        # Print dataset diagnostics and layout metrics to the logger
        logger.info("--- Ingested Raw DataFrame Profile Summary ---")
        logger.info(f"Matrix Dimension Formats: {raw_dataframe.shape}")
        logger.info(f"Column Manifest Layout: {raw_dataframe.columns.tolist()}")
        
        return raw_dataframe