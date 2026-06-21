from customer_churn.components.data_ingestion import DataIngestion
from customer_churn.utils.logging_setup import logger

class DataIngestionTrainingPipeline:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def run_pipeline(self):
        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        
        data_ingestion = DataIngestion(config=data_ingestion_config)
        raw_dataframe = data_ingestion.download_data()
        logger.info("Data ingestion completed successfully.")
        logger.info(f"Raw data shape: {raw_dataframe.shape}")
        logger.info(f"Raw data columns: {raw_dataframe.columns.tolist()}")
        logger.info(f"Raw data preview:\n{raw_dataframe.head()}")
        logger.info(f"Raw data info:\n{raw_dataframe.info()}")
        logger.info(f"Raw data description:\n{raw_dataframe.describe()}")
        
        
        # Optionally pass this DataFrame forward or cache it to data/raw/
        return raw_dataframe