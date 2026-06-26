import os
import pandas as pd
from customer_churn.entity.config_entity import DataIngestionConfig
from customer_churn.components.data_ingestion.factory import IngestionStrategyFactory
from customer_churn.utils.logging_setup import logger

class DataIngestion:
    """The data ingestion component's entrypoint context coordinator class."""
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> pd.DataFrame:
        try:
            # 1. Dynamically provision structural directories if missing
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            # 2. Query factory for the appropriate runtime execution driver
            strategy = IngestionStrategyFactory.get_strategy(self.config)
            
            # 3. Execute data download extraction flow
            df = strategy.extract()
            
            logger.info(f"Ingestion driver completed successfully. Features shape dimensions: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Fatal error executing data ingestion component matrix: {str(e)}")
            raise e