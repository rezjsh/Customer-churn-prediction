import pandas as pd
from customer_churn.entity.config_entity import DataIngestionConfig
from customer_churn.components.data_ingestion.factory import IngestionStrategyFactory
from customer_churn.utils.logging_setup import logger

class DataIngestion:
    """The component's entrypoint class called by the pipeline stage."""
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> pd.DataFrame:
        try:
            # 1. Ask factory for the strategy
            strategy = IngestionStrategyFactory.get_strategy(self.config)
            
            # 2. Execute the strategy contract
            df = strategy.extract()
            
            logger.info(f"Data ingestion completed successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error executing data ingestion component: {str(e)}")
            raise e