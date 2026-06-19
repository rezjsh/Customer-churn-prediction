import pandas as pd
from customer_churn.components.data_ingestion.interface import IngestionStrategy
from customer_churn.utils.logging_setup import logger

class FileIngestionStrategy(IngestionStrategy):
    """Concrete strategy for handling local files."""
    def __init__(self, file_path: str, file_type: str):
        self.file_path = file_path
        self.file_type = file_type.lower()

    def extract(self) -> pd.DataFrame:
        logger.info(f"Extracting data from local file: {self.file_path}")
        if self.file_type == 'csv':
            return pd.read_csv(self.file_path)
        raise ValueError(f"Unsupported file type: {self.file_type}")

class ApiIngestionStrategy(IngestionStrategy):
    """Concrete strategy for fetching from REST APIs."""
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def extract(self) -> pd.DataFrame:
        logger.info(f"Extracting data from API endpoint: {self.endpoint}")
        # API extraction logic here...
        return pd.DataFrame()