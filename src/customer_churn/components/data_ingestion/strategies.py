import os
import pandas as pd
import requests
from customer_churn.components.data_ingestion.interface import IngestionStrategy
from customer_churn.utils.logging_setup import logger

class FileIngestionStrategy(IngestionStrategy):
    """Concrete strategy for handling local structured files."""
    def __init__(self, file_path: str, file_type: str):
        self.file_path = file_path
        self.file_type = file_type.lower()

    def extract(self) -> pd.DataFrame:
        logger.info(f"Extracting data from local file path: {self.file_path}")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Target dataset file missing at: {self.file_path}")
            
        if self.file_type == 'csv':
            return pd.read_csv(self.file_path)
        elif self.file_type in ['excel', 'xlsx']:
            return pd.read_excel(self.file_path)
        elif self.file_type == 'json':
            return pd.read_json(self.file_path)
        else:
            raise ValueError(f"Unsupported file type extension configured: {self.file_type}")

class ApiIngestionStrategy(IngestionStrategy):
    """Concrete strategy for fetching streaming or batch payloads from secure REST APIs."""
    def __init__(self, endpoint: str, headers: dict = None, params: dict = None):
        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        self.params = params or {}

    def extract(self) -> pd.DataFrame:
        logger.info(f"Initiating out-of-network HTTP GET request to endpoint: {self.endpoint}")
        try:
            response = requests.get(self.endpoint, headers=self.headers, params=self.params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            # Normalize complex nested JSON configurations safely into flat matrices
            if isinstance(data, dict):
                # Search for typical record array keys if returned inside an envelope wrapper
                for key in ['data', 'records', 'results']:
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])
                return pd.DataFrame([data])
            elif isinstance(data, list):
                return pd.DataFrame(data)
                
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed during API extraction sequence: {str(e)}")
            raise RuntimeError(f"API Extraction error: {str(e)}") from e