from abc import ABC, abstractmethod
import pandas as pd

class IngestionStrategy(ABC):
    """Abstract interface defining the contract for all ingestion methods."""
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        pass