from abc import ABC, abstractmethod
import pandas as pd

class PreprocessingCommand(ABC):
    """Abstract interface defining an in-place engineering operations contract."""
    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass