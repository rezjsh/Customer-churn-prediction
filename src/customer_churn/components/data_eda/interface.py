from abc import ABC, abstractmethod
import pandas as pd


class AnalysisComponent(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, config) -> dict:
        pass