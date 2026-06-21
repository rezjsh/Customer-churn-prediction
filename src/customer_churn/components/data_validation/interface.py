from abc import ABC, abstractmethod
import pandas as pd

class ValidationStrategy(ABC):
    """Abstract interface defining the contract for validation checks."""
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Returns a tuple of (is_valid: bool, validation_message: str)"""
        pass