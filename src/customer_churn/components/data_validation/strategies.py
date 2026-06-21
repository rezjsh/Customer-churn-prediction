import pandas as pd
from customer_churn.components.data_validation.interface import ValidationStrategy
from customer_churn.utils.logging_setup import logger

class SchemaValidationStrategy(ValidationStrategy):
    """Ensures all expected columns exist and match expected data types."""
    def __init__(self, expected_schema: dict):
        self.expected_schema = expected_schema

    def validate(self, df: pd.DataFrame) -> tuple[bool, str]:
        logger.info("Executing Schema Validation Strategy...")
        missing_cols = []
        type_mismatches = []
        
        for col, expected_type in self.expected_schema.items():
            if col not in df.columns:
                missing_cols.append(col)
            else:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    type_mismatches.append(f"{col} (Expected: {expected_type}, Got: {actual_type})")

        if missing_cols or type_mismatches:
            msg = f"Schema validation failed. Missing cols: {missing_cols}. Type mismatches: {type_mismatches}"
            return False, msg
            
        return True, "Schema validation passed."


class DataQualityValidationStrategy(ValidationStrategy):
    """Checks for data quality rules like empty dataframes or excessive missing values."""
    def validate(self, df: pd.DataFrame) -> tuple[bool, str]:
        logger.info("Executing Data Quality Validation Strategy...")
        
        if df.empty:
            return False, "Data quality failed: DataFrame is entirely empty."
            
        # Example rule: Reject if target column 'Churn' is entirely missing
        if 'Churn' in df.columns and df['Churn'].isnull().all():
            return False, "Data quality failed: Target column 'Churn' contains only null values."

        return True, "Data quality checks passed."