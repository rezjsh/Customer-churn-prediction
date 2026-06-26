# src/customer_churn/components/data_validation/strategies.py
import pandas as pd
from customer_churn.components.data_validation.interface import ValidationStrategy
from customer_churn.utils.logging_setup import logger

class SchemaValidationStrategy(ValidationStrategy):
    """Ensures all expected columns exist and match expected data types, with structural exceptions."""
    def __init__(self, expected_schema: dict):
        self.expected_schema = expected_schema

    def validate(self, df: pd.DataFrame) -> tuple[bool, str]:
        logger.info("Executing Schema Validation Strategy...")
        missing_cols = []
        type_mismatches = []
        
        text_types = ['object', 'str', 'string', 'O']
        
        for col, expected_type in self.expected_schema.items():
            if col not in df.columns:
                missing_cols.append(col)
                continue
                
            actual_type = str(df[col].dtype)
            
            is_valid = False
            if actual_type == expected_type:
                is_valid = True
            elif expected_type in text_types and actual_type in text_types:
                is_valid = True
            # SPECIAL CASE FOR TELCO: Allow float expected schemas to see raw objects if they are fixable strings
            elif col == "TotalCharges" and expected_type in ['float64', 'float'] and actual_type in text_types:
                logger.warning("TotalCharges discovered as object type. Flagged for downstream numeric transformation transformation coercion.")
                is_valid = True
                
            if not is_valid:
                type_mismatches.append(f"{col} (Expected: {expected_type}, Got: {actual_type})")

        if missing_cols or type_mismatches:
            msg = f"Schema validation failed. Missing cols: {missing_cols}. Type mismatches: {type_mismatches}"
            return False, msg
            
        return True, "Schema validation passed."


class DataQualityValidationStrategy(ValidationStrategy):
    """Production-grade validator verifying missing bound limitations and class alignments."""
    def __init__(self, max_null_ratio: float = 0.10):
        self.max_null_ratio = max_null_ratio

    def validate(self, df: pd.DataFrame) -> tuple[bool, str]:
        logger.info("Executing Advanced Data Quality Validation Strategy...")
        
        if df.empty:
            return False, "Data quality failed: DataFrame is entirely empty."
            
        # 1. Null rate check across all columns
        for col in df.columns:
            # Handle standard missing values plus raw space strings
            null_count = df[col].isnull().sum()
            if df[col].dtype == 'object':
                null_count += (df[col] == " ").sum()
                
            null_ratio = null_count / len(df)
            if null_ratio > self.max_null_ratio:
                return False, f"Data quality failed: Column '{col}' exceeds maximum null threshold ({null_ratio:.2%} > {self.max_null_ratio:.2%})"

        # 2. Strict target validation
        if 'Churn' in df.columns:
            if df['Churn'].isnull().any():
                return False, "Data quality failed: Target column 'Churn' contains missing values."
                
            unique_classes = set(df['Churn'].dropna().unique())
            expected_classes = {"Yes", "No"}
            if not unique_classes.issubset(expected_classes):
                return False, f"Data quality failed: Target 'Churn' contains invalid classes {unique_classes - expected_classes}."
        else:
            return False, "Data quality failed: Missing required target variable 'Churn'."

        return True, "Data quality checks passed successfully."