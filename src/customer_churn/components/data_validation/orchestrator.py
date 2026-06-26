# src/customer_churn/components/data_validation/orchestrator.py
import pandas as pd
from customer_churn.entity.config_entity import DataValidationConfig
from customer_churn.components.data_validation.strategies import SchemaValidationStrategy, DataQualityValidationStrategy
from customer_churn.utils.logging_setup import logger

class DataValidation:
    """Entry point for the validation component handling multi-strategy execution sequences."""
    def __init__(self, config: DataValidationConfig):
        self.config = config
        # Initialized with a strict 10% null limit for production readiness
        self.strategies = [
            SchemaValidationStrategy(expected_schema=self.config.all_schema),
            DataQualityValidationStrategy(max_null_ratio=0.10)
        ]

    def validate_all(self, df: pd.DataFrame) -> bool:
        try:
            overall_status = True
            validation_messages = []

            for strategy in self.strategies:
                status, msg = strategy.validate(df)
                validation_messages.append(msg)
                if not status:
                    overall_status = False
                    logger.error(f"Validation step failed: {msg}")

            # Safe write status log output
            with open(self.config.STATUS_FILE, 'w', encoding='utf-8') as f:
                f.write(f"Validation status: {overall_status}\n")
                f.write("\n".join(validation_messages))
                
            if overall_status:
                logger.info("All data validation strategies passed successfully.")
                
            return overall_status

        except Exception as e:
            logger.error(f"Error during data validation execution flow: {str(e)}")
            raise e