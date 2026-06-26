from customer_churn.config.configuration import ConfigurationManager
from customer_churn.components.data_validation.orchestrator import DataValidation
from customer_churn.utils.logging_setup import logger
import pandas as pd

class DataValidationPipeline:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    def run_pipeline(self, dataframe: pd.DataFrame) -> bool:
        data_validation_config = self.config_manager.get_data_validation_config()
        
        data_validation = DataValidation(config=data_validation_config)
        status = data_validation.validate_all(df=dataframe)
        
        if not status:
            logger.warning("Pipeline halted: Data validation returned False.")
            raise ValueError("Data Validation failed. Check the status file for details.")
            
        return status