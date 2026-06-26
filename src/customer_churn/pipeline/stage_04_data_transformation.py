from customer_churn.utils.logging_setup import logger
from customer_churn.config.configuration import ConfigurationManager
from customer_churn.components.data_transformation.orchestrator import DataTransformation

class DataTransformationTrainingPipeline:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    def run_pipeline(self, raw_dataframe):
        logger.info("Accessing configuration managers parameters for Data Transformation lifecycle...")
        transformation_config = self.config_manager.get_data_transformation_config()
        
        transformer = DataTransformation(config=transformation_config)
        transformer.execute_preprocessing_and_split(raw_dataframe)