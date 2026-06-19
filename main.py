from src.customer_churn.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from customer_churn.utils.logging_setup import logger

STAGE_NAME = "Data Ingestion Stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        pipeline = DataIngestionTrainingPipeline()
        data = pipeline.run()
        
        logger.info(f">>>>>> Stage {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e