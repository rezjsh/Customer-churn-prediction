from customer_churn.config.configuration import ConfigurationManager
from customer_churn.pipeline.stage_02_data_eda import DataEDAPipeline
from customer_churn.pipeline.stage_03_data_validation import DataValidationPipeline
from customer_churn.pipeline.stage_04_data_transformation import DataTransformationTrainingPipeline
from customer_churn.pipeline.stage_05_model_trainer import ModelTrainerTrainingPipeline
from customer_churn.pipeline.stage_06_model_evaluation import ModelEvaluationTrainingPipeline
from src.customer_churn.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from customer_churn.utils.logging_setup import logger



def main():
    """
    Main execution function to orchestrate the Data Ingestion stage.
    """
    try:
        config_manager = ConfigurationManager()

        # --- Stage 1: Data Ingestion ---
        STAGE_NAME = "Stage 01: Data Ingestion"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionTrainingPipeline(config_manager)
        raw_df = data_ingestion_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # --- Stage 2: Data EDA ---
        STAGE_NAME = "Stage 02: Data EDA"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        eda_pipeline = DataEDAPipeline(config_manager)
        eda_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # --- Stage 3: Data Validation ---
        logger.info(">>>>>> Stage 03: Data Validation started <<<<<<")
        validation_pipeline = DataValidationPipeline(config_manager=config_manager)
        # We validate the processed data. If you prefer validating raw data, pass raw_data instead.
        validation_status = validation_pipeline.run_pipeline(dataframe=raw_df) 
        logger.info(">>>>>> Stage 03: Data Validation completed <<<<<<\n")

        # --- Stage 4: Data Transformation ---
        if validation_status:
            logger.info(">>>>>> Stage 04: Data Transformation started <<<<<<")
            transformation_pipeline = DataTransformationTrainingPipeline(config_manager=config_manager)
            transformation_pipeline.run_pipeline(raw_dataframe=raw_df)
            logger.info(">>>>>> Stage 04: Data Transformation completed <<<<<<\n")
        else:
            logger.error("Data Validation failed. Aborting Data Transformation stage.")

        
        # --- Stage 5: Model Training ---
        logger.info(">>>>>> Stage 05: Model Training started <<<<<<")
        model_trainer_pipeline = ModelTrainerTrainingPipeline(config_manager=config_manager)
        model_trainer_pipeline.run_pipeline()
        logger.info(">>>>>> Stage 05: Model Training completed <<<<<<\n")

        # --- Stage 6: Model Evaluation ---
        logger.info(">>>>>> Stage 06: Model Evaluation started <<<<<<")
        evaluation_pipeline = ModelEvaluationTrainingPipeline(config_manager=config_manager)
        evaluation_pipeline.run_pipeline()
        logger.info(">>>>>> Stage 06: Model Evaluation completed <<<<<<\n")

    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
   main()