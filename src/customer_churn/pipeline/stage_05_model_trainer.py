from customer_churn.utils.logging_setup import logger
from customer_churn.config.configuration import ConfigurationManager
from customer_churn.components.model_trainer.orchestrator import ModelTrainer

class ModelTrainerTrainingPipeline:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    def run_pipeline(self):
        logger.info("Loading configuration parameters for parallel estimator model training...")
        trainer_config = self.config_manager.get_model_trainer_config()
        
        trainer = ModelTrainer(config=trainer_config)
        trainer.train_all_models()