from customer_churn.utils.logging_setup import logger
from customer_churn.config.configuration import ConfigurationManager
from customer_churn.components.model_evaluation.orchestrator import ModelEvaluation

class ModelEvaluationTrainingPipeline:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    def run_pipeline(self):
        evaluation_config = self.config_manager.get_model_evaluation_config()
        
        evaluator = ModelEvaluation(config=evaluation_config)
        evaluator.evaluate_all_models()