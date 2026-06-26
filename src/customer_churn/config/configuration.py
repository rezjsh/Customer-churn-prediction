from customer_churn.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from customer_churn.utils.common import create_directory, read_yaml_file  
from customer_churn.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, EDAConfig, ModelEvaluationConfig, ModelTrainerConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH, schema_filepath = SCHEMA_FILE_PATH):
        self.config = read_yaml_file(Path(config_filepath))
        self.params = read_yaml_file(Path(params_filepath))
        self.schema = read_yaml_file(Path(schema_filepath))

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_type=config.source_type,
            file_path=Path(config.file_path),
            file_type=config.file_type
        )
    
    def get_eda_config(self) -> EDAConfig:
        config = self.config.data_eda
        params = self.params.eda_params

        create_directory([config.root_dir])

        return EDAConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            json_report_path=Path(config.json_report_path),
            text_report_path=Path(config.text_report_path),
            target_column=params.target_column,
            correlation_threshold=params.correlation_threshold,
            outlier_method=params.outlier_method
        )
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.columns

        dirs_to_create = [config.root_dir]
        create_directory(dirs_to_create)

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=Path(config.unzip_data_dir),
            all_schema=schema,
        )
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.Data_Preprocessing
        
        dirs_to_create = [config.root_dir]
        create_directory(dirs_to_create)

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            train_file_name=config.train_file_name,
            test_file_name=config.test_file_name,
            target_column=params.target_column,
            test_size=float(params.test_size),
            random_state=int(params.random_state),
            imputation_strategy=str(params.imputation_strategy),
            numerical_features=list(params.numerical_features),
            categorical_features=list(params.categorical_features)
        )
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.Model_Trainer_Params
        
        dirs_to_create = [config.root_dir, config.model_dir]
        create_directory(dirs_to_create)
        
        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            model_dir=Path(config.model_dir),
            target_column=params.target_column,
            models_to_train=list(params.models_to_train)
        )
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.Model_Trainer_Params
        
        dirs_to_create = [config.root_dir, config.model_dir]
        create_directory(dirs_to_create)
        
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_dir=Path(config.model_dir),
            metric_file_name=Path(config.metric_file_name),
            target_column=params.target_column
        )