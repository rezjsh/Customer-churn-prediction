from customer_churn.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from customer_churn.utils.common import create_directory, read_yaml_file  
from customer_churn.entity.config_entity import DataIngestionConfig, DataValidationConfig, EDAConfig
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