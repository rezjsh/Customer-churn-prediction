from customer_churn.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from customer_churn.utils.common import read_yaml_file  
from customer_churn.entity.config_entity import DataIngestionConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml_file(Path(config_filepath))
        self.params = read_yaml_file(Path(params_filepath))

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_type=config.source_type,
            file_path=Path(config.file_path),
            file_type=config.file_type
        )