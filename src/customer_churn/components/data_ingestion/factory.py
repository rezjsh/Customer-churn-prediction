from customer_churn.entity.config_entity import DataIngestionConfig
from customer_churn.components.data_ingestion.interface import IngestionStrategy
from customer_churn.components.data_ingestion.strategies import FileIngestionStrategy, ApiIngestionStrategy


class IngestionStrategyFactory:
    """Factory responsible strictly for instantiating the correct strategy."""
    @staticmethod
    def get_strategy(config: DataIngestionConfig) -> IngestionStrategy:
        source_type = config.source_type.lower()
        
        if source_type == 'file':
            return FileIngestionStrategy(file_path=str(config.file_path), file_type=config.file_type)
        elif source_type == 'api':
            return ApiIngestionStrategy(endpoint=str(config.file_path)) # example mapping
        else:
            raise ValueError(f"Unknown or unimplemented source type: {source_type}")