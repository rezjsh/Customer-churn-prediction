from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_type: str
    file_path: Path
    file_type: str


@dataclass(frozen=True)
class EDAConfig:
    root_dir: Path
    data_path: Path
    json_report_path: Path
    text_report_path: Path
    target_column: str
    correlation_threshold: float
    outlier_method: str


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_file_name: str
    test_file_name: str
    target_column: str
    test_size: float
    random_state: int
    imputation_strategy: str # Added field tracking target mapping
    numerical_features: list
    categorical_features: list


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    model_dir: Path
    target_column: str
    models_to_train: list  

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_dir: Path
    metric_file_name: Path
    target_column: str