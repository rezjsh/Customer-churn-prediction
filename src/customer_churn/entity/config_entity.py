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