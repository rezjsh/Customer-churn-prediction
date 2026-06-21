import pandas as pd
import logging
from typing import List
from customer_churn.entity.config_entity import EDAConfig
from customer_churn.components.data_eda.analyzers import AnalysisComponent
from customer_churn.components.data_eda.strategies import ReportStrategy

class ComprehensiveEDAReport:
    def __init__(self, config: EDAConfig):
        self.config = config
        self._analyzers: List[AnalysisComponent] = []
        self._raw_results = {}

    def add_analyzer(self, analyzer: AnalysisComponent) -> 'ComprehensiveEDAReport':
        self._analyzers.append(analyzer)
        return self

    def run_pipeline(self, df: pd.DataFrame) -> dict:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
            
        logging.info(f"Starting EDA processing pipeline across {len(self._analyzers)} modules...")
        
        for analyzer in self._analyzers:
            analysis_data = analyzer.analyze(df, self.config)
            self._raw_results.update(analysis_data)
            
        return self._raw_results

    def export_report(self, strategy: ReportStrategy, filepath: str):
        """Applies the strategy and saves the output."""
        strategy.generate(self._raw_results, filepath)
        logging.info(f"Report successfully generated at: {filepath}")