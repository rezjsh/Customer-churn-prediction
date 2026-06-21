import pandas as pd
from customer_churn.components.data_eda.visualizers import BivariateVisualizer, CorrelationHeatmapVisualizer, DistributionVisualizer
from customer_churn.components.data_eda.orchestrator import ComprehensiveEDAReport
from customer_churn.components.data_eda.analyzers import (
    BivariateAnalyzer, OverviewAnalyzer, MissingDataAnalyzer, OutlierAnalyzer, TargetDistributionAnalyzer, UnivariateAnalyzer
)
from customer_churn.components.data_eda.strategies import TextReportStrategy, JsonReportStrategy
from customer_churn.utils.logging_setup import logger

class DataEDAPipeline:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def run_pipeline(self):
        eda_config = self.config_manager.get_eda_config()
        
        # Load the data generated from Stage 01
        df = pd.read_csv(eda_config.data_path)
        
        # Initialize Orchestrator
        eda_engine = ComprehensiveEDAReport(config=eda_config)
        
        # Register components
        eda_engine.add_analyzer(OverviewAnalyzer()) \
                  .add_analyzer(MissingDataAnalyzer()) \
                  .add_analyzer(OutlierAnalyzer()) \
                  .add_analyzer(TargetDistributionAnalyzer()) \
                  .add_analyzer(UnivariateAnalyzer()) \
                  .add_analyzer(BivariateAnalyzer()) \
                  .add_analyzer(DistributionVisualizer()) \
                  .add_analyzer(CorrelationHeatmapVisualizer()) \
                  .add_analyzer(BivariateVisualizer())
                  
        # Execute analysis
        eda_engine.run_pipeline(df)
        
        # Export using dual strategies
        eda_engine.export_report(TextReportStrategy(), eda_config.text_report_path)
        eda_engine.export_report(JsonReportStrategy(), eda_config.json_report_path)
        
        logger.info("Data EDA Pipeline completed successfully.")