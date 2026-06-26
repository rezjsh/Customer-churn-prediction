import pandas as pd
from customer_churn.components.data_eda.visualizers import BivariateVisualizer, CorrelationHeatmapVisualizer, DistributionVisualizer
from customer_churn.components.data_eda.orchestrator import ComprehensiveEDAReport
from customer_churn.components.data_eda.analyzers import (
    BivariateAnalyzer, CardinalityAnalyzer, MulticollinearityAnalyzer, MutualInformationAnalyzer, OverviewAnalyzer, MissingDataAnalyzer, OutlierAnalyzer, TargetDistributionAnalyzer, UnivariateAnalyzer
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

        # 1. Clean TotalCharges (spaces to NaN, then float)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(r'^\s*$', None, regex=True))

        # 2. Drop unique identifiers to keep analyzers focused
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        
        # Initialize Orchestrator
        eda_engine = ComprehensiveEDAReport(config=eda_config)
        
        # Register components
        eda_engine.add_analyzer(OverviewAnalyzer()) \
                  .add_analyzer(MissingDataAnalyzer()) \
                  .add_analyzer(OutlierAnalyzer()) \
                  .add_analyzer(TargetDistributionAnalyzer()) \
                  .add_analyzer(UnivariateAnalyzer()) \
                  .add_analyzer(BivariateAnalyzer()) \
                  .add_analyzer(CardinalityAnalyzer()) \
                  .add_analyzer(MulticollinearityAnalyzer()) \
                  .add_analyzer(MutualInformationAnalyzer()) \
                  .add_analyzer(DistributionVisualizer()) \
                  .add_analyzer(CorrelationHeatmapVisualizer()) \
                  .add_analyzer(BivariateVisualizer())
                  
        # Execute analysis
        eda_engine.run_pipeline(df)
        
        # Export using dual strategies
        eda_engine.export_report(TextReportStrategy(), eda_config.text_report_path)
        eda_engine.export_report(JsonReportStrategy(), eda_config.json_report_path)
        
        logger.info("Data EDA Pipeline completed successfully.")