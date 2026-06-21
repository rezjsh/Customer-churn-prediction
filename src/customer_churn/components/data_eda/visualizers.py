import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from customer_churn.components.data_eda.analyzers import AnalysisComponent
from customer_churn.utils.logging_setup import logger

class DistributionVisualizer(AnalysisComponent):
    """Generates distribution plots for numerical features."""
    def analyze(self, df: pd.DataFrame, config) -> dict:
        plot_dir = os.path.join(config.root_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(os.path.join(plot_dir, f"{col}_dist.png"))
            plt.close()
        return {"visualizations": "Distributions saved to plots folder."}

class CorrelationHeatmapVisualizer(AnalysisComponent):
    """Generates a heatmap for numerical correlations."""
    def analyze(self, df: pd.DataFrame, config) -> dict:
        plot_dir = os.path.join(config.root_dir, "plots")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
        plt.close()
        return {"visualizations": "Correlation heatmap saved."}
    


class BivariateVisualizer(AnalysisComponent):
    """Generates visual comparisons of features against the target variable."""
    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Generating Bivariate Visualizations...")
        plot_dir = os.path.join(config.root_dir, "plots", "bivariate")
        os.makedirs(plot_dir, exist_ok=True)
        target = config.target_column
        
        if target not in df.columns:
            return {"visualizations": "Target missing, skipping bivariate plots."}

        # 1. Numerical vs Target: Boxplots
        numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col != target]
        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df, x=target, y=col, palette="Set2")
            plt.title(f"{col} vs {target}")
            plt.savefig(os.path.join(plot_dir, f"boxplot_{col}_vs_{target}.png"), bbox_inches='tight')
            plt.close()

        # 2. Categorical vs Target: Stacked Bar Charts
        # Picking a subset of important categorical columns to avoid plot bloat
        cat_cols = ['InternetService', 'Contract', 'PaymentMethod', 'gender']
        cat_cols = [col for col in cat_cols if col in df.columns]
        
        for col in cat_cols:
            # Calculate percentages for stacking
            cross_tab = pd.crosstab(df[col], df[target], normalize='index')
            cross_tab.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='viridis')
            plt.title(f"Churn Proportion across {col}")
            plt.ylabel("Proportion")
            plt.xticks(rotation=45)
            plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig(os.path.join(plot_dir, f"stacked_bar_{col}_vs_{target}.png"), bbox_inches='tight')
            plt.close()

        return {"visualizations": "Bivariate plots successfully saved."}