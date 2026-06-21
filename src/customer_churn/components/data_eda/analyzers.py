import pandas as pd
import numpy as np
from customer_churn.components.data_eda.interface import AnalysisComponent
from customer_churn.utils.logging_setup import logger



class OverviewAnalyzer(AnalysisComponent):
    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Dataset Overview")
        return {
            "num_rows": int(df.shape[0]),
            "num_cols": int(df.shape[1]),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

class MissingDataAnalyzer(AnalysisComponent):
    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Missing Data Profile")
        null_counts = df.isnull().sum()
        profile = {
            col: {"count": int(count), "percentage": round(float((count / len(df)) * 100), 2)}
            for col, count in null_counts.items() if count > 0
        }
        return {"missing_data": profile}

class OutlierAnalyzer(AnalysisComponent):
    """Identifies outliers using the IQR method based on config thresholds."""
    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info(f"Analyzing: Outliers using {config.outlier_method.upper()} method")
        numeric_df = df.select_dtypes(include=[np.number])
        outliers_profile = {}

        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = int(((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum())
            if outlier_count > 0:
                outliers_profile[col] = {"outlier_count": outlier_count, "percentage": round((outlier_count/len(df))*100, 2)}
                
        return {"outliers": outliers_profile}

class TargetDistributionAnalyzer(AnalysisComponent):
    """Analyzes class imbalance for the configured target variable."""
    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info(f"Analyzing: Target Distribution for '{config.target_column}'")
        if config.target_column not in df.columns:
            return {"target_distribution": "Target column not found in dataset."}
            
        dist = df[config.target_column].value_counts(normalize=True).round(4) * 100
        return {"target_distribution": dist.to_dict()}


class CardinalityAnalyzer(AnalysisComponent):
    """Identifies the number of unique values in categorical columns."""
    def analyze(self, df: pd.DataFrame, config) -> dict:
        cat_df = df.select_dtypes(include=['object'])
        cardinality = {col: int(cat_df[col].nunique()) for col in cat_df.columns}
        return {"cardinality": cardinality}
    

class UnivariateAnalyzer(AnalysisComponent):
    """Analyzes single variables for statistical shape and central tendencies."""
    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info("Analyzing: Univariate Statistics")
        univariate_stats = {}
        
        # Numerical Features: Check for skewness and kurtosis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            skewness = df[numeric_cols].skew().round(2).to_dict()
            kurtosis = df[numeric_cols].kurtosis().round(2).to_dict()
            univariate_stats["numerical_shape"] = {
                col: {"skewness": skewness[col], "kurtosis": kurtosis[col]}
                for col in numeric_cols
            }

        # Categorical Features: Get the top prevailing category (Mode)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if not cat_cols.empty:
            modes = df[cat_cols].mode().iloc[0].to_dict()
            univariate_stats["categorical_mode"] = modes

        return {"univariate_analysis": univariate_stats}


class BivariateAnalyzer(AnalysisComponent):
    """Analyzes relationships between features and the target variable."""
    def analyze(self, df: pd.DataFrame, config) -> dict:
        logger.info(f"Analyzing: Bivariate Relationships against '{config.target_column}'")
        bivariate_stats = {}
        target = config.target_column

        if target not in df.columns:
            return {"bivariate_analysis": "Target column missing."}

        # 1. Numerical vs. Target (Difference in means)
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target]
        if numeric_cols:
            # Group by target and calculate the mean for each numerical feature
            mean_by_target = df.groupby(target)[numeric_cols].mean().round(2).to_dict()
            bivariate_stats["numerical_vs_target_mean"] = mean_by_target

        # 2. Categorical vs. Target (Cross-tabulation / Churn Rate per category)
        cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col != target]
        if cat_cols:
            cat_target_interactions = {}
            for col in cat_cols:
                # Creates a percentage breakdown of churn for each category level
                crosstab = pd.crosstab(df[col], df[target], normalize='index').round(4) * 100
                cat_target_interactions[col] = crosstab.to_dict()
            bivariate_stats["categorical_vs_target_crosstab"] = cat_target_interactions

        return {"bivariate_analysis": bivariate_stats}