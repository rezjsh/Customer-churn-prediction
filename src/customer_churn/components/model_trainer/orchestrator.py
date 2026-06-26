# src/customer_churn/components/model_trainer/orchestrator.py
import os
import joblib
import pandas as pd
from customer_churn.utils.logging_setup import logger
from sklearn.model_selection import GridSearchCV
from customer_churn.entity.config_entity import ModelTrainerConfig
from customer_churn.components.model_trainer.factory import EstimatorFactory

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_all_models(self):
        logger.info("Model Trainer: Loading post-split structural training features matrix.")
        
        if not os.path.exists(self.config.train_data_path):
            raise FileNotFoundError(f"Missing required training matrices dataset at {self.config.train_data_path}")
            
        train_df = pd.read_csv(self.config.train_data_path)
        
        # Isolate targets and features
        X_train = train_df.drop(columns=[self.config.target_column])
        y_train = train_df[self.config.target_column]

        # Explicitly enforce clean type formatting to protect models like XGBoost from boolean/object bugs
        for col in X_train.columns:
            if X_train[col].dtype == 'bool':
                X_train[col] = X_train[col].astype(int)

        # Store explicit training schema metadata
        feature_names = X_train.columns.tolist()
        logger.info(f"Model training initiated with {len(feature_names)} features.")

        os.makedirs(self.config.model_dir, exist_ok=True)

        for model_meta in self.config.models_to_train:
            name = model_meta.get("name")
            should_tune = model_meta.get("tune", False)
            
            try:
                if should_tune:
                    logger.info(f"--- Launching Hyperparameter Tuning Pipeline for: {name} ---")
                    base_estimator = EstimatorFactory.get_model(name, params=None)
                    param_grid = model_meta.get("param_grid", {})
                    
                    search = GridSearchCV(
                        estimator=base_estimator,
                        param_grid=param_grid,
                        cv=5,
                        scoring='f1',
                        n_jobs=-1,
                        verbose=1
                    )
                    search.fit(X_train, y_train)
                    estimator = search.best_estimator_
                    logger.info(f"Optimized tuning parameters recovered for {name}: {search.best_params_}")
                else:
                    logger.info(f"--- Launching Direct Training Pipeline for: {name} ---")
                    params = model_meta.get("params", {})
                    estimator = EstimatorFactory.get_model(name, params)
                    estimator.fit(X_train, y_train)
                
                # BUNDLE MODEL WITH METADATA: Save feature order schema alongside the binary
                model_bundle = {
                    "estimator": estimator,
                    "features_schema": feature_names,
                    "target_column": self.config.target_column
                }
                
                model_filename = f"{name}_model.joblib"
                save_path = os.path.join(self.config.model_dir, model_filename)
                
                joblib.dump(model_bundle, save_path)
                logger.info(f"Model Bundle saved successfully at: {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to complete model training iteration for '{name}': {str(e)}")
                continue