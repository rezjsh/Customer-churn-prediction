# src/customer_churn/components/model_trainer/factory.py
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

class EstimatorFactory:
    @staticmethod
    def get_model(model_name: str, params: dict = None):
        name = model_name.lower()
        parameters = params if params is not None else {}
        logging.info(f"Model Factory: Initializing '{name}' with operational configurations: {parameters}")
        
        if name == "logistic_regression":
            return LogisticRegression(**parameters)
            
        elif name == "random_forest":
            return RandomForestClassifier(**parameters)
            
        elif name == "xgboost":
            # For customer churn imbalance, use scale_pos_weight if class_weight is passed
            xgb_params = parameters.copy()
            if 'class_weight' in xgb_params:
                xgb_params.pop('class_weight')  # XGBoost doesn't accept class_weight directly
            if 'eval_metric' not in xgb_params:
                xgb_params['eval_metric'] = 'logloss'
            return XGBClassifier(**xgb_params)
            
        elif name == "svm":
            svc_params = {k: v for k, v in parameters.items() if k != 'probability'}
            if 'class_weight' not in svc_params:
                svc_params['class_weight'] = 'balanced'
            base_svc = SVC(**svc_params)
            return CalibratedClassifierCV(estimator=base_svc, ensemble=False)
            
        else:
            raise ValueError(f"Model architecture type '{model_name}' is currently unsupported.")