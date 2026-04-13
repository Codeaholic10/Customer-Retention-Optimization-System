import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from logger import logging
from exception import CustomException

class ChurnModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'XGBoost': XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        }
        self.best_model = None
        self.best_model_name = None
        self.cluster_models = {}

    def _evaluate_models(self, X_train, y_train, X_test, y_test):
        try:
            report = {}
            for model_name, model in self.models.items():
                logging.info(f"Training {model_name}...")
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                report[model_name] = {'auc': auc, 'model': model}
                logging.info(f"{model_name} AUC: {auc}")
            return report
        except Exception as e:
            raise CustomException(e, sys)

    def train_global_model(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting global model training.")
            report = self._evaluate_models(X_train, y_train, X_test, y_test)
            
            # Select best model based on AUC
            best_model_name = max(report, key=lambda k: report[k]['auc'])
            self.best_model = report[best_model_name]['model']
            self.best_model_name = best_model_name
            best_auc = report[best_model_name]['auc']
            
            logging.info(f"Best global model: {best_model_name} with AUC: {best_auc}")
            return {
                'best_model_name': best_model_name,
                'auc': best_auc,
                'report': {k: v['auc'] for k, v in report.items()}
            }
        except Exception as e:
            raise CustomException(e, sys)

    def train_cluster_models(self, X_train, y_train, X_test, y_test, train_clusters, test_clusters):
        try:
            logging.info("Starting per-cluster model training.")
            clusters = np.unique(train_clusters)
            cluster_results = {}
            
            for cluster in clusters:
                logging.info(f"Training models for cluster {cluster}...")
                X_train_c = X_train[train_clusters == cluster]
                y_train_c = y_train[train_clusters == cluster]
                X_test_c = X_test[test_clusters == cluster]
                y_test_c = y_test[test_clusters == cluster]
                
                # Minimum samples check
                if len(np.unique(y_train_c)) < 2 or len(np.unique(y_test_c)) < 2:
                    logging.warning(f"Skipping cluster {cluster} due to insufficient class diversity.")
                    continue
                
                report = self._evaluate_models(X_train_c, y_train_c, X_test_c, y_test_c)
                best_model_name = max(report, key=lambda k: report[k]['auc'])
                
                self.cluster_models[cluster] = {
                    'model_name': best_model_name,
                    'model': report[best_model_name]['model'],
                    'auc': report[best_model_name]['auc']
                }
                
                cluster_results[cluster] = {
                    'best_model': best_model_name,
                    'auc': report[best_model_name]['auc']
                }
                logging.info(f"Best model for cluster {cluster}: {best_model_name} (AUC: {report[best_model_name]['auc']})")
                
            return cluster_results
        except Exception as e:
            raise CustomException(e, sys)

    def save_models(self, global_path='artifacts/global_churn_model.joblib', cluster_path_prefix='artifacts/cluster_churn_model'):
        try:
            if self.best_model:
                os.makedirs(os.path.dirname(global_path), exist_ok=True)
                joblib.dump(self.best_model, global_path)
                logging.info(f"Global model saved at {global_path}")
                
            for cluster, model_info in self.cluster_models.items():
                cluster_model_path = f"{cluster_path_prefix}_{cluster}.joblib"
                os.makedirs(os.path.dirname(cluster_model_path), exist_ok=True)
                joblib.dump(model_info['model'], cluster_model_path)
                logging.info(f"Cluster {cluster} model saved at {cluster_model_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def predict_probabilities(self, X, cluster=None):
        try:
            if cluster is not None and cluster in self.cluster_models:
                model = self.cluster_models[cluster]['model']
            elif self.best_model is not None:
                model = self.best_model
            else:
                raise ValueError("No global model or cluster model found to predict.")
                
            return model.predict_proba(X)[:, 1]
        except Exception as e:
            raise CustomException(e, sys)
