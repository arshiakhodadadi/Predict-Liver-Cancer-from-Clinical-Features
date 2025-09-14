import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, recall_score, precision_score, r2_score, mean_squared_error,f1_score,confusion_matrix
import joblib
import logging

logging.basicConfig(level=logging.INFO)

class XGboostTrainer:
    def __init__(self, params=None, is_classification=True, grid_search=False, num_classes=None, eval_metric=None):
        self.is_classification = is_classification
        self.grid_search = grid_search
        self.eval_metric = eval_metric if eval_metric else ('auc' if is_classification else 'rmse')
        if is_classification:
            if num_classes is not None and num_classes < 2:
                logging.error("Number of classes must be at least 2 for classification.")
                raise ValueError("Number of classes must be at least 2 for classification.")
            self.model = xgb.XGBClassifier(
                objective='binary:logistic' if num_classes == 2 else 'multi:softprob',
                random_state=42,
                **(params or {})
            )
        else:
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                **(params or {})
            )
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_name = None

    def train(self, X, y, feature_names=None):
        try:
            if X is None or y is None:
                logging.error("Training data is not provided.")
                raise ValueError("Training data is not provided.")
            if len(X) != len(y):
                logging.error("X and y must have the same length.")
                raise ValueError("X and y must have the same length.")
            self.feature_name = feature_names if feature_names is not None else (X.columns if isinstance(X, pd.DataFrame) else None)
            X = pd.DataFrame(X, columns=self.feature_name) if self.feature_name else X
            if self.is_classification and self.model.objective == 'multi:softprob':
                num_classes = len(np.unique(y))
                self.model.set_params(num_classes=num_classes)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logging.info("Training model...")
            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                early_stopping_rounds=50,
                verbose=False,
                eval_metric=self.eval_metric
            )
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def optimize(self):
        try:
            if self.X_train is None or self.y_train is None:
                logging.error("Training data is not available. Please train the model first.")
                raise ValueError("Training data is not available.")
            param_dist = {
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 10]
            }
            scoring = 'roc_auc' if self.is_classification else 'neg_mean_squared_error'
            if self.grid_search:
                search = GridSearchCV(
                    self.model,
                    param_grid=param_dist,
                    cv=5,
                    scoring=scoring,
                    verbose=1,
                    n_jobs=-1
                )
            else:
                search = RandomizedSearchCV(
                    self.model,
                    param_distributions=param_dist,
                    n_iter=50,
                    cv=5,
                    scoring=scoring,
                    verbose=1,
                    n_jobs=-1,
                    random_state=42
                )
            search.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                early_stopping_rounds=30,
                verbose=False,
                eval_metric=self.eval_metric
            )
            self.model = search.best_estimator_
            logging.info(f"Best Parameters is: {search.best_params_}")
            logging.info(f"Best CV score is : {search.best_score_}")
        except Exception as e:
            logging.error(f"Error during optimization: {str(e)}")
            raise

    def evaluate(self):
        try:
            if not hasattr(self.model, 'feature_importances_'):
                logging.error("Model is not trained yet.")
                raise ValueError("Model is not trained yet.")
            if self.X_test is None or self.y_test is None:
                logging.error("Test data is not available. Please train the model first.")
                raise ValueError("Test data is not available.")
            y_pred = self.model.predict(self.X_test)
            metrics = {}
            if self.is_classification:
                y_pred_proba = self.model.predict_proba(self.X_test)
                if self.model.objective == 'binary:logistic':
                    auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
                metrics = {
                    'AUC': auc,
                    'Recall': recall_score(self.y_test, y_pred, average='weighted'),
                    'Precision': precision_score(self.y_test, y_pred, average='weighted'),
                    'F1 Score': f1_score(self.y_test, y_pred, average='weighted'),
                    'Confusion Matrix': confusion_matrix(self.y_test, y_pred).tolist()

                }
            else:
                metrics = {
                    'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                    'R2': r2_score(self.y_test, y_pred)
                }
            logging.info(f"Evaluation Metrics: {metrics}")
            return metrics
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

    def feature_importance(self, top_n=10):
        try:
            if not hasattr(self.model, 'feature_importances_'):
                logging.error("Model is not trained yet.")
                raise ValueError("Model is not trained yet.")
            if self.feature_name is None:
                logging.warning("Feature names are not provided. Using default indices.")
                self.feature_name = [f"feature_{i}" for i in range(self.X_train.shape[1])]
            importance = self.model.feature_importances_
            logging.info(f"Feature importances: {importance}")
            return sorted(zip(self.feature_name, importance), key=lambda x: x[1], reverse=True)[:top_n]
        except Exception as e:
            logging.error(f"Error during feature importance calculation: {str(e)}")
            raise

    def save(self, path="Model.joblib"):
        try:
            joblib.dump(self.model, path)
            logging.info(f"Model saved to {path}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise

    def load(self, path="Model.joblib"):
        try:
            if not os.path.exists(path):
                logging.error(f"Model file {path} does not exist")
                raise FileNotFoundError(f"Model file {path} does not exist")
            self.model = joblib.load(path)
            logging.info(f"Model loaded from {path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
