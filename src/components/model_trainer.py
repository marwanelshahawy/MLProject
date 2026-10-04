import sys 
import os 
from dataclasses import dataclass

from src.exception import customException
from src.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfg:
    trained_model_file_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfg()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            logging.info('train and test data split into dependent and independent features')
            THRESHOLD = 0.5

            models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                        "Decision Tree": DecisionTreeClassifier(random_state=42),
                        "Random Forest": RandomForestClassifier(random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                        "AdaBoost": AdaBoostClassifier(random_state=42),
                        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
                        "CatBoost": CatBoostClassifier(verbose=False, random_state=42),
                        "LightGBM": LGBMClassifier(random_state=42),
                        "K-Neighbors": KNeighborsClassifier()
            }

            params = {
                        "Logistic Regression": {
                            "C": [0.01, 0.1, 1.0, 10.0],
                            "penalty": ["l2"],
                            "class_weight": [None, "balanced"],
                            "solver": ["lbfgs", "liblinear"]
                        },
                        
                        "Decision Tree": {
                            "criterion": ["gini", "entropy"],
                            "max_depth": [3, 5, 10, None],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4],
                            "class_weight": [None, "balanced"]
                        },
                        
                        "Random Forest": {
                            "n_estimators": [50, 100, 200],
                            "max_depth": [5, 10, 15, None],
                            "min_samples_split": [2, 5],
                            "min_samples_leaf": [1, 2],
                            "class_weight": [None, "balanced", "balanced_subsample"]
                        },
                        
                        "Gradient Boosting": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [3, 5, 7],
                            "subsample": [0.8, 1.0]
                        },
                        
                        "AdaBoost": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.1, 1.0]
                        },
                        
                        "XGBoost": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [3, 5, 7],
                            "scale_pos_weight": [1, 2.5, 3]  
                        },
                        
                        "CatBoost": {
                            "iterations": [100, 200],
                            "learning_rate": [0.03, 0.05, 0.1],
                            "depth": [4, 6, 8],
                            "auto_class_weights": [None, "Balanced"]
                        },
                        "LightGBM": {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "num_leaves": [31, 50, 100],
                            "class_weight": [None, "balanced"]
                        },
                        
                        "K-Neighbors": {
                            "n_neighbors": [3, 5, 7, 9],
                            "weights": ["uniform", "distance"],
                            "metric": ["euclidean", "manhattan"]
                        }
            }
            logging.info("Model training and hyperparameter tuning initiated.")
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params,threshold=THRESHOLD)
            logging.info("Model training and hyperparameter tuning completed.")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
        
            if best_model_score < .6:
                raise customException("No best model found with score greater than the threshold.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            proba = best_model.predict_proba(X_test)[:, 1]

            predicted = (proba >= THRESHOLD).astype(int)

            recall = recall_score(y_test, predicted)

            logging.info(f"Recall score of the best model on test data: {recall}")
            
            return recall

        except Exception as e:
            raise customException(e,sys)
