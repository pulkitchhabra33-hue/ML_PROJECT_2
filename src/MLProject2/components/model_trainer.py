# Importing necessary libraries

import os
import sys

from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score 
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.MLProject2.exception import CustomException
from src.MLProject2.logger import logging
from src.MLProject2.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
    
    def eval_metrics(self, actual, pred):
        acc= accuracy_score(actual, pred)
        return acc
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test= (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models= {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier()
            }

            params = {
                
                "Logistic Regression": {
                "penalty": ["l2"],
                "C": [0.01, 0.1, 1, 10]
                },

                "Random Forest": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10]
                },

                "Gradient Boosting": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1]
                },

                "XGBoost": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1]
                }
            }


            model_report: dict= evaluate_models(X_train, X_test, y_train, y_test, models, params)
            
            ## To get the best model score from the dictionary
            best_model_score= max(sorted(model_report.values()))

            ## To get the best model name from the dictionary
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]

            print(f"Best Model Found: {best_model_name} with accuracy score: {best_model_score}")

            actual_model= best_model_name
            best_params= params[actual_model]

            mlflow.set_tracking_uri("https://dagshub.com/pulkitchhabra33-hue/ML_PROJECT_2.mlflow/")
            with mlflow.start_run():

                best_model.fit(X_train, y_train)
                predicted= best_model.predict(X_test)

                acc= accuracy_score(y_test, predicted)

                mlflow.log_param("model", actual_model)
                mlflow.log_metric("accuracy", acc)

                mlflow.sklearn.log_model(best_model, "model")


            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found. Accuracy: {best_model_score}")
                
            ## Saving the best model to the file
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            return best_model_score

        except Exception as e:
            logging.error("Error occurred during model training", exc_info= True)
            raise CustomException(e, sys)