import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.ml_model.logger import logging
from src.ml_model.exception import CustomException
from src.ml_model.utils import evaluate_models, save_object
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_scaled_df, X_test_scaled_df, y_train, y_test):
        try:
            logging.info("model training started")

            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGB Classifier": XGBClassifier()
            }

            model_report:dict = evaluate_models(X_train_scaled_df, X_test_scaled_df, y_train, y_test, models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            

            logging.info(f"Best model found on validation set is {best_model}")  


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        
            logging.info("Model training completed successfully")

            return best_model
        
        except Exception as e:
            logging.info("Error occured while training the model")
            raise CustomException(e, sys)