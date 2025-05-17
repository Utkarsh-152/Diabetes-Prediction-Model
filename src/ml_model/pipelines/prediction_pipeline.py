import sys
import os
import pandas as pd
import numpy as np
from src.ml_model.exception import CustomException
from src.ml_model.logger import logging
from src.ml_model.utils import load_object 

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "transformed_data", "preprocessor.pkl")

    def predict(self, data):
        try:
            # Load the model and preprocessor
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Scale the features using the saved preprocessor
            scaled_data = preprocessor.transform(data)
            scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

            # Make prediction
            prediction = model.predict(scaled_data)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        HighBP: int, 
        HighChol: int, 
        BMI: int, 
        Stroke: int, 
        HeartDiseaseorAttack: int, 
        HvyAlcoholConsump: int, 
        GenHlth: int, 
        MentHlth: int, 
        PhysHlth: int, 
        DiffWalk: int, 
        Age: int, 
        Income: int
    ):
        self.HighBP = HighBP
        self.HighChol = HighChol
        self.BMI = BMI
        self.Stroke = Stroke
        self.HeartDiseaseorAttack = HeartDiseaseorAttack
        self.HvyAlcoholConsump = HvyAlcoholConsump
        self.GenHlth = GenHlth
        self.MentHlth = MentHlth
        self.PhysHlth = PhysHlth
        self.DiffWalk = DiffWalk
        self.Age = Age
        self.Income = Income

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "HighBP": [self.HighBP],
                "HighChol": [self.HighChol],
                "BMI": [self.BMI],
                "Stroke": [self.Stroke],
                "HeartDiseaseorAttack": [self.HeartDiseaseorAttack],
                "HvyAlcoholConsump": [self.HvyAlcoholConsump],
                "GenHlth": [self.GenHlth],
                "MentHlth": [self.MentHlth],
                "PhysHlth": [self.PhysHlth],
                "DiffWalk": [self.DiffWalk],
                "Age": [self.Age],
                "Income": [self.Income]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)