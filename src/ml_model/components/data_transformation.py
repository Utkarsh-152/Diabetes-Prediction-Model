import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from src.ml_model.logger import logging
from src.ml_model.exception import CustomException
from src.ml_model.utils import save_object

@dataclass
class DataTransformationConfig:
    X_train_data_path: str = os.path.join("artifacts", "transformed_data", "X_train.csv")
    X_test_data_path: str = os.path.join("artifacts", "transformed_data", "X_test.csv")
    y_train_data_path: str = os.path.join("artifacts", "transformed_data", "y_train.csv")
    y_test_data_path: str = os.path.join("artifacts", "transformed_data", "y_test.csv")
    preprocessor_path: str = os.path.join("artifacts", "transformed_data", "preprocessor.pkl")



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating StandardScaler object")
            scaler = StandardScaler()
            return scaler
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path):
        try:

            os.makedirs(os.path.dirname(self.data_transformation_config.X_train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.X_test_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.y_train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.y_test_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_path),exist_ok=True)


            # Read the train and test data
            logging.info("Reading raw data")
            data = pd.read_csv(raw_data_path)

            data = data.drop(data[data['Diabetes_012'] == 1].index)
            data = data.rename(columns={'Diabetes_012': 'Diabetes_binary'})
            data['Diabetes_binary'] = data['Diabetes_binary'].map({0: 0, 2: 1})

            logging.info("changing datatype of all features")   
            data["Diabetes_binary"] = data["Diabetes_binary"].astype(int)
            data["HighBP"] = data["HighBP"].astype(int)
            data["HighChol"] = data["HighChol"].astype(int)
            data["CholCheck"] = data["CholCheck"].astype(int)
            data["BMI"] = data["BMI"].astype(int)
            data["Smoker"] = data["Smoker"].astype(int)
            data["Stroke"] = data["Stroke"].astype(int)
            data["HeartDiseaseorAttack"] = data["HeartDiseaseorAttack"].astype(int)
            data["PhysActivity"] = data["PhysActivity"].astype(int)
            data["Fruits"] = data["Fruits"].astype(int) 
            data["Veggies"] = data["Veggies"].astype(int)
            data["HvyAlcoholConsump"] = data["HvyAlcoholConsump"].astype(int)
            data["AnyHealthcare"] = data["AnyHealthcare"].astype(int)
            data["NoDocbcCost"] = data["NoDocbcCost"].astype(int)
            data["GenHlth"] = data["GenHlth"].astype(int)
            data["MentHlth"] = data["MentHlth"].astype(int)
            data["PhysHlth"] = data["PhysHlth"].astype(int)
            data["DiffWalk"] = data["DiffWalk"].astype(int)
            data["Sex"] = data["Sex"].astype(int)
            data["Age"] = data["Age"].astype(int)
            data["Education"] = data["Education"].astype(int)
            data["Income"] = data["Income"].astype(int)

            logging.info("dropping duplicate rows")
            data.drop_duplicates(inplace = True)
            
            X = data.drop('Diabetes_binary', axis=1)
            y = data['Diabetes_binary']

            # Feature selection using SelectKBest with Chi-Square
            selector = SelectKBest(score_func=chi2, k=12)
            X_new = selector.fit_transform(X, y)

            # Get the selected feature indices
            selected_columns = selector.get_support(indices=True)
            important_features = X.columns[selected_columns].tolist()
            logging.info(f"important features: {important_features}")
            X_selected = pd.DataFrame(X_new, columns=important_features)

            nm = NearMiss(version = 1 , n_neighbors = 10)
            X_sm,y_sm= nm.fit_resample(X_selected,y)

            # First split: train vs temp (which will be further split into val and test)
            X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42) 

            logging.info("Applying StandardScaler transformation")
            preprocessor = self.get_data_transformer_object()
            preprocessor.fit(X_train)
            
            X_train_scaled = preprocessor.transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)           
            
            logging.info("Saving transformed data")
            # Convert scaled arrays back to dataframes with column names
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            # Save transformed datasets
            X_train_scaled_df.to_csv(self.data_transformation_config.X_train_data_path, index=False)
            X_test_scaled_df.to_csv(self.data_transformation_config.X_test_data_path, index=False)
            
            pd.DataFrame(y_train).to_csv(self.data_transformation_config.y_train_data_path, index=False)
            pd.DataFrame(y_test).to_csv(self.data_transformation_config.y_test_data_path, index=False)

            # Save the preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor
            )
            
            
            logging.info("Data transformation completed successfully")
            
            return (
                X_train_scaled_df, X_test_scaled_df,
                y_train, y_test, self.data_transformation_config.preprocessor_path
            )


        except Exception as e:
            raise CustomException(e,sys)