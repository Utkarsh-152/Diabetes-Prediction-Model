import os
import sys
import pymysql
import pandas as pd
from dotenv import load_dotenv  
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from src.ml_model.logger import logging
from src.ml_model.exception import CustomException
import pickle
from sklearn.metrics import accuracy_score, recall_score
load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def get_data_from_mysql():
    """
    Fetches data from MySQL database using environment variables for connection.
    Returns a pandas DataFrame containing the data.
    """
    try:
        logging.info("Data read from mysql started")
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        logging.info("Connection established for data ingestion")
        
        # Create SQLAlchemy engine for pandas
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{db}')
        
        # Read data into pandas DataFrame
        # Note: You'll need to specify your table name here
        query = "SELECT * FROM diabetes"  # Replace with your actual table name
        df = pd.read_sql(query, engine)
        
        logging.info(f"Successfully read {len(df)} rows from database")
        return df
        
    except SQLAlchemyError as e:
        logging.error(f"Database error occurred: {str(e)}")
        raise CustomException(e, sys)
    except Exception as e:
        logging.error(f"Error occurred while reading from database: {str(e)}")
        raise CustomException(e, sys)
    finally:
        if 'connection' in locals():
            connection.close()
            logging.info("Database connection closed")

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, X_test, y_train, y_test, models):
    try:

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]    
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            
            model_test_accuracy_score = accuracy_score(y_test, y_test_pred)
            model_test_recall_score = recall_score(y_test_pred, y_test)


            report[list(models.keys())[i]] = model_test_accuracy_score
        return report
    
    except Exception as e:
        logging.info("Error occured while evaluating models")
        raise CustomException(e, sys)