import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.ml_model.logger import logging
from src.ml_model.exception import CustomException
from src.ml_model.utils import get_data_from_mysql

logging.info("Data Ingestion libraries imported successfully")

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        
        try:
            data = get_data_from_mysql()
            logging.info("Reading completed from mysql database")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,header=True,index=False)
            logging.info("Data saved successfully to csv")

            return(
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            logging.info("Error occured while reading dataset")
            raise CustomException(e, sys)