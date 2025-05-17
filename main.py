import sys
from src.ml_model.exception import CustomException
from src.ml_model.logger import logging
from src.ml_model.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.ml_model.components.data_transformation import DataTransformation, DataTransformationConfig
from src.ml_model.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.ml_model.components.model_monitering import ModelPerformanceMonitoring

if __name__ == "__main__":
    logging.info("Application has started")

    try:
        #Data Ingestion
        data_ingestion = DataIngestion()
        data_ingestion_config = DataIngestionConfig()
        raw_data = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")

        #Data Tranformation
        data_transformation = DataTransformation()
        data_transformation_config = DataTransformationConfig()
        X_train_scaled_df, X_test_scaled_df, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(raw_data)
        logging.info("Data transformation completed successfully")

        #model training
        data_modeling = ModelTrainer()
        data_modeling_config = ModelTrainerConfig()
        best_model = data_modeling.initiate_model_trainer(X_train_scaled_df=X_train_scaled_df,X_test_scaled_df=X_test_scaled_df,y_train=y_train,y_test=y_test)

        #model eval report generation
        monitor = ModelPerformanceMonitoring()
        metrics = monitor.generate_report(best_model, X_test_scaled_df, y_test)
        logging.info("Model performance monitoring completed successfully") 
    except Exception as e:
        logging.info("Error occured while running the application")
        raise CustomException(e, sys)
