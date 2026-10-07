import sys
import os
from src.exception import logging
from src.exception import customException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            # Data Ingestion
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed.")

            # Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
            logging.info("Data transformation completed.")

            # Model Training
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed. Best model recall score: {model_score}")

            return model_score

        except Exception as e:
            raise customException(e, sys)

