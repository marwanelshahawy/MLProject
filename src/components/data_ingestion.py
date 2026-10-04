import os 
import sys 
from src.exception import customException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join ('artifacts','train.csv')
    test_data_path:str = os.path.join ('artifacts','test.csv')
    raw_data_path:str = os.path.join ('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.Ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv('data\\raw\\Telco-Customer-Churn.csv')
            logging.info('Read the dataset as a DataFrame')

            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.Ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test split initiated')
            target_column = df['Churn']
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42,stratify=target_column)
            train_set.to_csv(self.Ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.Ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is completed')

            return(
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
            )

        except Exception as e:
            raise customException(e,sys)

