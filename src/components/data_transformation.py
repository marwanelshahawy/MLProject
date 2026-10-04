import pandas as pd 
import numpy as np
from dataclasses import dataclass
import sys

from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import customException
from src.logger import logging
from src.utils import save_object

import os 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_object(self):
        try:
            
            gender_cols = ['gender']

            yes_no_cols = ['Partner','Dependents','PhoneService','PaperlessBilling']

            num_cols = ['tenure','MonthlyCharges','TotalCharges']

            multi_cat_cols = [
                'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                'Contract','PaymentMethod'
                ]
            
            gender_pipeline = Pipeline(steps=[
                ('ordinal',OrdinalEncoder(categories=[['Female', 'Male']]))
            ])

            yes_no_pipeline = Pipeline(steps=[
                ('ordinal',OrdinalEncoder(categories=[['No', 'Yes']]*len(yes_no_cols)))
            ])

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])  

            cat_pipeline = Pipeline(steps=[
                ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f'Numerical columns: {num_cols}')
            logging.info(f'Categorical columns: {multi_cat_cols}')
            logging.info(f'Gender columns: {gender_cols}')
            logging.info(f'Yes/No columns: {yes_no_cols}')
            logging.info('Pipeline Initiated')

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_cols),
                ('cat_pipeline',cat_pipeline,multi_cat_cols),
                ('gender_pipeline',gender_pipeline,gender_cols),
                ('yes_no_pipeline',yes_no_pipeline,yes_no_cols)

            ])
            logging.info("ColumnTransformer preprocessor initialized successfully.")

            return preprocessor


        except Exception as e:
            raise customException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            internet_services = [
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]

            for df in [train_df, test_df]:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'].fillna(df['TotalCharges'].median())

            logging.info("Converted TotalCharges to numeric and handled missing values.")

            for col in internet_services:
                train_df[col] = train_df[col].replace({'No internet service': 'No'})
                test_df[col] = test_df[col].replace({'No internet service': 'No'})

            logging.info("Replaced 'No internet service' with 'No' in internet service columns.")

            for df in [train_df, test_df]:
                df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
            
            logging.info("Replaced 'No phone service' with 'No' in MultipleLines column.")

            if 'customerID' in train_df.columns:
                train_df.drop(columns=['customerID'], inplace=True)
                test_df.drop(columns=['customerID'], inplace=True)

            logging.info("Dropped 'customerID' column from train and test data.")

            preprocessor_obj = self.get_transformation_object()

            target_column_name = 'Churn'
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name].map({'No': 0, 'Yes': 1})

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name].map({'No': 0, 'Yes': 1})

            logging.info("Separated input features and target feature from train and test data.")
            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Preprocessing completed successfully.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),exist_ok=True)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise customException(e,sys)

