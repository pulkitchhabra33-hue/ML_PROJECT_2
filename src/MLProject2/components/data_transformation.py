import os
import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
import pickle

from src.MLProject2.exception import CustomException
from src.MLProject2.logger import logging
from src.MLProject2.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

            categorical_columns = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]
            
            
            
            num_Pipeline= Pipeline(steps=[
                ('imputer', SimpleImputer(strategy= 'median')),
                ('scaler', StandardScaler())
            ])
            
            cat_Pipeline= Pipeline(steps=[
                ('imputer', SimpleImputer(strategy= 'most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown= 'ignore')),
                ('scaler', StandardScaler(with_mean= False))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocesor= ColumnTransformer(
                [
                ('num_pipeline', num_Pipeline, numerical_columns),
                ('cat_pipeline', cat_Pipeline, categorical_columns)
            ]
            )
            
            return preprocesor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Reading train and test data")

            preprocessor_obj= self.get_data_transformer_object()
                
            target_column_name= 'Churn'
            numerical_columns= ['tenure', 'MonthlyCharges', 'TotalCharges']

            ## divide the train dataset to independent and dependent feature

            input_feature_train_df= train_df.drop(columns= [target_column_name], axis= 1)
            target_feature_train_df= train_df[target_column_name]

            ## divide the test dataset to independent and dependent feature

            input_feature_test_df= test_df.drop(columns= [target_column_name], axis= 1)
            target_feature_test_df= test_df[target_column_name]

            ## transform the data usinf preprocessor object

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr= preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessor_obj.transform(input_feature_test_df)

            train_arr= np.hstack(
                (input_feature_train_arr, target_feature_train_df.values.reshape(-1, 1))
            )

            test_arr= np.hstack(
                (input_feature_test_arr, target_feature_test_df.values.reshape(-1, 1))
            )

            logging.info("Saved preprocessing object")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
                


