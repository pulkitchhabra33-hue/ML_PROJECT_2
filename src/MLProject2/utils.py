import os
import sys
import pandas as pd
from src.MLProject2.logger import logging
from src.MLProject2.exception import CustomException
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

load_dotenv()

host= os.getenv("host")
user= os.getenv("user")
password= os.getenv("password")
db= os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL data started")
    try:
        my_db= pymysql.connect(
            host= host,
            user= user,
            password= password,
            db= db
        )
        
        logging.info(f"Connection Established: {my_db}")
        df= pd.read_sql_query("select * from churn_dataset", my_db)
        print(df.head())

        return df
    
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        logging.info("Saving object started")

        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Object saved successfully")

    except Exception as e:
        raise CustomException(e, sys)
        
def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    try:
        logging.info("Model Evaluation started")
        report= {}

        y_train = np.where(y_train == "Yes", 1, 0)
        y_test = np.where(y_test == "Yes", 1, 0)

        
        for model_name, model in models.items():
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            
            if len(np.unique(y_test_pred)) > 2:
                y_test_pred = (y_test_pred > 0.5).astype(int)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            print(f"{model_name}: {test_model_score}")

        return report
    
    except Exception as e:
        raise CustomException(e, sys)