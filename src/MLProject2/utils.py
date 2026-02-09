import os
import sys
import pandas as pd
from src.MLProject2.logger import logging
from src.MLProject2.exception import CustomException
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np

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





