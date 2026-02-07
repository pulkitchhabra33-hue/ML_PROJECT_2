import sys
from src.MLProject2.logger import logging
from src.MLProject2.exception import CustomException
from src.MLProject2.components.data_ingestion import DataIngestion

if __name__== "__main__":
    logging.info("Application has started")
    
    try:
        data_ingestion= DataIngestion()
        data_ingestion.initiate_data_ingestion()
    
    except Exception as e:
        logging.error("Appliaction has failed", exc_info= True)
        raise CustomException(e,sys)