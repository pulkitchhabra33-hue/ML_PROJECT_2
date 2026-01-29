import sys
from src.MLProject2.logger import logging
from src.MLProject2.exception import CustomException

if __name__== "__main__":
    logging.info("Application has started")
    
    try:
        pass
    
    except Exception as e:
        logging.error("Appliaction has failed", exc_info= True)
        raise CustomException(e,sys)