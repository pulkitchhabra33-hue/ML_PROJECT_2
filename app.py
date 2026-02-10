import sys
from src.MLProject2.logger import logging
from src.MLProject2.exception import CustomException
from src.MLProject2.components.data_ingestion import DataIngestion
from src.MLProject2.components.data_transformation import DataTransformation
from src.MLProject2.components.model_trainer import ModelTrainer

if __name__== "__main__":
    logging.info("Application has started")
    
    try:
        #data ingestion
        data_ingestion= DataIngestion()
        train_data_path, test_data_path= data_ingestion.initiate_data_ingestion()

        #data transsformation
        data_transformation= DataTransformation()
        train_array, test_array, _= data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        #model trainer
        model_trainer= ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_array, test_array))
    
    except Exception as e:
        logging.error("Appliaction has failed", exc_info= True)
        raise CustomException(e,sys)