import os
import logging
from datetime import datetime

log_file= f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log"
logs_dir= os.path.join(os.getcwd(), "logs", log_file)
os.makedirs(logs_dir, exist_ok= True)
LOG_FILE_PATH= os.path.join(logs_dir, log_file)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    level= logging.INFO
)
