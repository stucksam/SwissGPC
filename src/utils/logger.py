import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Define log file path
LOG_FILE = os.path.join(LOG_DIR, datetime.now().strftime('%Y_%m_%d_%H_%M.log'))

# Configure logger
logging.basicConfig(
    level=logging.DEBUG,  # Set logging level (change to INFO or WARNING in production)
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)


# Get a named logger
def get_logger(name):
    return logging.getLogger(name)
