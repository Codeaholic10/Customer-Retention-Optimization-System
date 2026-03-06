import logging
import os
import sys
from datetime import datetime

# ------------------------------------------------------------------ #
#  File handler – timestamped log file inside logs/
# ------------------------------------------------------------------ #
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

LOG_FORMAT = "[ %(asctime)s ] %(filename)s:%(lineno)d %(name)s - %(levelname)s - %(message)s"

# ------------------------------------------------------------------ #
#  Named application logger
# ------------------------------------------------------------------ #
logger = logging.getLogger("CustomerChurnLogger")
logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if the module is reloaded
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    # Console (stdout) handler – so logs are visible in the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

# Prevent messages from propagating to the root logger
logger.propagate = False
