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
#  Best-effort UTF-8 reconfiguration for the Windows console so that
#  Unicode characters in log messages don't raise UnicodeEncodeError.
# ------------------------------------------------------------------ #
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

# ------------------------------------------------------------------ #
#  Named application logger
# ------------------------------------------------------------------ #
logger = logging.getLogger("CustomerChurnLogger")
logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers if the module is reloaded
if not logger.handlers:
    # File handler – always UTF-8
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
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
