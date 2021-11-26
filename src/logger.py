import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
stdout_logger = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_logger)
