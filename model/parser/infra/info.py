import logging
import datetime
import time
import pytz
import sys


def cst_time():
    return datetime.datetime.fromtimestamp(round(time.time()), pytz.timezone('Asia/Shanghai'))


def get_logger(name, filename, level=logging.INFO, formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(filename)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger

