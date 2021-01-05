import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def initialise_logger(log_level, name):
    logger = logging.getLogger(name)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_fmt)
    return logger
