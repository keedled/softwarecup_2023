# logging_module.py

import logging

def setup_logger():
    logger = logging.getLogger("MyApp")
    handler = logging.FileHandler('app.log')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
