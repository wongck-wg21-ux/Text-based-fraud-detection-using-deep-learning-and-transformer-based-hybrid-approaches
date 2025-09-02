import logging

def get_logger():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s — %(message)s")
    return logging.getLogger("fraud_detector")