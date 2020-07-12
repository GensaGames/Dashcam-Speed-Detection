import logging
import os

from app import Settings


def build_logger(include_file=False):

    formatter = logging.Formatter(
        '%(asctime)-15s %(message)s')

    log = logging.getLogger(
        os.path.basename(__file__))
    log.setLevel(logging.DEBUG)

    path_to = Settings.BUILD
    if not os.path.exists(path_to):
        os.makedirs(path_to)

    if include_file:
        handler = logging.FileHandler(
            filename=path_to + Settings.NAME_LOGS)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        log.addHandler(handler)

    handler1 = logging.StreamHandler()
    handler1.setLevel(logging.DEBUG)
    handler1.setFormatter(formatter)
    log.addHandler(handler1)
    return log


logger = build_logger()


def get_logger():
    return logger
