import logging
import sys

dateformat = "%m/%d/%Y  %H:%M:%S"
logformat = "%(asctime)s [%(levelname)-5.5s] [%(name)s] %(message)s"
consoleforma = "[%(levelname)-5.5s] [%(name)s] %(message)s"


def build_logger():
    
    logFormatter = logging.Formatter(logformat,datefmt = dateformat)
    consoleFormatter = logging.Formatter("[%(levelname)-5.5s] [%(name)s] %(message)s")
    log = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format(".","logs"))
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(consoleFormatter)
    consoleHandler.setLevel(logging.INFO)
    log.addHandler(consoleHandler)

    log.setLevel(logging.DEBUG)
    return log
