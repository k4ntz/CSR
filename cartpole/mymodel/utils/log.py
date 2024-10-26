import os
import sys
import logging
from time import strftime

# PATH = os.path.abspath('.') + '/logs/'
FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'

class Log(object):
    def __init__(self, logfile):
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
        filename = logfile.split('.')
        self.log_filename = '{0}_{1}.{2}'.format(filename[0], strftime("%Y-%m-%d"), filename[1])

        self.logger.addHandler(self.get_file_handler(self.log_filename))
        self.logger.addHandler(self.get_console_handler())
        self.logger.setLevel(logging.DEBUG)

    def get_file_handler(self, filename):
        filehandler = logging.FileHandler(filename, encoding="utf-8")
        filehandler.setFormatter(self.formatter)
        return filehandler

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler