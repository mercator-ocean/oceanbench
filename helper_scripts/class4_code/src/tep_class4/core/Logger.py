import logging
from colorlog import ColoredFormatter
import datetime
import hashlib
## Log wrapper
# Creation C.REGNIER Juillet 2014
# Creation C.REGNIER October 2020 :  add color logger

class Logger(object):

    """ Class Logger : define level of log with a number """

    def __init__(self, name, stream='', dirlog='') :
        self.name=name
        self.dirlog = dirlog
        current_datetime = datetime.datetime.now()
        log_name = current_datetime.strftime(f"Log_{name}_%Y-%m-%d_%H-%M")
        # Calculate the MD5 hash of the log name
        hash = hashlib.md5(log_name.encode()).hexdigest()
        self.logname = f'{self.dirlog}{log_name}_{hash}.txt'
        LOGFORMAT = " %(log_color)s %(asctime)s  %(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
        self.formatter = ColoredFormatter(LOGFORMAT,
                         datefmt=None,
                         reset=True,
                         log_colors={
                                     'DEBUG':    'cyan',
                                     'INFO':     'green',
                                     'WARNING':  'yellow',
                                     'ERROR':    'red',
                                     'CRITICAL': 'red,bg_white',
                                      },
                )
        self.stream = stream
        if stream == 'logfile':
            print(f'Add logfile {self.logname}')
            self.stream = logging.FileHandler(self.logname)
        else:
            self.stream = logging.StreamHandler()

    def run(self,level):
        log = logging.getLogger(self.name)
        ## Clean logger
        if (log.hasHandlers()):
           log.handlers.clear()
        ## Level of logging output
        if level == 0 :
            log.setLevel(logging.NOTSET)
        elif level == 1 :
            log.setLevel(logging.DEBUG)
        elif level == 2 :
            log.setLevel(logging.INFO)
        elif level == 3 :
            log.setLevel(logging.WARNING)
        elif level == 4 :
            log.setLevel(logging.ERROR)
        elif level == 5 :
            log.setLevel(logging.CRITICAL)
        elif level == 6 :
            log.setLevel(logging.DEBUG)
            self.stream = ''
        elif level == 7 :
            log.setLevel(logging.INFO)
            self.stream = ''
        if self.stream == 'logfile':
            self.stream = logging.FileHandler(self.logname)
        else:
            self.stream = logging.StreamHandler()
        self.stream.setLevel(level)
        self.stream.setFormatter(self.formatter)
        log.addHandler(self.stream)

        return log
