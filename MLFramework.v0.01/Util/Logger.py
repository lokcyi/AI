import logging, sys
import logging.handlers
import pathlib
class Logger(object):
    def __init__(self, name='logger', level=logging.DEBUG):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        # while self.logger.hasHandlers():
        #     for i in self.logger.handlers:
        #         self.logger.removeHandler(i)
        if not self.logger.hasHandlers():
            self.logger.propagate = False
            pathlib.Path('./log/%s.log' % name).parent.mkdir(parents=True, exist_ok=True)

            # fh = logging.FileHandler('%s.log' % name, 'w')
            # Add the log message handler to the logger
            fh = logging.handlers.RotatingFileHandler(
                        filename=('./log/%s.log' % name), maxBytes=5*1024*1024, backupCount=10, encoding="utf-8", delay=0)

            self.logger.addHandler(fh)

            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.ERROR)
            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)-7s] - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # add the handlers to the logger
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


if __name__ == "__main__":
    log = Logger(name='test')
    log.debug('ML Base init..%s' % '測試')
    log.debug('ML Base init..%s' % '測試')
    log.debug('ML Base init..%s' % '測試')
    log = Logger(name='test')
    log.debug('ML Base init..%s' % '測試')
    log.debug('ML Base init..%s' % '測試')