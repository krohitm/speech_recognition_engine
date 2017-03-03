import logging.config

logging.config.fileConfig('logging.conf')
log = logging.getLogger(__name__)

log.info("Logger initialised")
log.debug("Logger initialised")
log.warn("Logger initialised")
log.error("Logger initialised")
