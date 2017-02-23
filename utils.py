"""
Common functions are written here
"""

import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

logger.info("Test info statement")
logger.debug("Test debug statement")
logger.warn("Test warn statement")
logger.error("Test error statement")
