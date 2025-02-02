# datasynth/__init__.py

__version__ = "1.0.0"
__date__ = "2025-02-01"
__author__ = "Edward Y. Liu"
__status__ = "Development"

import logging

from . import feature

logging.basicConfig(
    level=logging.INFO,
    filename="datasynth.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("datasynth")
logger.info("datasynth initialized")
