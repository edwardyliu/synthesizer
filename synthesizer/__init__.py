# synthesizer/__init__.py

__version__ = "1.0.0"
__date__ = "2025-02-01"
__author__ = "Edward Y. Liu"
__status__ = "Development"

import logging

# import classes
from .dataset.synthesizer import DatasetSynthesizer


logging.basicConfig(
    level=logging.INFO,
    filename="synthesizer.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("synthesizer")
logger.info("synthesizer: initialized.")
