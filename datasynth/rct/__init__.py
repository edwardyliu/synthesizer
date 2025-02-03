# datasynth/design/__init__.py

import logging

from .generator import RCTGenerator
from .crossover import CrossoverRCTGenerator
from .parallel import ParallelRCTGenerator


logger = logging.getLogger("datasynth")
logger.info("datasynth.rct: initialized.")
