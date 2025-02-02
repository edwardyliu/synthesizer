# datasynth/feature/__init__.py

import logging

from .normal import NormalDistributionGenerator
from .uniform import UniformDistributionGenerator


logger = logging.getLogger("datasynth")
logger.info("datasynth.feature initialized")
