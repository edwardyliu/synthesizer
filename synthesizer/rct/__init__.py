# synthesizer/design/__init__.py

import logging

from .generator import RCTGenerator, RCTGeneratorGrouped
from .crossover import CrossoverRCTGenerator
from .factorial import FactorialRCTGenerator
from .matched_pairs import MatchedPairsRCTGenerator
from .parallel import ParallelRCTGenerator
from .withdrawal import WithdrawalRCTGenerator

logger = logging.getLogger("synthesizer")
logger.info("synthesizer.rct: initialized.")
