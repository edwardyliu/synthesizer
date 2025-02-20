# synthesizer/blocking/__init__.py

import logging

from .generator import BlockingGenerator, SimpleBlockingGenerator


logger = logging.getLogger("synthesizer")
logger.info("synthesizer.blocking: initialized.")
