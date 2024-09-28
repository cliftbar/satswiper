import logging
from datetime import datetime

import pytz

from config import app_conf

log_level: int = logging.getLevelName(app_conf.general.log_level.upper())


def log(msg: str, level: int = logging.INFO, source: str = None):
    if log_level <= level:
        before: str = f"{datetime.now(tz=pytz.UTC).isoformat()} {logging.getLevelName(level)}"
        if source is not None:
            before = f"{before} {source}"
        print(f"{before}: {msg}")
