from pathlib import Path

from config.config import AppConfig, init_config

app_conf: AppConfig = init_config(Path("./resources/config.yaml"))