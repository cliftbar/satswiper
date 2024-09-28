from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class CopernicusConfig:
    cdse_user: str
    cdse_password: str


@dataclass()
class CatalogApiConfig:
    client_name: str
    client_id: str
    client_secret: str


@dataclass()
class OpenEoConfig:
    client_name: str
    client_id: str
    client_secret: str


@dataclass()
class TargetConfig:
    collection_id: str
    bands_rgb: list[str]
    bbox: dict[str, float]
    day_lag: int
    max_cloud_cover: int
    osm_feature_ids: list[str]


@dataclass
class GeneralConfig:
    tmp_dir: str
    save_dir: str
    png_mode: str
    log_level: str = "debug"


@dataclass
class AppConfig:
    general: GeneralConfig
    copernicus: CopernicusConfig
    catalog_api: CatalogApiConfig
    openeo_conf: OpenEoConfig
    targets: dict[str, TargetConfig]


def init_config(conf_fi: Path) -> AppConfig:
    with open(conf_fi) as conf:
        conf_vals: dict = yaml.safe_load(conf)

        general_conf: GeneralConfig = GeneralConfig(**conf_vals["general"])
        copernicus_conf: CopernicusConfig = CopernicusConfig(**conf_vals["copernicus"])
        catalog_conf: CatalogApiConfig = CatalogApiConfig(**conf_vals["catalog_api"])
        openeo_conf: OpenEoConfig = OpenEoConfig(**conf_vals["openeo"])
        targets_conf: dict[str, TargetConfig] = {k: TargetConfig(**v) for k, v in conf_vals["targets"].items()}

    return AppConfig(general_conf, copernicus_conf, catalog_conf, openeo_conf, targets_conf)
