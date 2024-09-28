import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.patheffects as pe
import numpy as np
import openeo
import osmnx as ox
import pytz
import rasterio
import xarray
from PIL import Image
from geopandas import GeoSeries
from matplotlib import pyplot as plt
from numpy import ndarray
from openeo import Connection, DataCube
from rasterio import DatasetReader
from rasterio.plot import reshape_as_image, reshape_as_raster
from shapely.geometry.geo import shape

from basic_log import log
from config import app_conf
from config.config import TargetConfig
from pixel_algos.true_color import evaluatePixelArray, mode_depth


def openeo_combine(source_mode="int16", png_mode: str = "uint16", clean_tmp: bool = False, do_thumbnail: bool = True, force_png: bool = False):
    connection: Connection = openeo.connect("openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc_client_credentials(client_id=app_conf.openeo_conf.client_id,
                                                    client_secret=app_conf.openeo_conf.client_secret)

    tmp_base: Path = Path(app_conf.general.tmp_dir)
    save_base: Path = Path(app_conf.general.save_dir)

    name: str
    t_conf: TargetConfig
    for name, t_conf in app_conf.targets.items():
        log(f"target {name} combined starting", level=logging.INFO)

        dt_now: datetime = datetime.now(tz=pytz.UTC)
        dt_start: datetime = dt_now - timedelta(days=t_conf.day_lag)

        dl_name: Path = Path(name, f"{dt_now.strftime('%Y-%m-%dT%H-%M')}_{t_conf.collection_id.lower()}_true-color")
        log(json.dumps(connection.describe_collection(t_conf.collection_id)), logging.DEBUG)
        # Get Stats
        sentinel2_stats: DataCube = connection.load_collection(
            t_conf.collection_id,
            spatial_extent=t_conf.bbox,
            temporal_extent=[dt_start.date(), dt_now.date()],
            bands=t_conf.bands_rgb[0:1],
            max_cloud_cover=t_conf.max_cloud_cover
        )

        stats_file: Path = Path(tmp_base, dl_name.with_suffix(".nc"))
        stats_file.parent.mkdir(exist_ok=True, parents=True)
        try:
            sentinel2_stats.download(stats_file)
        except Exception as e:
            log(f"download failed: {e}", level=logging.ERROR)
            raise e
        nc_data = xarray.open_dataset(stats_file)
        nc_dt: datetime = datetime.fromtimestamp(nc_data["t"].to_numpy().tolist()[-1] / 1000000000, tz=pytz.utc)
        nc_data.close()
        log(f"{stats_file} complete, latest ts {nc_dt}", level=logging.INFO)

        png_fi: Path = Path(save_base, nc_dt.strftime("%Y"), name,
                            f"{nc_dt.strftime('%Y-%m-%dT%H-%M')}_{t_conf.collection_id.lower()}_true-color.png")
        if png_fi.exists() and not force_png:
            log(f"{png_fi} already exists, skipping", level=logging.INFO)
            continue

        # Get Data
        data_name: Path = Path(name, f"{nc_dt.strftime('%Y-%m-%dT%H-%M')}_{t_conf.collection_id.lower()}_true-color")
        sentinel2_cube: DataCube = connection.load_collection(
            t_conf.collection_id,
            spatial_extent=t_conf.bbox,
            temporal_extent=[dt_start.date(), dt_now.date()],
            bands=t_conf.bands_rgb,
            max_cloud_cover=t_conf.max_cloud_cover,
        )

        sentinel2_cube = sentinel2_cube.max_time()

        data_file: Path = Path(tmp_base, data_name.with_suffix(".tiff"))
        data_file.parent.mkdir(exist_ok=True, parents=True)
        sentinel2_cube.download(data_file)
        log(f"{data_file} complete", level=logging.INFO)

        rfi: DatasetReader
        with rasterio.open(data_file) as rfi:
            raster: np.ndarray = rfi.read()
        raster_shaped: np.ndarray = reshape_as_image(raster)

        new_arr = np.empty(raster_shaped.shape, dtype=png_mode)
        for height_y in range(raster_shaped.shape[0]):
            log(f"y {height_y}/{raster_shaped.shape[0]}", level=logging.DEBUG)
            for width_x in range(raster_shaped.shape[1]):
                new_pix = evaluatePixelArray(raster_shaped[height_y, width_x], depth_in=mode_depth[source_mode],
                                             depth_out=mode_depth[png_mode])
                new_arr[height_y][width_x] = new_pix

        if t_conf.osm_feature_ids:
            new_arr = overlays(new_arr, t_conf.bbox, png_mode, t_conf.osm_feature_ids)


        png_fi.parent.mkdir(exist_ok=True, parents=True)
        log(f"{png_fi} convert complete", level=logging.INFO)
        with rasterio.open(png_fi, mode="w", driver="PNG", height=new_arr.shape[0],
                           width=new_arr.shape[1], count=new_arr.shape[2], dtype=png_mode) as dst:
            dst.write(reshape_as_raster(new_arr))
        if do_thumbnail:
            thumbnail_size: int = 600
            dest_path: Path = png_fi.parent / (png_fi.stem + "_thumbnail" + png_fi.suffix)
            with Image.open(png_fi, mode="r") as src:
                src.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)

                src.save(dest_path)

        log(f"{png_fi} complete", level=logging.INFO)

        if clean_tmp:
            log(f"tmp file cleanup complete", level=logging.INFO)
            stats_file.unlink(missing_ok=True)
            data_file.unlink(missing_ok=True)


def overlays(base_img: ndarray, bbox: dict[str, float], mode: str, feature_ids: list[str]) -> ndarray:
    # Get OSM Items

    osm_items = ox.geocode_to_gdf(query=feature_ids, by_osmid=True)

    # Plot Setup
    fig, ax = plt.subplots()
    ax.axis('off')

    fig.set_size_inches(base_img.shape[1] / 100, base_img.shape[0] / 100)
    fig.set_dpi(100)

    ax = fig.add_axes((0, 0, 1, 1))
    fig.set_frameon(False)
    ax.axis('off')
    plt.ylim([bbox["south"], bbox["north"]])
    plt.xlim([bbox["west"], bbox["east"]])

    # Plot Items
    for osm_item in osm_items.iterfeatures():
        geo: GeoSeries = GeoSeries(shape(osm_item["geometry"]), crs="EPSG:32610")

        ax.annotate(osm_item["properties"]["name"], xy=(geo.centroid.x, geo.centroid.y), xytext=(7, -3),
                    textcoords="offset points", fontsize=20, color="black",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.plot(geo.centroid.x, geo.centroid.y, marker="o", color="lightblue", markersize=10)

    # Store plot as Image
    c = fig.canvas
    c.draw()
    plot_img_flat: ndarray = np.frombuffer(c.tostring_argb(), dtype='uint8')  # (H * W * 3,)
    plot_img: ndarray = plot_img_flat.reshape(*reversed(c.get_width_height()), 4)  # (H, W, 3)
    plot_img = plot_img.astype(dtype=mode)
    plot_img = (plot_img / mode_depth["uint8"]) * mode_depth[mode]

    # Overlay
    new_arr = np.empty(plot_img.shape, dtype=plot_img.dtype)
    for x in range(plot_img.shape[0]):
        for y in range(plot_img.shape[1]):
            # If not a white pixel, store overlay pixels, else store base pixels
            # assumes rgb to rgba
            if plot_img[x][y].tolist()[0] != 0.0:
                new_arr[x][y] = plot_img[x][y].tolist()[1:] + [mode_depth[mode]]
            else:
                new_arr[x][y] = base_img[x][y].tolist() + [mode_depth[mode]]

    return new_arr


def main():
    openeo_combine(clean_tmp=False, png_mode=app_conf.general.png_mode)


if __name__ == '__main__':
    main()
