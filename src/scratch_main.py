import json
import logging
from datetime import datetime, timedelta
from typing import Any

import matplotlib.patheffects as pe
import numpy as np
import openeo
import osmnx as ox
import pytz
import rasterio
import xarray
from cdsetool.query import query_features, FeatureQuery, geojson_to_wkt
from geopandas import GeoDataFrame, GeoSeries
from matplotlib import pyplot as plt, pyplot
from oauthlib.oauth2 import BackendApplicationClient
from openeo import Connection, DataCube
from rasterio import DatasetReader
from rasterio.plot import reshape_as_image, reshape_as_raster
from requests import Response
from requests_oauthlib import OAuth2Session
from sentinelhub import SentinelHubRequest, DataCollection, MimeType, SHConfig, CRS, bbox_to_dimensions, BBox
from shapely.geometry.geo import shape

from basic_log import log
from config import app_conf
from config.config import TargetConfig
from pixel_algos.scratch import evaluatePixelArray
from pixel_algos.true_color import mode_depth

node_ids: list[str] = ["N150975844", "N357310459", "N357316899", "N2630391344", "N5140704501", "N357306798",
                       "N2630400840", "N357309129", "N357306559", "N357311595", "N2630373085", "N4969488263",
                       "N357314113"]
way_ids: list[str] = ["W89636422", "W146985880", "W149173768", "W368567701", "W542527388", "W548667995",
                      "W722330441"]


def cdse_tool():
    dtnow: datetime = datetime.now(tz=pytz.UTC)

    with open("./resources/mthood.json", "r") as gf:
        geometry: dict = json.load(gf)
    features: FeatureQuery = query_features("Sentinel2",
                                            {"startDate": dtnow - timedelta(days=10), "completionDate": dtnow,
                                             # "processingLevel": "S2MSI2A",
                                             # "sensorMode": "TCI",
                                             # "productType": "MSIL2A",
                                             "geometry": geojson_to_wkt(geometry), }, )

    print(features.features[0])

    # dl: Path = Path(f"./download/{int(dtnow.timestamp())}")
    # dl.mkdir(exist_ok=True, parents=True)
    # list(download_features(features, str(dl), {"concurrency": 4, "monitor": StatusMonitor(),
    #     "credentials": Credentials(app_conf.copernicus.cdse_user, app_conf.copernicus.cdse_password), }, ))


def catalog_auth():
    # Your client credentials
    client_id = app_conf.catalog_api.client_id
    client_secret = app_conf.catalog_api.client_secret

    # Create a session
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    # Get token for the session
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=client_secret, include_client_id=True)

    # All requests using this session will have an access token automatically added
    resp = oauth.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")
    print(resp.status_code)

    return oauth, token


def catalog_api():
    dtnow: datetime = datetime.now(tz=pytz.UTC)
    dtstart: datetime = dtnow - timedelta(days=10)
    with open("./resources/mthood.json", "r") as gf:
        geometry: dict = json.load(gf)

    client, token = catalog_auth()

    data = {
        "datetime": f"{dtstart.isoformat()}/{dtnow.isoformat()}",
        "collections": ["sentinel-2-l2a"],
        "limit": 5,
        "intersects": geometry["features"][0]["geometry"],
    }

    url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    response: Response = client.post(url, json=data)

    print(response.text)


def plot_image(
        image: np.ndarray, factor: float = 1.0, clip_range: tuple[float, float] | None = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def sentinel_hub():
    config: SHConfig = SHConfig()
    config.instance_id = "satswiper"
    config.sh_client_id = app_conf.catalog_api.client_id
    config.sh_client_secret = app_conf.catalog_api.client_secret
    config.sh_base_url = "https://services.sentinel-hub.com"
    config.sh_auth_base_url = "https://services.sentinel-hub.com"
    print(config)

    resolution_m: int = 10

    with open("./resources/mthood.json", "r") as gf:
        geometry: dict = json.load(gf)

    mthood_bbox = BBox(bbox=geometry["bbox"], crs=CRS.WGS84)
    mthood_size = bbox_to_dimensions(mthood_bbox, resolution=resolution_m)

    with open("./resources/sentinel2_l2a_true-color-optimized.evalscript", "r") as fi:
        evalscript_true_color = fi.read()

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    name="s2", service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=('2022-06-01', '2022-06-30'),
                other_args={"dataFilter": {"mosaickingOrder": "leastCC"}})
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=mthood_bbox,
        size=mthood_size,
        config=config,
    )

    plot_image(request_true_color.get_data()[0], factor=3.5 / 255, clip_range=(0, 1))


def openeo_dl():
    connection: Connection = openeo.connect("openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc_client_credentials(client_id=app_conf.catalog_api.client_id,
                                                    client_secret=app_conf.catalog_api.client_secret)

    dtnow: datetime = datetime.now(tz=pytz.UTC)
    dtstart: datetime = dtnow - timedelta(days=10)

    # print(json.dumps(connection.describe_collection("SENTINEL2_L2A")))
    sentinel2_cube: DataCube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": -121.85, "south": 45.26, "east": -121.52, "north": 45.46},
        temporal_extent=[dtstart.date(), dtnow.date()],
        bands=["B04", "B03", "B02"],
        max_cloud_cover=100,
    )

    sentinel2_cube.max_time()

    # sentinel2_cube.apply_dimension(process=tci_function, dimension="x")

    dl_name = "rgb"
    dl_file = f"./download/{dl_name}.tiff"
    # sentinel2_cube.download(dl_file)

    # tiff: ImageFile = Image.open(dl_file)
    # png = tiff.convert("RGB")
    # png.save(f"./download/{dl_name}.png", "PNG")

    rfi: DatasetReader
    with rasterio.open(dl_file) as rfi:
        raster: np.ndarray = rfi.read()

    raster_shaped: np.ndarray = reshape_as_image(raster)
    #

    new_arr = np.empty(raster_shaped.shape, dtype=np.uint8)
    for height_y in range(raster_shaped.shape[0]):
        for width_x in range(raster_shaped.shape[1]):
            new_pix = evaluatePixelArray(raster_shaped[height_y, width_x])
            new_arr[height_y][width_x] = new_pix
            # print(new_arr)
            # new_arr[height_y][width_x][1] = new_pix * 256
            # new_arr[height_y][width_x][2] = new_pix * 256

            # This works
            # new_arr[height_y][width_x][0] = raster_shaped[height_y, width_x][0] * 2.5 / 32760 * 256
            # new_arr[height_y][width_x][1] = raster_shaped[height_y, width_x][1] * 2.5 / 32760 * 256
            # new_arr[height_y][width_x][2] = raster_shaped[height_y, width_x][2] * 2.5 / 32760 * 256
            # if new_pix[0] > 1:
            #     print(new_pix)
    # raster_shaped.astype("uint8")
    pyplot.imshow(new_arr)
    pyplot.show()

    with rasterio.open(f"./download/{dl_name}.png", mode="w", driver="PNG", height=new_arr.shape[0],
                       width=new_arr.shape[1], count=3, dtype="uint8") as dst:
        dst.write(reshape_as_raster(new_arr))

    # blue = sentinel2_cube.band("B02")
    # red = sentinel2_cube.band("B04")
    # green = sentinel2_cube.band("B03")
    # nir = sentinel2_cube.band("B08") * 0.0001

    # blue.max_time()
    # blue.download("./download/blue.tiff")

    # evi_cube = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)

    # evi_cube = pixel_algos.true_color.evaluatePixel({"B02": blue, "B04": red, "B03": green})
    # test = pixel_algos.true_color.sRGB(blue)
    #
    # evi_composite = evi_cube.max_time()
    # evi_composite.download("./download/evi-composite.png")

    # s2f = sentinel2_cube.filter_bands(["B04", "B03", "B02"])
    # merge_cubes()
    # s2f.apply(pixel_algos.true_color.satEnh2)
    # s2f.save_result
    # s2f.max_time()
    # s2f.download("./download/s2.png")

    # blue_cube_math = (red + green + blue) / 3.0 * (1 - sat) + blue + sat
    # red_cube_math = (red + green + blue) / 3.0 * (1 - sat) + red + sat
    # green_cube_math = (red + green + green) / 3.0 * (1 - sat) + red + sat

    # blue_cube_math.apply(openeo.pixel_algos.clip, blue_cube_math, 0, 1)
    # blue_cube_math.merge_cubes(red_cube_math)


def openeo_dl(source_mode="int16", png_mode: str = "uint16"):
    connection: Connection = openeo.connect("openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc_client_credentials(client_id=app_conf.openeo_conf.client_id,
                                                    client_secret=app_conf.openeo_conf.client_secret)

    name: str
    t_conf: TargetConfig
    for name, t_conf in app_conf.targets.items():
        log(f"target {name} tiff starting", level=logging.INFO)

        dtnow: datetime = datetime.now(tz=pytz.UTC)
        dtstart: datetime = dtnow - timedelta(days=t_conf.day_lag)

        log(json.dumps(connection.describe_collection("SENTINEL2_L2A")), logging.DEBUG)
        sentinel2_cube: DataCube = connection.load_collection(
            t_conf.collection_id,
            spatial_extent=t_conf.bbox,
            temporal_extent=[dtstart.date(), dtnow.date()],
            bands=t_conf.bands_rgb,
            max_cloud_cover=t_conf.max_cloud_cover,
        )

        sentinel2_cube = sentinel2_cube.max_time()
        dl_name = f"rgb_testgood"
        dl_file = f"./download/{dl_name}.tiff"
        sentinel2_cube.download(dl_file)
        log(f"{dl_file} complete", level=logging.INFO)

        rfi: DatasetReader
        with rasterio.open(dl_file) as rfi:
            raster: np.ndarray = rfi.read()
        raster_shaped: np.ndarray = reshape_as_image(raster)

        new_arr = np.empty(raster_shaped.shape, dtype=png_mode)
        for height_y in range(raster_shaped.shape[0]):
            log(f"y {height_y}/{raster_shaped.shape[0]}", level=logging.DEBUG)
            for width_x in range(raster_shaped.shape[1]):
                new_pix = evaluatePixelArray(raster_shaped[height_y, width_x], depth_in=mode_depth[source_mode],
                                             depth_out=mode_depth[png_mode])
                new_arr[height_y][width_x] = new_pix

        png_fi: str = f"./download/{dl_name}.png"
        log(f"{png_fi} convert complete", level=logging.INFO)
        with rasterio.open(png_fi, mode="w", driver="PNG", height=new_arr.shape[0],
                           width=new_arr.shape[1], count=3, dtype=png_mode) as dst:
            dst.write(reshape_as_raster(new_arr))
        log(f"{png_fi} complete", level=logging.INFO)


def openeo_dl_nc(source_mode="int16", png_mode: str = "uint16"):
    connection: Connection = openeo.connect("openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc_client_credentials(client_id=app_conf.openeo_conf.client_id,
                                                    client_secret=app_conf.openeo_conf.client_secret)

    name: str
    t_conf: TargetConfig
    for name, t_conf in app_conf.targets.items():
        log(f"target {name} as netCDF starting", level=logging.INFO)

        dtnow: datetime = datetime.now(tz=pytz.UTC)
        dtstart: datetime = dtnow - timedelta(days=t_conf.day_lag)

        log(json.dumps(connection.describe_collection("SENTINEL2_L2A")), logging.DEBUG)
        sentinel2_cube: DataCube = connection.load_collection(
            t_conf.collection_id,
            spatial_extent=t_conf.bbox,
            temporal_extent=[dtstart.date(), dtnow.date()],
            bands=t_conf.bands_rgb,
            max_cloud_cover=t_conf.max_cloud_cover,
        )

        dl_name = f"rgb_combine"
        dl_file = f"./download/{dl_name}.nc"
        # sentinel2_cube.download(dl_file)
        log(f"{dl_file} complete", level=logging.INFO)

        nc_data = xarray.open_dataset(dl_file)
        nc_dt: np.datetime64 = nc_data["t"][-1].values
        log(nc_dt)
        nc_latest = nc_data.sel(t=nc_data["t"][-1].values)

        nc_latest = nc_latest.drop_vars(names="t")
        red = nc_latest["B02"].astype(dtype=source_mode).to_numpy()
        green = nc_latest["B03"].astype(dtype=source_mode).to_numpy()
        blue = nc_latest["B04"].astype(dtype=source_mode).to_numpy()

        new_arr = np.empty((*nc_latest["B02"].shape, 3), dtype=png_mode)
        for height_y in range(nc_latest["B02"].shape[0]):
            log(f"y {height_y}/{nc_latest['B02'].shape[0]}", level=logging.DEBUG)
            for width_x in range(nc_latest["B02"].shape[1]):
                pix = [red[height_y, width_x], green[height_y, width_x], blue[height_y, width_x]]
                new_pix = evaluatePixelArray(pix, depth_in=mode_depth[source_mode], depth_out=mode_depth[png_mode])
                new_arr[height_y][width_x] = new_pix

        png_fi: str = f"./download/{dl_name}.png"
        log(f"{png_fi} convert complete", level=logging.INFO)
        with rasterio.open(png_fi, mode="w", driver="PNG", height=new_arr.shape[0],
                           width=new_arr.shape[1], count=3, dtype=png_mode) as dst:
            dst.write(reshape_as_raster(new_arr))
        log(f"{png_fi} save complete", level=logging.INFO)


def ox_test():
    rfi: DatasetReader
    # rfi = rasterio.open("E:/MyFiles/Code/satswiper/src/download/mt-hood/2024-09-23T00-00_sentinel2_l2a_true-color.tiff")
    rfi = rasterio.open("E:/MyFiles/Code/satswiper/src/download/mt-hood/2024-09-18T00-00_sentinel2_l2a_true-color.png",
                        crs="EPSG:32610")
    raster: np.ndarray = rfi.read()
    # raster = raster / (32760 * 2)
    raster_shaped: np.ndarray = reshape_as_image(raster)

    map_features: dict = {
        "natural": ["peak", "glacier"],
        "name": ["Mount Hood", "Government Camp", "Silcox Hut", "Historic Warming Hut", "Timberline Lodge",
                 "Mt. Hood Skibowl"],
        # "osmid": ["150975844", "2630437418", "368567701", "548667995"]
        # "name": True
    }
    osmids = [150975844, 149173768, 368567701, 548667995, 2630391344, 5140704501, 146985880, 542527388, 89636422,
              357310459, 357316899]
    peaks: GeoDataFrame = ox.features_from_bbox(**app_conf.targets["mt-hood"].bbox, tags=map_features)
    peaks = peaks[peaks.index.isin(osmids, "osmid")]
    nodeIds = ["N150975844", "N357310459", "N357316899", "N2630391344", "N5140704501"]
    wayIds = ["W89636422", "W146985880", "W149173768", "W368567701", "W542527388", "W548667995", ]
    nds = ox.geocode_to_gdf(query=nodeIds + wayIds, by_osmid=True)
    peaks = nds
    # peaks.loc("osmid" == 150975844)
    # peaks.index.get_level_values("osmid")

    # baseTiff = geopandas.read_file("E:/MyFiles/Code/satswiper/src/download/mt-hood/2024-09-23T00-00_sentinel2_l2a_true-color.tiff")

    # fig, ax = plt.subplots()
    # plt.imshow(raster_shaped)
    # plt.plot(data=peaks)
    # plt.plot(baseTiff)
    # geom: GeoSeries = peaks.get(peaks.name == "Elk Cove").geometry
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.set_size_inches(raster.shape[2] / 100, raster.shape[1] / 100)
    fig.set_dpi(100)
    ax = fig.add_axes((0, 0, 1, 1))

    # fig.patch.set_visible(False)
    fig.set_frameon(False)
    # fig.tight_layout()
    # fig.bbox_inches = None
    ax.axis('off')
    # peaks.plot(ax=ax)
    plt.ylim([app_conf.targets["mt-hood"].bbox["south"], app_conf.targets["mt-hood"].bbox["north"]])
    plt.xlim([app_conf.targets["mt-hood"].bbox["west"], app_conf.targets["mt-hood"].bbox["east"]])

    # Know loop location test
    ax.plot(-121.7098, 45.3252, "bo", markersize=20)
    for peak in peaks.iterfeatures():
        geo: GeoSeries = GeoSeries(shape(peak["geometry"]), crs="EPSG:32610")

        ax.annotate(peak["properties"]["name"], xy=(geo.centroid.x, geo.centroid.y), xytext=(7, -3),
                    textcoords="offset points", fontsize=20,
                    path_effects=[pe.withStroke(linewidth=4, foreground="red")])
        # ax.text(geo.centroid.x, geo.centroid.y, peak["properties"]["name"],fontsize=20,position=(5, 5),path_effects=[pe.withStroke(linewidth=4, foreground="red")])
        ax.plot(geo.centroid.x, geo.centroid.y, "bo", markersize=10)
    # plt.show()

    # plt.axis('off')
    c = fig.canvas
    c.draw()
    mode = "uint16"
    image_flat = np.frombuffer(c.tostring_argb(), dtype='uint8')  # (H * W * 3,)
    image = image_flat.reshape(*reversed(c.get_width_height()), 4)  # (H, W, 3)
    image = image.astype(dtype=mode)
    image = (image / mode_depth["uint8"]) * mode_depth[mode]

    raster_shaped = (raster_shaped / mode_depth["uint16"]) * mode_depth[mode]
    new_arr = np.empty(image.shape, dtype=image.dtype)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y].tolist() != [0, mode_depth[mode], mode_depth[mode],
                                        mode_depth[mode]]:
                new_arr[x][y] = image[x][y].tolist()[1:] + [mode_depth[mode]]
            else:
                new_arr[x][y] = raster_shaped[x][y].tolist() + [mode_depth[mode]]
            # if image[x][y].tolist() == [255, 255, 255, 255]:
            #     new_arr[x][y] = [255, 255, 255, 0]
            # else:
            #     new_arr[x][y] = image[x][y].tolist()[1:] + [255]
    with rasterio.open("test.png", mode="w", driver="PNG", height=new_arr.shape[0],
                       width=new_arr.shape[1], count=4, dtype=mode) as dst:
        dst.write(reshape_as_raster(new_arr))
    plt.show()

    # m_mosaic, m_output = merge([raster, new_arr])
    # with rasterio.open("merge.png", mode="w", driver="PNG", height=m_output.shape[0],
    #                    width=m_output.m_output[1], count=4, dtype="uint8") as dst:
    #     dst.write(reshape_as_raster(m_output))


def main():
    # cdse_tool()
    # catalog_api()
    # sentinel_hub()
    openeo_dl()


if __name__ == '__main__':
    main()
