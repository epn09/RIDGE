import argparse
import functools
import json
import time
from datetime import datetime

import ee
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.enums import WktVersion
from tagee import terrainAnalysis


def single_aspect_difference(aspect: ee.Image, kernel: ee.Kernel) -> ee.Image:
    diff = aspect.convolve(kernel).abs()
    diff = ee.Image(180).subtract(diff).abs()
    return diff


def aspect_difference(dem: ee.Image) -> ee.Image:
    # Aspect, Aspect difference
    aspect = ee.Terrain.aspect(dem)

    # Weights for a 3x3 kernel
    w00 = [0, 0, 0]
    w11, w12, w13 = np.identity(3)

    kernels = [
        [w11, w00, -w13],
        [w13, w00, -w11],
        [w12, w00, -w12],
        [w00, w11 - w13, w00],
    ]
    kernels = [
        ee.Kernel.fixed(3, 3, np.array(kernel).tolist(), 1, 1, False)
        for kernel in kernels
    ]
    aspect_difference = functools.reduce(
        lambda x, y: x.min(y),
        [single_aspect_difference(aspect, kernel) for kernel in kernels],
    )
    return aspect_difference


def read_dem(
    dem_id: str, mask_below_zero=True, crs: ee.Projection | None = None
) -> ee.Image:
    """Read DEM and perform water masking."""

    # DEM
    dem = ee.Image(dem_id)

    if mask_below_zero:
        # Mask less than or equal to 0m area for DEM
        dem = dem.updateMask(dem.gt(0))

    if crs is not None:
        dem = dem.reduceResolution(ee.Reducer.mean()).reproject(
            crs.atScale(dem.projection().nominalScale())
        )

    return dem


def relative_elevation(dem: ee.Image) -> ee.Image:
    zmin = dem.reduceNeighborhood(
        reducer=ee.Reducer.min(), kernel=ee.Kernel.square(30, "meters")
    )
    rel_elev = dem.subtract(zmin)
    return rel_elev


def elevation_prime(dem: ee.Image) -> ee.Image:
    zmin = dem.reduceNeighborhood(
        reducer=ee.Reducer.median(), kernel=ee.Kernel.circle(30, "meters")
    )
    rel_elev = dem.subtract(zmin)
    return rel_elev


def get_inputs(dem: ee.Image, scale_to_integer=True) -> ee.Image:
    # Relative elevation
    Rev = relative_elevation(dem).rename("rel_elev")

    # Slope
    Slp = ee.Terrain.slope(dem).rename("slope")

    # Calculating aspect difference
    Adf = aspect_difference(dem).rename("aspect_diff")

    # Calculate curvature
    Crv: ee.Image = terrainAnalysis(dem).select("MaximalCurvature").rename("max_curv")

    # Elevation prime
    Epr = elevation_prime(dem).rename("elev_prime")

    if not scale_to_integer:
        return Rev.addBands([Slp, Adf, Crv])

    Rev = Rev.multiply(10)
    Slp = Slp.multiply(10)
    Adf = Adf.multiply(10)
    Crv = Crv.multiply(1000)
    Epr = Epr.multiply(10)
    max_value = 2**15 - 1
    return (
        Rev.addBands([Slp, Adf, Crv, Epr])
        .min(max_value)
        .max(-max_value)
        .round()
        .toInt16()
        .updateMask(Crv.gte(10))
    )


def read_config(
    args: argparse.Namespace,
) -> tuple[ee.Geometry, str, str, str, int, str | None]:
    cases_df = pd.read_csv("cases.csv").set_index("case_name")
    case_info = cases_df.loc[args.case_name]

    with open(case_info["geojson_file"]) as fp:
        geojson = json.load(fp)
    basin = ee.Geometry(geojson)

    if case_info["buffer_km"] > 0:
        basin = basin.buffer(float(case_info["buffer_km"] * 1e3))

    suffix = "input_vars"
    no_data = -(2**15)

    suffix = "_" + (args.suffix or suffix)

    output_dir: str = (
        case_info["output_dir"]
        + suffix
        + "_"
        + datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    )
    prefix: str = case_info["prefix"] + suffix
    dem_id: str = case_info["dem_id"]
    projection = case_info["projection"]
    if not isinstance(projection, str):
        projection = None

    return basin, output_dir, prefix, dem_id, no_data, projection


def main(args: argparse.Namespace):
    ee.Initialize()

    basin, output_dir, prefix, dem_id, no_data, projection = read_config(args)

    if projection is not None:
        crs = ee.Projection(
            CRS.from_string(projection).to_wkt(version=WktVersion.WKT1_GDAL)
        )
    else:
        crs = None

    scale = 10
    dem = read_dem(dem_id, crs=crs)
    output_image = get_inputs(dem)

    output_image = output_image.unmask(no_data, sameFootprint=False)

    task = ee.batch.Export.image.toDrive(
        image=output_image,
        description=prefix,
        scale=scale,
        region=basin,
        folder=output_dir,
        maxPixels=1e13,  # Cannot be larger than 1e13
        fileFormat="GeoTIFF",
        shardSize=256,
        fileDimensions=23296,
        skipEmptyTiles=True,
        formatOptions={"cloudOptimized": True, "noData": no_data},
    )
    task.start()

    if args.wait:
        while task.active():
            print("Waiting...")
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wait", action="store_true", help="Wait for the GEE task to finish."
    )

    parser.add_argument("case_name")
    parser.add_argument("--suffix", type=str, help="Override default suffix")

    args = parser.parse_args()

    main(args)
