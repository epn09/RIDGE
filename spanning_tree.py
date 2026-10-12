# %%
import logging
from pathlib import Path
from typing import Literal

import dask
import dask.array as da
import geopandas as gpd
import graph
import numpy as np
import pandas as pd
import portalocker
import rioxarray
import shapely
import xarray as xr
from numba import njit
from tap import Tap
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.dask import TqdmCallback

_logger = logging.getLogger(__name__)


class CmdArguments(Tap):
    input_path: Path  # Path to levee index file.
    output_path: str  # Output path
    min_length: int = 10  # Minimum length of paths to keep (in pixels)
    lock: bool = False  # Whether to lock the output file when writing concurrently
    mode: Literal["a", "w"] = "w"
    """Mode for writing the output file: 'a' for append, 'w' for write."""
    simplify: bool = False  # Whether to simplify the output geometries
    progress_bar: bool = False  # Show progress bar for tasks managed by dask
    split_length: int = 0
    """Split long paths into chunks of this length (in pixels).
    A diagonal connection is counted as sqrt(2) pixels.
    If 0, no splitting is done.
    """
    layer_name: str | None = "lines"
    max_curv_thresh: float = 0.01
    input_var_names: list[str] | None = None

    def configure(self):
        self.add_argument("input_path")
        self.add_argument("output_path")


# %%
@njit
def exclude_short_paths(path_vertices, path_lens, threshold):
    """
    Exclude paths shorter than threshold.
    """
    new_path_vertices = []
    new_path_lens = []
    start = 0
    for plen in path_lens:
        end = start + plen
        if plen >= threshold:
            new_path_vertices.extend(path_vertices[start:end])
            new_path_lens.append(plen)
        start += plen
    return np.array(new_path_vertices), np.array(new_path_lens)


# %%
@njit
def split_long_paths(path_vertices, path_lens, split_length, nx):
    """
    Split paths into chunks of length `split_length`, roughly.

    For example, when `split_length` is 10, a path of length 24 will be split into
    two chunks of length 12, instead of three chunks of length 10, 10, and 4.

    long_path_ids records the original path ID for each chunk.
    """

    iy = path_vertices // nx
    ix = path_vertices % nx

    new_path_vertices = []
    new_path_lens = []
    long_path_ids = []
    start = 0
    for plen in path_lens:
        end = start + plen

        length = 0
        for i in range(start, end - 1):
            length += np.sqrt((ix[i + 1] - ix[i]) ** 2 + (iy[i + 1] - iy[i]) ** 2)

        n_chunks = max(1, int(length / split_length))
        chunk_length = length / n_chunks

        long_path_id = len(new_path_lens)

        if n_chunks <= 1:
            new_path_vertices.extend(path_vertices[start:end])
            new_path_lens.append(plen)
            long_path_ids.append(long_path_id)
            start += plen
            continue

        length = 0
        chunk_start = start
        finished_chunks = 0
        for i in range(start, end - 1):
            length += np.sqrt((ix[i + 1] - ix[i]) ** 2 + (iy[i + 1] - iy[i]) ** 2)
            if length >= chunk_length:
                new_path_vertices.extend(path_vertices[chunk_start : i + 2])
                new_path_lens.append(i + 2 - chunk_start)
                long_path_ids.append(long_path_id)
                chunk_start = i + 1
                length = 0

                finished_chunks += 1
                if finished_chunks == n_chunks - 1:
                    # Stop here, the rest of the path will be the last chunk
                    break

        if end - chunk_start == 1:
            # Very rarely, the last chunk may consist of a single vertex
            # due to accumulation of rounding errors.
            # In this case, we add the last vertex to the previous chunk.
            new_path_vertices.append(path_vertices[chunk_start])
            new_path_lens[-1] += 1
        elif chunk_start < end:
            new_path_vertices.extend(path_vertices[chunk_start:end])
            new_path_lens.append(end - chunk_start)
            long_path_ids.append(long_path_id)

        start += plen

    return np.array(new_path_vertices), np.array(new_path_lens), np.array(long_path_ids)


# %%
def non_zero(arr, threshold):
    idx = da.flatnonzero(arr >= threshold)
    values = da.ravel(arr)[idx]
    return dask.compute(idx, values)


# %%
def open_input_vars_ds(filepath):
    ds = rioxarray.open_rasterio(filepath, chunks={"band": "auto", "y": 512})
    ds = ds.where(ds != -32768)
    ds[0] /= 10
    ds[1] /= 10
    ds[2] /= 10
    ds[3] /= 1000
    ds[4] /= 10
    return ds


# %%
@njit
def stats(arr, lens):
    res = np.empty((len(lens), 6), dtype=np.float32)
    start = 0
    for i in range(len(lens)):
        end = start + lens[i]
        res[i, :5] = np.nanpercentile(arr[start:end], [5, 25, 50, 75, 95])
        median = res[i, 2]
        res[i, 5] = np.nanmedian(np.abs(arr[start:end] - median))
        start = end
    return res


def to_geoseries(
    path_vertices, path_lens, ds, properties: dict[str, xr.DataArray] | None = None
):
    if properties is None:
        properties = {}

    iy, ix = np.unravel_index(path_vertices, (ds.sizes["y"], ds.sizes["x"]))

    coords = np.array([ds.x.values[ix], ds.y.values[iy]]).T
    res = gpd.GeoSeries(
        shapely.MultiLineString(np.split(coords, np.cumsum(path_lens))[:-1]),
        crs=ds.rio.crs,
    ).explode()
    res = gpd.GeoDataFrame({"geometry": res})

    prop_vals = dask.compute(
        {prop_name: da.ravel(ds)[path_vertices] for prop_name, ds in properties.items()}
    )[0]

    def colnames(prop):
        return [
            f"{prop}_5",
            f"{prop}_25",
            f"{prop}_50",
            f"{prop}_75",
            f"{prop}_95",
            f"{prop}_mad",
        ]

    for prop_name, prop_val in prop_vals.items():
        res[colnames(prop_name)] = stats(prop_val, path_lens)

    return res


def to_float32(df: pd.DataFrame | gpd.GeoDataFrame):
    columns = df.select_dtypes(include=["float64"]).columns
    df[columns] = df[columns].astype(np.float32)
    return df


def calc_spanning_tree(ds, max_curv_thresh):
    _logger.info("Extracting points with curvature >= %s", max_curv_thresh)
    idx, values = non_zero(ds[3], max_curv_thresh)  # Maximum curvature larger than 0.01
    _logger.info("Found %d valid points", len(idx))

    if len(idx) == 0:
        return np.array([]), np.array([])

    _logger.info("Calculating spanning trees")
    vertices, compressed_edge = graph.compress(
        graph.spanning_tree(idx, values, ds.sizes["y"], ds.sizes["x"])
    )
    _logger.info("Decomposing the spanning trees")
    path_vertices, path_lens = graph.decompose(compressed_edge)
    path_vertices = graph.decompress(path_vertices, vertices)

    return path_vertices, path_lens


def main(args: CmdArguments):
    _logger.info("Opening input variables")
    ds = open_input_vars_ds(args.input_path)

    if args.input_var_names:
        if len(args.input_var_names) != ds.sizes["band"]:
            raise ValueError(
                f"Number of input variable names ({len(args.input_var_names)}) "
                "does not match the number of bands in the input dataset "
                f"({ds.sizes['band']})."
            )
        properties = dict(zip(args.input_var_names, ds))
    elif hasattr(ds, "long_name"):
        properties = dict(zip(ds.long_name, ds))
    else:
        raise ValueError(
            "Input variable names are not provided and cannot be inferred from the dataset. "
            "Please provide input variable names using the --input-vars-name argument."
        )

    _logger.info("Input shape: %s", ds.shape)

    _logger.info("Starting spanning tree calculation")
    path_vertices, path_lens = calc_spanning_tree(ds, args.max_curv_thresh)
    _logger.info("Found %d vertices in %d paths", len(path_vertices), len(path_lens))

    if len(path_vertices) == 0:
        _logger.info("No valid vertex found. Nothing to do.")
        return

    if args.min_length > 0:
        _logger.info("Removing short branches")
        # 10 here is 10 pixels, so equivalent to 100 meters
        path_vertices, path_lens = exclude_short_paths(
            path_vertices, path_lens, args.min_length
        )
        _logger.info(
            "%d vertices remain in %d paths", len(path_vertices), len(path_lens)
        )

    if len(path_vertices) == 0:
        _logger.info("No valid vertex found after removing short branches.")
        return

    if args.split_length > 0:
        _logger.info(
            "Splitting long paths into chunks of length %d pixels", args.split_length
        )
        path_vertices, path_lens, long_path_ids = split_long_paths(
            path_vertices, path_lens, args.split_length, ds.sizes["x"]
        )
        _logger.info(
            "%d vertices in %d paths after splitting",
            len(path_vertices),
            len(path_lens),
        )
    else:
        long_path_ids = np.arange(len(path_lens))

    _logger.info("Calculating geomorphic properties")
    res = to_geoseries(path_vertices, path_lens, ds, properties)
    res = to_float32(res)
    res["long_line_id"] = long_path_ids + 1  # Make it 1-based ID

    if args.simplify:
        tolerance = abs(float(ds.x[0] - ds.x[1]))
        _logger.info("Simplifying geometries. Maximum tolerance: %g", tolerance)
        res.geometry = res.geometry.simplify(tolerance)

    _logger.info("Calculating additional properties")
    res["sinuosity"] = res.geometry.length / res.geometry.interpolate(0).distance(
        res.geometry.interpolate(1, True)
    )

    lock = portalocker.Lock(
        Path(args.output_path).with_suffix(".lock"),
        flags=portalocker.LockFlags.EXCLUSIVE,
    )
    try:
        if args.lock:
            _logger.info("Acquiring exclusive lock")
            lock.acquire()

        _logger.info("Writing results to file")
        res.to_file(args.output_path, layer=args.layer_name, mode=args.mode)
    finally:
        _logger.info("Releasing lock")
        lock.release()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-s [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    )
    with dask.config.set(num_workers=1), logging_redirect_tqdm():
        args = CmdArguments().parse_args()
        if args.progress_bar:
            pbar = TqdmCallback()
            pbar.register()
        main(args)
    _logger.info("Done")
