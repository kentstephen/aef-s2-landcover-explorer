# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.3",
#     "xarray",
#     "zarr>=3.1",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "anywidget>=0.9",
#     "numpy",
#     "duckdb>=1.5.5",
#     "pyproj",
#     "pillow",
# ]
# ///


"""Where the ground changed, from AlphaEarth, and what it is, from ESA WorldCover.

Stephen, 2026-09-25: "the focus is on aef change and change year ... vizing
the embeddings with h3. geoarrow deck.gl like before. xarray-sql and h3ronpy
udf to churn thru the embeddings ... i'm particularly interested in the
viridis aef change hexagons showing me where to look and including other
datasets to show me what's actually happening."

- The map is the AlphaEarth hexagons in viridis: how much the ground's 64
  numbers moved over the year window (or the year its change stood out).
  They say where to look.
- Hold space: the hexagons go and the
  Sentinel-2 yearly mosaic (Earth Genome, 2022 to 2025) fills the view,
  opening on 2022 the first time and after that where you left off. Scroll
  while holding to step through the years; with space held the map can be
  dragged. Let go and the hexagons come back. Over the imagery only the
  outlines: white under the pointer, gold on the picked cell, blue on a
  searched H3 string.
- Below zoom 9 the map is ESA WorldCover 2021 land cover; the hexagons
  take over from zoom 9.
- The card at the top right: the imagery year, how many hexagons in view
  changed in each year, and the clicked hexagon's account (its year-to-year
  steps, and its land cover from ESA WorldCover 2021).

The readers (AlphaEarth COGs and mosaic, the S2 tiles) and the atlas's face
are carried over from s2-wsf-aef-overture-atlas.py, in this repo's history:
https://github.com/kentstephen/aef-s2-landcover-explorer/blob/2f4978aa960960705eb4edee3c71c55d37c80aa3/s2-wsf-aef-overture-atlas.py
(the pair and slider notebooks: https://github.com/kentstephen/s2-wsf-aef-overture-pair).
WSF and Overture buildings are out for now.

Run: uv run marimo run aef-s2-landcover-explorer.py --sandbox (it fills the window;
X or Esc gives the notebook back)
molab: https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-landcover-explorer.py

Attribution: "The AlphaEarth Foundations Satellite Embedding dataset is
produced by Google and Google DeepMind" (CC BY 4.0). ESA WorldCover 10 m
2021 v200 (c) ESA WorldCover project, contains modified Copernicus Sentinel
data (2021) processed by the ESA WorldCover consortium (CC BY 4.0).
Sentinel-2 yearly mosaics by Earth Genome (CC BY 4.0). Photon (komoot) over
OpenStreetMap data (ODbL). Overture Maps divisions for place names (ODbL).
Basemap by Carto.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import asyncio
    import itertools
    import json
    import math
    import os
    import re
    import tempfile
    import time
    import traceback
    import urllib.parse
    import urllib.request
    import zlib

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import xarray as xr
    import zarr
    import duckdb
    import marimo as mo
    import anywidget
    import traitlets

    from obstore.store import S3Store
    from zarr.storage import ObjectStore
    from async_geotiff import GeoTIFF, Window
    from datafusion import udf
    from xarray_sql import XarrayContext
    from h3ronpy import change_resolution
    from h3ronpy.vector import coordinates_to_cells
    from pyproj import Transformer

    import io
    from PIL import Image

    # CPU work (the DataFusion folds, tile compositing and PNG encoding, the
    # footprint rasterizing) leaves the event loop for ONE small pool, so the
    # network reads (asyncio, US East to the us-west-2 buckets) keep flowing
    # while it runs and a small machine (molab) is not flooded: at most
    # CPU_WORKERS such jobs at once, the rest queue
    from concurrent.futures import ThreadPoolExecutor

    CPU_WORKERS = max(2, min(4, (os.cpu_count() or 2) - 1))
    _cpu_pool = ThreadPoolExecutor(CPU_WORKERS, thread_name_prefix="cpu")

    async def cpu(fn, *args):
        """fn(*args) on the CPU pool, awaited."""
        return await asyncio.get_running_loop().run_in_executor(_cpu_pool, lambda: fn(*args))

    return (
        GeoTIFF,
        Image,
        ObjectStore,
        S3Store,
        Transformer,
        Window,
        XarrayContext,
        anywidget,
        asyncio,
        change_resolution,
        coordinates_to_cells,
        cpu,
        duckdb,
        io,
        itertools,
        json,
        math,
        mo,
        np,
        os,
        pa,
        pq,
        re,
        tempfile,
        time,
        traceback,
        traitlets,
        udf,
        urllib,
        xr,
        zarr,
        zlib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Where the ground changed

    AlphaEarth hexagons in viridis show where the ground changed over the
    year window. **Hold space** to see the Sentinel-2 imagery instead;
    **scroll** while holding to change its year; let go for the hexagons
    again. **Click** a hexagon for its account and its land cover (gold
    outline); click it again to clear it. **Click on the imagery** (space
    held) to pick the cell there: the card adds its H3 string and lat, long,
    each with a copy button. Below zoom 9 the map is ESA WorldCover 2021
    land cover; the hexagons start at zoom 9.

    | Key | Does |
    | --- | --- |
    | `space` (hold) | the Sentinel-2 imagery instead of the hexagons; the map still drags |
    | scroll, space held | the imagery year |
    | `[` `]` | the imagery year, back and forward |
    | `;` `'` | the imagery darker, brighter |
    | `S` | AEF Change: how much it changed |
    | `D` | AEF Change Year: the year of the biggest change |
    | `-` `=` | the first year read, earlier, later |
    | `_` `+` | the last year read, earlier, later |
    | `L` | place names on the map, off and on |
    | `/` | search a place, or paste an H3 string (flies there, outlined in blue) |
    | `X` | fill the window, and back |
    | `Esc` | close the about box, the menu, the card, then fill the window |

    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-landcover-explorer.py)
    <small>molab runs in the same region as the data and is the faster place to open this notebook.
    Locally: `uv run marimo run aef-s2-landcover-explorer.py --sandbox`</small>
    """)
    return


@app.cell
def _(os, tempfile):
    # ---- constants ----------------------------------------------------------
    # The imagery is Earth Genome's yearly mosaic, 2022 to 2025; AlphaEarth
    # runs 2017 to 2025. Not all nine years by default (Stephen, 2026-09-25:
    # "we don't need to aggregate default from 2017 to 2025 because that's a
    # lot"): the window opens at 2021, the year before the imagery starts,
    # so every imagery year has a step into it and can be a change year
    # ("the scroll should include all years including 22"). Widen it with
    # the window control.
    S2_YEARS = (2022, 2023, 2024, 2025)
    AEF_YEARS_ALL = tuple(range(2017, 2026))
    AEF_FROM0, AEF_TO0 = 2021, 2025
    # the first hold opens on the first imagery year, the scroll goes forward
    # from there (Stephen, 2026-09-25: "for the scroll it starts at 2022"),
    # and every later hold opens where the last one left off ("it should
    # persist where i leave off")
    S2_YEAR0 = 2022
    S2_SCALE0 = 1.0

    # the zoom -> H3 ladder, the settlement pair's: res 8 at zoom 9, 9 at
    # 10.4, 10 at 11.8, 11 at 13.2, 12 from 14.6
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 6
    MIN_RES, MAX_RES = 5, 12
    CELL_BUDGET = 300_000
    MOSAIC_MIN_RES = 11
    AEF_LEVEL_FOR_RES = {5: 7, 6: 7, 7: 5, 8: 4, 9: 3, 10: 1}
    AEF_MAX_FILES = 2500

    S2_STAC = "https://stac.earthgenome.org/search"
    S2_COLLECTION = "sentinel2-yearly-mosaics"
    # where the yearly mosaic is nodata, the same year's temporal mosaic fills
    # the hole pixel by pixel (2022 and 2023 only on this STAC)
    S2_FILL_COLLECTION = "sentinel2-temporal-mosaics"
    # the mosaic pyramid ends at z9 (L5, 306 m); z7-8 are rendered from L5 by
    # decimation (slow, so no lower)
    S2_TILE_MIN_Z, S2_PYRAMID_Z, S2_TCI_MAX_Z = 7, 9, 14

    # every S3 read to us-west-2: a short timeout so a stalled request is
    # retried instead of waited on (US East Coast, 2026-09-24)
    S3_OPTS = {"timeout": "6s", "connect_timeout": "3s"}

    AEF_PREFIX = "tge-labs/aef-mosaic"
    AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
    AEF_NODATA = -128
    AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "aef-lcms")

    # ---- ESA WorldCover 2021 (v200), 10 m, straight from ESA's bucket -------
    # (Stephen, 2026-09-25: "let's use ESA directly"). One COG per 3 x 3
    # degree tile, EPSG:4326, 36000 px a side, six overviews, named by its
    # south-west corner (N30E114). 2021 only: ESA says the 2020 and 2021 maps
    # were made with different algorithms and should not be compared for
    # change, so the land cover here describes the ground, it does not date
    # anything.
    WC_BUCKET, WC_REGION = "esa-worldcover", "eu-central-1"
    WC_PREFIX = "v200/2021/map"
    WC_S3_OPTS = {"timeout": "10s", "connect_timeout": "5s"}
    WC_MAX_TILES = 16
    WC_CLASSES = (
        (10, "tree cover"), (20, "shrubland"), (30, "grassland"), (40, "cropland"),
        (50, "built-up"), (60, "bare or sparse vegetation"), (70, "snow and ice"),
        (80, "permanent water"), (90, "herbaceous wetland"), (95, "mangroves"), (100, "moss and lichen"),
    )
    # zoomed out, below HEX_ZOOM, the map is WorldCover itself, as tiles read
    # from the same COGs (Stephen, 2026-09-25: "esa to seven then h3
    # handoff"). Not ESA's palette: it paints built-up red beside greens.
    # Built-up is the darkest (deep violet), vegetation in greens by
    # lightness, cropland gold, water blue; no red anywhere.
    WC_TILE_MIN_Z = 4
    WC_TILE_COLORS = {
        10: "2d6a3e", 20: "8fae5a", 30: "cfd99a", 40: "e8c547", 50: "3d2b7a", 60: "d9cbb5",
        70: "f4f6f8", 80: "3a7dc9", 90: "6bb8b0", 95: "2f7f6f", 100: "c9d6c0",
    }

    # the place under a click: the Overture divisions PMTiles answer at once in
    # the browser (locality, county, region); then the whole ladder, locality
    # up to country with each country's own word for the level (local_type),
    # from Overture's divisions GeoParquet as Fused partitions it on Source
    # Cooperative (the pair notebook's lookup: 7 s cold, 1 to 3 s after)
    OV_DIV_PM = "https://overturemaps-extras-us-west-2.s3.us-west-2.amazonaws.com/tiles/2026-08-19.0/divisions.pmtiles"
    ADMIN_PQ = "s3://us-west-2.opendata.source.coop/fused/overture/2026-05-20-0/theme=divisions"

    VIEW_W, VIEW_H = 700, 780
    PAD = 1.3
    SETTLE = 0.35
    # hexagons from zoom 9, WorldCover below (Stephen, 2026-09-25: the
    # hexagons as tiles "making my computer hum", so a min zoom for zoomed
    # out; zoomed in stays)
    HEX_ZOOM = 9.0
    LABELS_SLOT = "watername_ocean"
    RASTER_TILE = 256
    HOME = {"longitude": 114.29, "latitude": 30.58, "zoom": 7.2}  # Wuhan, the pair notebook's start

    # how long a still press takes to become a hold, and how far the pointer
    # may drift before it counts as a pan instead
    HOLD_MS, HOLD_SLOP_PX = 200, 5

    # the hexagons reach the browser as tiles of cell numbers, not polygons
    # (Stephen, 2026-09-25: "send the hexagons as imagery"): each 256 px map
    # tile is drawn at HEX_TILE_PX a side, every pixel the row of the hexagon
    # it falls in, colored in the browser
    HEX_TILE_PX = 512
    # the zoom ladder below picks the READ res (which AlphaEarth overview is
    # read). With the hexagons drawn as an image their count no longer costs
    # the browser, so two knobs, same download:
    # HEX_UP: the hexagons drawn are this many levels finer than the read res
    #   (1 is about one hexagon per pixel of the read; past that, empty cells)
    # CARRY_RES: the fold runs this many levels finer than the hexagons drawn,
    #   and each hexagon shows its most-changed finer cell ("carry the peak"),
    #   so a small change is not averaged away zoomed out
    # (0, 1): hexagons at the read res, each its brightest patch
    # (1, 0): hexagons about a pixel of the read each, nothing carried
    HEX_UP = 0
    CARRY_RES = 1

    # the fill fades with how much the cell changed: quiet ground faint
    ALPHA_FILL = 235
    ALPHA_QUIET = 45
    VIRIDIS = "440154470d6048186a482374472e7c4538824241863e4a893a548c365d8d32658e2e6d8e2b758e287d8e25848e228c8d1f948c1e9c8920a38625ab822eb37c3aba7648c16e58c7656ccd5a7fd34e93d741a8db34c0df25d5e21aeae51afde725"
    return (
        ADMIN_PQ,
        AEF_FROM0,
        AEF_INDEX_URL,
        AEF_LEVEL_FOR_RES,
        AEF_MAX_FILES,
        AEF_NODATA,
        AEF_PREFIX,
        AEF_RES,
        AEF_TO0,
        AEF_X0,
        AEF_Y0,
        AEF_YEARS_ALL,
        ALPHA_FILL,
        ALPHA_QUIET,
        BASE_RES,
        CACHE_DIR,
        CARRY_RES,
        CELL_BUDGET,
        HEX_TILE_PX,
        HEX_UP,
        HEX_ZOOM,
        HOLD_MS,
        HOLD_SLOP_PX,
        HOME,
        LABELS_SLOT,
        MAX_RES,
        MIN_RES,
        MOSAIC_MIN_RES,
        OV_DIV_PM,
        PAD,
        PER_RES,
        RASTER_TILE,
        S2_COLLECTION,
        S2_FILL_COLLECTION,
        S2_PYRAMID_Z,
        S2_SCALE0,
        S2_STAC,
        S2_TCI_MAX_Z,
        S2_TILE_MIN_Z,
        S2_YEAR0,
        S2_YEARS,
        S3_OPTS,
        SETTLE,
        VIEW_H,
        VIEW_W,
        VIRIDIS,
        WC_BUCKET,
        WC_CLASSES,
        WC_MAX_TILES,
        WC_PREFIX,
        WC_REGION,
        WC_S3_OPTS,
        WC_TILE_COLORS,
        WC_TILE_MIN_Z,
        ZOOM0,
    )


@app.cell
def _(
    BASE_RES,
    CELL_BUDGET,
    MAX_RES,
    MIN_RES,
    PAD,
    PER_RES,
    VIEW_H,
    VIEW_W,
    ZOOM0,
    math,
):
    # ---- the camera -> box and res --------------------------------------------
    CELL_KM2 = {5: 252.9, 6: 36.13, 7: 5.161, 8: 0.7373, 9: 0.1053, 10: 0.01505, 11: 0.00215, 12: 0.000307}

    def _lat_to_y(lat):
        r = math.radians(lat)
        return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2

    def _y_to_lat(y):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))

    def view_to_bbox(vs):
        """The flat camera footprint (W, S, E, N) of ONE pane; the widget reports
        the pane's canvas size (`w`, `h`) with every move."""
        world = 512 * (2 ** vs["zoom"])
        w, h = vs.get("w") or VIEW_W, vs.get("h") or VIEW_H
        half_lon = 360.0 * w / world / 2
        yc, half_y = _lat_to_y(vs["latitude"]), h / world / 2
        return (
            vs["longitude"] - half_lon,
            _y_to_lat(yc + half_y),
            vs["longitude"] + half_lon,
            _y_to_lat(yc - half_y),
        )

    def pad_box(b, f=PAD):
        dx, dy = (b[2] - b[0]) * (f - 1) / 2, (b[3] - b[1]) * (f - 1) / 2
        return (max(-179.9, b[0] - dx), max(-85.0, b[1] - dy), min(179.9, b[2] + dx), min(85.0, b[3] + dy))

    def box_km2(b):
        w = (b[2] - b[0]) * 111.32 * math.cos(math.radians((b[1] + b[3]) / 2))
        return abs(w * (b[3] - b[1]) * 110.57)

    def res_for_view(vs, box):
        r = max(MIN_RES, min(MAX_RES, BASE_RES + math.floor((vs["zoom"] - ZOOM0) / PER_RES)))
        while r > MIN_RES and box_km2(box) / CELL_KM2[r] > CELL_BUDGET:
            r -= 1
        return r

    def contains(outer, inner):
        return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]

    return CELL_KM2, contains, pad_box, res_for_view, view_to_bbox


@app.cell
def _(XarrayContext, coordinates_to_cells, pa, udf):
    # THE FOLD IS THE H3 UDF INSIDE DATAFUSION (repo rule). One context, every fold.
    ctx = XarrayContext()
    ctx.register_udf(
        udf(
            lambda la, lo, r: pa.array(coordinates_to_cells(la.to_numpy(), lo.to_numpy(), r[0].as_py())),
            [pa.float64(), pa.float64(), pa.int32()],
            pa.uint64(),
            "stable",
            name="h3_latlng_to_cell",
        )
    )
    return (ctx,)



@app.cell
def _(
    AEF_INDEX_URL,
    AEF_LEVEL_FOR_RES,
    AEF_MAX_FILES,
    AEF_NODATA,
    AEF_PREFIX,
    AEF_RES,
    AEF_X0,
    AEF_Y0,
    AEF_YEARS_ALL,
    CACHE_DIR,
    GeoTIFF,
    MOSAIC_MIN_RES,
    ObjectStore,
    S3Store,
    S3_OPTS,
    Transformer,
    Window,
    asyncio,
    cpu,
    ctx,
    duckdb,
    itertools,
    np,
    os,
    pq,
    time,
    xr,
    zarr,
):
    # ---- AlphaEarth: the COG overviews (mosaic past res 10), one fold per year --
    # `aef_fold(box, res, year)` for any year in AEF_YEARS_ALL (2017..2025, the
    # whole run; the window control picks from them); each year has its own COG index
    # slice (cached as parquet under tmp) and its own mosaic time index.
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True, client_options=S3_OPTS)
    _mstore = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True, prefix=AEF_PREFIX, client_options=S3_OPTS)
    _ds = xr.open_zarr(ObjectStore(_mstore, read_only=True), chunks=None, consolidated=False)
    _ti = {y: int(np.where(_ds.time.values == y)[0][0]) for y in AEF_YEARS_ALL}
    # The mosaic is sharded (4096 px shards of 256 px chunks, 64 bands, int8):
    # one read of a window fetches its chunks ONE AFTER ANOTHER, so a 9 km
    # view took 31 s a year from the US East Coast (1.2 MB/s, 38 MB; zarr's
    # async.concurrency made no difference). The window is read instead as
    # its chunk-aligned blocks, all at once, through zarr's async API on this
    # loop: 2.8 s for the same year (measured 2026-09-24, Wuhan, 16 blocks)
    _memb = zarr.open_group(ObjectStore(_mstore, read_only=True), mode="r")["embeddings"]
    _mb = int(_memb.chunks[-1])
    _memb = _memb._async_array
    _msem = asyncio.Semaphore(48)

    async def _mosaic(ti, y0, y1, x0, x1):
        """The mosaic's (64, y1 - y0, x1 - x0) int8 window for time index ti."""
        out = np.empty((64, y1 - y0, x1 - x0), np.int8)

        async def one(r0, r1, c0, c1):
            async with _msem:
                b = await _memb.getitem((ti, slice(None), slice(r0, r1), slice(c0, c1)))
            out[:, r0 - y0:r1 - y0, c0 - x0:c1 - x0] = b

        await asyncio.gather(*(
            one(max(r, y0), min(r + _mb, y1), max(c, x0), min(c + _mb, x1))
            for r in range(y0 // _mb * _mb, y1, _mb) for c in range(x0 // _mb * _mb, x1, _mb)
        ))
        return out

    os.makedirs(CACHE_DIR, exist_ok=True)
    _IDX, _PATHS, _CRS = {}, {}, {}
    for _y in AEF_YEARS_ALL:
        _idx_path = os.path.join(CACHE_DIR, f"aef_index_{_y}_world.parquet")
        if not os.path.exists(_idx_path):
            _c = duckdb.connect()
            _c.execute("INSTALL httpfs; LOAD httpfs")
            _t = _c.execute(f"""
                SELECT path, crs, utm_west, utm_south, utm_east, utm_north,
                       wgs84_west, wgs84_south, wgs84_east, wgs84_north
                FROM read_parquet('{AEF_INDEX_URL}')
                WHERE year = {_y}
            """).arrow().read_all()
            pq.write_table(_t, _idx_path)
            _c.close()
        _tab = pq.read_table(_idx_path)
        _IDX[_y] = {k: _tab[k].to_numpy() for k in _tab.column_names if k not in ("path", "crs")}
        _PATHS[_y] = _tab["path"].to_pylist()
        _CRS[_y] = _tab["crs"].to_pylist()

    _open = {}
    _sem = asyncio.Semaphore(64)
    _tf_fwd, _tf_inv = {}, {}

    def _tf(crs):
        if crs not in _tf_fwd:
            _tf_fwd[crs] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            _tf_inv[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        return _tf_fwd[crs], _tf_inv[crs]

    async def _get(path):
        rel = path.split("source.coop/")[1]
        if rel not in _open:
            async with _sem:
                _open[rel] = await GeoTIFF.open(rel, store=_store)
        return _open[rel]

    async def _read_cog(year, i, li, box):
        """One file's overview window over the box: (int8 (64, h, w), lon, lat)
        or None. Through the file's affine (these COGs are stored south-up)."""
        g = await _get(_PATHS[year][i])
        ov = g.overviews[li]
        H, W = ov.shape
        t = g.transform
        sx, sy = t.a * (g.width / W), t.e * (g.height / H)
        fwd, inv = _tf(_CRS[year][i])
        W_, S_, E_, N_ = box
        lons = np.concatenate([np.linspace(W_, E_, 5), np.full(5, E_), np.linspace(E_, W_, 5), np.full(5, W_)])
        lats = np.concatenate([np.full(5, N_), np.linspace(N_, S_, 5), np.full(5, S_), np.linspace(S_, N_, 5)])
        ux, uy = fwd.transform(lons, lats)
        cc = (np.asarray(ux) - t.c) / sx
        rr = (np.asarray(uy) - t.f) / sy
        c0 = max(0, int(np.floor(np.nanmin(cc))))
        c1 = min(W, int(np.ceil(np.nanmax(cc))))
        r0 = max(0, int(np.floor(np.nanmin(rr))))
        r1 = min(H, int(np.ceil(np.nanmax(rr))))
        if c1 <= c0 or r1 <= r0:
            return None
        async with _sem:
            ra = await ov.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))

        def _place():
            a = np.asarray(np.ma.filled(ra.as_masked(), AEF_NODATA)).reshape(64, r1 - r0, c1 - c0)
            xs = t.c + (np.arange(c0, c1) + 0.5) * sx
            ys = t.f + (np.arange(r0, r1) + 0.5) * sy
            X, Y = np.meshgrid(xs, ys)
            lon, lat = inv.transform(X, Y)
            return a, lon, lat

        return await cpu(_place)

    _DEQ = ", ".join(f"avg(signum(e{i:02d}) * power(e{i:02d} / 127.5, 2)) AS e{i:02d}" for i in range(64))
    _seq = itertools.count()  # a table name per fold: the years fold side by side on the CPU pool

    def _fold_rows_sync(res, box, cols, lat, lon):
        W_, S_, E_, N_ = box
        name = f"aef_{next(_seq)}"
        ds1 = xr.Dataset(
            {f"e{i:02d}": (("i",), cols[i]) for i in range(64)} | {"lat": (("i",), lat), "lon": (("i",), lon)},
            coords={"i": np.arange(lat.size)},
        )
        ctx.from_dataset(name, ds1, chunks={"i": 262_144})
        try:
            return ctx.sql(f"""
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, count(*) AS naef, {_DEQ}
                FROM {name}
                WHERE e00 != {AEF_NODATA}
                  AND lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                GROUP BY cell
            """).to_arrow_table()
        finally:
            ctx.deregister_table(name)

    async def aef_window(box, year):
        """The mosaic's native 10 m window under the box for one year:
        (int8 (64, h, w), lon0, lat0 of the north-west corner, pixel) or None."""
        W_, S_, E_, N_ = box
        x0, x1 = int((W_ - AEF_X0) / AEF_RES), int((E_ - AEF_X0) / AEF_RES)
        y0, y1 = int((AEF_Y0 - N_) / AEF_RES), int((AEF_Y0 - S_) / AEF_RES)
        if x1 <= x0 or y1 <= y0:
            return None
        emb = await _mosaic(_ti[year], y0, y1, x0, x1)
        return emb, AEF_X0 + x0 * AEF_RES, AEF_Y0 - y0 * AEF_RES, AEF_RES

    async def aef_fold(box, res, year, read_res=None):
        """Mean AlphaEarth vector per res cell over the box for one year, from
        the source that suits read_res (default res): a finer res than the
        read's own gives cells of about a pixel each (the carried peak).
        Returns (arrow table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        rr = res if read_res is None else read_res
        if rr >= MOSAIC_MIN_RES:
            x0, x1 = int((W_ - AEF_X0) / AEF_RES), int((E_ - AEF_X0) / AEF_RES)
            y0, y1 = int((AEF_Y0 - N_) / AEF_RES), int((AEF_Y0 - S_) / AEF_RES)
            emb = await _mosaic(_ti[year], y0, y1, x0, x1)
            lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
            lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
            t1 = time.time()

            def _mosaic():
                LON, LAT = np.meshgrid(lon, lat)
                return _fold_rows_sync(res, box, emb.reshape(64, -1), LAT.ravel(), LON.ravel())

            out = await cpu(_mosaic)
            return out, f"AEF {year} mosaic {t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        li = AEF_LEVEL_FOR_RES[rr]
        ix = _IDX[year]
        hit = np.where(
            (ix["wgs84_east"] > W_) & (ix["wgs84_west"] < E_) & (ix["wgs84_north"] > S_) & (ix["wgs84_south"] < N_)
        )[0]
        if len(hit) == 0:
            return None, f"AEF {year}: no COG tiles under the view"
        if len(hit) > AEF_MAX_FILES:
            return None, f"AEF {year}: {len(hit):,} tiles under the view; zoom in"
        parts = await asyncio.gather(*(_read_cog(year, int(i), li, box) for i in hit))
        parts = [p for p in parts if p is not None]
        if not parts:
            return None, f"AEF {year}: nothing read"
        t1 = time.time()

        def _cogs():
            cols = np.concatenate([p[0].reshape(64, -1) for p in parts], axis=1)
            lon = np.concatenate([p[1].ravel() for p in parts])
            lat = np.concatenate([p[2].ravel() for p in parts])
            return _fold_rows_sync(res, box, cols, lat, lon), cols.shape[1]

        out, npx = await cpu(_cogs)
        return out, (
            f"AEF {year} ov{li} ({10 * 2 ** (li + 1)} m) {len(parts)} files {npx / 1e6:.2f} Mpx "
            f"{t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        )

    return aef_fold, aef_window


@app.cell
def _(
    GeoTIFF,
    Image,
    RASTER_TILE,
    S2_COLLECTION,
    S2_FILL_COLLECTION,
    S2_PYRAMID_Z,
    S2_SCALE0,
    S2_STAC,
    S2_TCI_MAX_Z,
    S2_TILE_MIN_Z,
    S3Store,
    S3_OPTS,
    Window,
    asyncio,
    cpu,
    io,
    json,
    math,
    np,
    time,
    urllib,
):
    # ---- Sentinel-2 TCI tiles, by YEAR: the left pane ---------------------------
    # STAC once per (year, z9 ancestor tile), every footprint under the tile
    # composited in numpy (black = nodata -> alpha 0; first footprint to paint a
    # pixel wins), one PNG. The year lives in the item id
    # (`10SFJ_2024-01-01_2025-01-01`); the STAC datetime filter does not
    # constrain these items, so it is enforced on the id. The yearly footprints
    # come first, then the same year's S2_FILL_COLLECTION footprints (ids
    # suffixed `#fill`): first-to-paint-wins is the backfill.
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True, client_options=S3_OPTS)
    _R = 6378137.0
    _items = {}  # item id -> {tci: path, bbox}
    _boxes = {}  # (year, rounded box) -> item ids
    _open = {}
    _sem = asyncio.Semaphore(32)
    _png = {}  # (year, z, x, y, scale) -> PNG bytes or None
    _arr = {}  # (year, z, x, y) -> the composited RGBA tile before the gain, or None
    _gain = {"v": float(S2_SCALE0)}  # the strip's `gamma` (Stephen, 2026-09-24: "use gamma instead of the scale")
    _LUT = {}  # gamma -> the 256-entry curve
    _tstat = {"served": 0, "blank": 0, "ms": 0.0}
    _fill = {}  # year -> [pixels painted by the fill collection, pixels painted]

    def _png_of(out, g):
        """The gamma (the header's `gamma`) on the composited bytes, then PNG:
        v -> 255 (v / 255) ** (1 / gamma), a lift of the midtones that keeps
        the bright end unclipped (a gain clipped it). Pure: runs on the pool."""
        if g not in _LUT:
            _LUT[g] = (255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** (1.0 / g)).round().astype(np.uint8)
        rgba = out if g == 1.0 else np.concatenate([_LUT[g][out[..., :3]], out[..., 3:]], axis=2)
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    async def _encode(key, out):
        _png[key] = await cpu(_png_of, out, key[-1])
        if len(_png) > 6000:
            _png.pop(next(iter(_png)))
        return _png[key]

    def _stac(box):
        body = json.dumps(
            {"collections": [S2_COLLECTION, S2_FILL_COLLECTION], "bbox": list(box), "limit": 200}
        ).encode()
        req = urllib.request.Request(S2_STAC, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["features"]

    async def _s2_items(box, year):
        key = (year, tuple(round(v, 2) for v in box))
        if key not in _boxes:
            loop = asyncio.get_running_loop()
            both = await loop.run_in_executor(None, _stac, box)
            feats = [f for f in both if f.get("collection") != S2_FILL_COLLECTION]
            ids, fill_ids = [], []
            for f in both:
                if not f["id"].endswith(f"{year}-01-01_{year + 1}-01-01"):
                    continue
                if f.get("collection") == S2_FILL_COLLECTION:
                    iid = f["id"] + "#fill"
                    _items[iid] = {"tci": f["assets"]["TCI"]["href"].split("source.coop/")[1], "bbox": f.get("bbox"), "fill": True}
                    fill_ids.append(iid)
                    continue
                _items[f["id"]] = {"tci": f["assets"]["TCI"]["href"].split("source.coop/")[1], "bbox": f.get("bbox")}
                ids.append(f["id"])
            if not ids and feats:
                # the STAC lags the bucket (2025 is there for every tile round
                # Dixie, uploaded 2026-01-31, and the search does not know it):
                # the same MGRS tile's path with the year swapped, the sibling's
                # bbox; a tile that is not there reads as empty, not an error
                seen = set()
                for f in feats:
                    tile = f["id"].split("_")[0]
                    if tile in seen:
                        continue
                    seen.add(tile)
                    iid = f"{tile}_{year}-01-01_{year + 1}-01-01"
                    base = f["assets"]["TCI"]["href"].split("source.coop/")[1].rsplit("/", 2)[0]
                    _items[iid] = {"tci": f"{base}/{iid}/TCI.tif", "bbox": f.get("bbox")}
                    ids.append(iid)
            _boxes[key] = ids + fill_ids  # yearly first: the fill only paints what they left
        return _boxes[key]

    async def _get(rel):
        if rel not in _open:
            async with _sem:
                try:
                    _open[rel] = await GeoTIFF.open(rel, store=_store)
                except Exception:
                    _open[rel] = None  # not in the bucket (a synthesized year): empty
        return _open[rel]

    def _tile_ll(z, x, y):
        n = 2 ** z
        lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
        return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)

    async def _items_for_tile(z, x, y, year):
        # STAC per z9 ancestor tile from z9 up; below it, per the tile itself
        d = max(0, z - S2_PYRAMID_Z)
        ids = await _s2_items(_tile_ll(z - d, x >> d, y >> d), year)
        W_, S_, E_, N_ = _tile_ll(z, x, y)
        out = []
        for i in ids:
            b = _items[i].get("bbox")
            if not b or (b[0] < E_ and b[2] > W_ and b[1] < N_ and b[3] > S_):
                out.append(i)
        return out

    async def s2_tile_png(z, x, y, year):
        """PNG bytes for Web Mercator tile (z, x, y) of the year's TCI mosaic, or
        None (below S2_TILE_MIN_Z, or no footprint under the tile)."""
        key = (year, z, x, y, _gain["v"])
        if key in _png:
            return _png[key]
        if (year, z, x, y) in _arr:
            # composited already at another scale: re-encode, no read
            out = _arr[(year, z, x, y)]
            return (await _encode(key, out)) if out is not None else None
        if z < S2_TILE_MIN_Z or z > S2_TCI_MAX_Z:
            _tstat["blank"] += 1
            return None
        ids = await _items_for_tile(z, x, y, year)
        if not ids:
            _tstat["blank"] += 1
            return None
        t0 = time.time()
        T = RASTER_TILE
        n = 2 ** z
        world = 2 * math.pi * _R
        tpx = world / (n * T)
        tx0, ty1 = -world / 2 + x * world / n, world / 2 - y * world / n
        xs = tx0 + (np.arange(T) + 0.5) * tpx
        ys = ty1 - (np.arange(T) + 0.5) * tpx
        li = min(S2_TCI_MAX_Z - z, S2_TCI_MAX_Z - S2_PYRAMID_Z)  # L5 (306 m) below z9: decimated

        async def _read(iid):
            """One footprint's window under the tile: (ra, c0, r0, h, w, px, L, Tt) or None."""
            g = await _get(_items[iid]["tci"])
            if g is None:
                return None
            lv = [g, *g.overviews][li]
            L, _B, R_, Tt = g.bounds
            H, W = lv.shape
            px = (R_ - L) / W
            c0, c1 = max(0, int(math.floor((tx0 - L) / px))), min(W, int(math.ceil((tx0 + T * tpx - L) / px)))
            r0, r1 = max(0, int(math.floor((Tt - ty1) / px))), min(H, int(math.ceil((Tt - (ty1 - T * tpx)) / px)))
            if c1 <= c0 or r1 <= r0:
                return None
            async with _sem:
                ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
            return ra, c0, r0, r1 - r0, c1 - c0, px, L, Tt

        # all the footprints at once (the round trips overlap), painted in
        # their order after (first to paint a pixel wins, yearly before fill)
        reads = await asyncio.gather(*(_read(i) for i in ids))

        def _composite():
            out = np.zeros((T, T, 4), np.uint8)
            painted = []
            for rd in reads:
                if rd is None:
                    painted.append(0)
                    continue
                ra, c0, r0, h, w, px, L, Tt = rd
                a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(-1, h, w)[:3]
                cols = np.floor((xs - (L + c0 * px)) / px).astype(np.int64)
                rows = np.floor(((Tt - r0 * px) - ys) / px).astype(np.int64)
                okc, okr = (cols >= 0) & (cols < w), (rows >= 0) & (rows < h)
                rgb = a[:, np.clip(rows, 0, h - 1)[:, None], np.clip(cols, 0, w - 1)[None, :]].transpose(1, 2, 0)
                valid = okr[:, None] & okc[None, :] & (rgb.sum(2) > 0) & (out[..., 3] == 0)
                out[valid, :3] = rgb[valid]
                out[valid, 3] = 255
                painted.append(int(valid.sum()))
            return out, painted

        out, painted = await cpu(_composite)
        fy = _fill.setdefault(year, [0, 0])
        for iid, n_new in zip(ids, painted):
            fy[1] += n_new
            if _items[iid].get("fill"):
                fy[0] += n_new
        if not out[..., 3].any():
            _tstat["blank"] += 1
            _png[key] = None
            _arr[(year, z, x, y)] = None
            return None
        _arr[(year, z, x, y)] = out
        if len(_arr) > 2000:
            _arr.pop(next(iter(_arr)))
        png = await _encode(key, out)
        _tstat["served"] += 1
        _tstat["ms"] += 1000 * (time.time() - t0)
        return png

    def s2_set_scale(v):
        """The header's `gamma`: the curve the next S2 tiles are encoded with.
        Returns True when it changed (the caller then re-asks deck for the tiles)."""
        v = float(min(4.0, max(0.1, v)))
        if v == _gain["v"]:
            return False
        _gain["v"] = v
        return True

    def s2_raster_stats():
        """The tile counters, plus `fill`: for each year whose served tiles took
        any pixels from S2_FILL_COLLECTION, the share of painted pixels that did
        (over every tile served so far, not the view)."""
        fill = {y: f / p for y, (f, p) in _fill.items() if f and p}
        return dict(_tstat, cached=len(_png), scale=_gain["v"], fill=fill)

    return s2_raster_stats, s2_set_scale, s2_tile_png


@app.cell
def _(duckdb):
    # ---- DuckDB: the frame's join and the tables under the map --------------
    con = duckdb.connect()
    return (con,)


@app.cell
def _(ADMIN_PQ, HOME, duckdb):
    # ---- the place under a click: one point query against fused/overture ------
    # (the pair notebook's, as it was: Stephen, 2026-09-25, "it works well in
    # the pair notebook")
    # Overture's divisions theme as Fused geo-partitions it on Source
    # Cooperative, 79 GeoParquet files per type, each row with a bbox struct.
    # division_area says which polygons hold the point: country, region,
    # county, localadmin, locality, every level Overture draws, anywhere.
    # division, joined on the ids, adds local_type, the country's own word for
    # the level (city, town, village, prefecture, governorate, state). DuckDB
    # reads the footers, keeps the row groups whose bbox stats can hold the
    # point, and runs ST_Contains on what is left. Its own connection, a
    # cursor per call so a click and the warm-up can overlap, the object cache
    # on so the footers are read once: 7 s cold, 1 to 3 s after.
    import threading as _th

    _dv = {"con": None, "err": None}
    _lock = _th.Lock()
    _AREA = f"{ADMIN_PQ}/type=division_area/*.parquet"
    _DIV = f"{ADMIN_PQ}/type=division/*.parquet"
    _ORDER = {"locality": 0, "localadmin": 1, "county": 2, "region": 3, "country": 4}

    def _connect():
        with _lock:
            if _dv["con"] is None and _dv["err"] is None:
                try:
                    c = duckdb.connect()
                    for ext in ("spatial", "httpfs"):
                        try:
                            c.execute(f"LOAD {ext}")
                        except Exception:
                            c.execute(f"INSTALL {ext}; LOAD {ext}")
                    # an open bucket: the region and path-style URLs (the
                    # bucket name has dots in it), nothing to sign with.
                    # GLOBAL, because a cursor is its own session and a plain
                    # SET would not reach it
                    c.execute("SET GLOBAL s3_region='us-west-2'; SET GLOBAL s3_url_style='path'; SET GLOBAL enable_object_cache=true")
                    _dv["con"] = c
                except Exception as e:
                    _dv["err"] = e
            return _dv["con"]

    _Q_AREA = (
        "SELECT subtype, names.primary, names.common['en'], country, division_id "
        f"FROM read_parquet('{_AREA}', hive_partitioning=0) "
        "WHERE bbox.xmin <= $x AND bbox.xmax >= $x AND bbox.ymin <= $y AND bbox.ymax >= $y "
        "AND class = 'land' AND ST_Contains(geometry, ST_Point($x, $y))"
    )
    # the country filter is what makes the join quick: the files are spatial,
    # so each row group carries a tight country range and most are skipped
    # unread (25 s cold at Wuhan against 165 s without it, 1 to 3 s warm)
    _Q_DIV = (
        "SELECT id, local_type['en'], population "
        f"FROM read_parquet('{_DIV}', hive_partitioning=0) "
        "WHERE country = $country AND list_contains($ids, id)"
    )

    def division_at(lon, lat):
        """The divisions holding the point, smallest first: a list of
        {subtype, name, name_en, local_type, population}, locality up to
        country, whichever Overture draws there. Raises on a failed read so
        the caller can say so."""
        c = _connect()
        if c is None:
            raise _dv["err"]
        cur = c.cursor()
        rows = cur.execute(_Q_AREA, {"x": float(lon), "y": float(lat)}).fetchall()
        seen, out = set(), []
        for sub, name, name_en, country, did in rows:
            if sub in seen:
                continue
            seen.add(sub)
            out.append({"subtype": sub, "name": name, "name_en": name_en, "local_type": None,
                        "population": None, "id": did, "country": country})
        out.sort(key=lambda d: _ORDER.get(d["subtype"], -1))
        ids = [d["id"] for d in out if d["id"]]
        country = next((d["country"] for d in out if d["country"]), None)
        if ids and country:
            try:
                extra = {i: (lt, pop) for i, lt, pop in cur.execute(_Q_DIV, {"country": country, "ids": ids}).fetchall()}
            except Exception:
                extra = {}
            for d in out:
                d["local_type"], d["population"] = extra.get(d["id"], (None, None))
        return out

    # the footers, read now rather than on the first click, off the main
    # thread: about 7 s for division_area and 25 s more for division
    def _warm():
        try:
            division_at(HOME["longitude"], HOME["latitude"])
        except Exception:
            pass

    _th.Thread(target=_warm, daemon=True).start()
    return (division_at,)





@app.cell
def _(
    AEF_LEVEL_FOR_RES,
    GeoTIFF,
    Image,
    MOSAIC_MIN_RES,
    RASTER_TILE,
    S3Store,
    WC_BUCKET,
    WC_CLASSES,
    WC_MAX_TILES,
    WC_PREFIX,
    WC_REGION,
    WC_S3_OPTS,
    WC_TILE_COLORS,
    WC_TILE_MIN_Z,
    Window,
    asyncio,
    cpu,
    ctx,
    io,
    itertools,
    math,
    np,
    time,
    xr,
):
    # ---- ESA WorldCover: the class shares per hexagon, one fold per (box, res) --
    # The same fold as AlphaEarth: every pixel's lon/lat and class through the
    # H3 UDF in DataFusion, a count per class per cell. Below MOSAIC_MIN_RES
    # it reads the overview whose pixel matches AlphaEarth's for that res
    # (overview i is 10 * 2^(i + 1) m; the file has six), from there the
    # native 10 m. The overviews are a sample of the classes, not a blend, so
    # a share is a count of sampled pixels.
    _store = S3Store(WC_BUCKET, region=WC_REGION, skip_signature=True, client_options=WC_S3_OPTS)
    _open = {}
    _sem = asyncio.Semaphore(32)
    _seq = itertools.count()
    WC_CODES = tuple(c for c, _ in WC_CLASSES)

    def _tile_name(lat, lon):
        """ESA's tile name for the 3 x 3 degree tile whose south-west corner is (lat, lon)."""
        return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"

    async def _get(name):
        if name not in _open:
            async with _sem:
                try:
                    _open[name] = await GeoTIFF.open(f"{WC_PREFIX}/ESA_WorldCover_10m_2021_v200_{name}_Map.tif", store=_store)
                except Exception as e:
                    # open ocean: ESA has no tile there (async-tiff wraps the 404)
                    if not (isinstance(e, FileNotFoundError) or "NotFound" in str(e) or "NoSuchKey" in str(e)):
                        raise
                    _open[name] = None
        return _open[name]

    async def _read(lat0, lon0, li, box):
        """One tile's window under the box: (uint8 classes (h, w), lon of the
        columns, lat of the rows) or None."""
        g = await _get(_tile_name(lat0, lon0))
        if g is None:
            return None
        lv = g if li < 0 else g.overviews[li]
        H, W = lv.shape
        px = 3.0 / W
        top = lat0 + 3
        W_, S_, E_, N_ = box
        c0, c1 = max(0, int(math.floor((W_ - lon0) / px))), min(W, int(math.ceil((E_ - lon0) / px)))
        r0, r1 = max(0, int(math.floor((top - N_) / px))), min(H, int(math.ceil((top - S_) / px)))
        if c1 <= c0 or r1 <= r0:
            return None
        async with _sem:
            ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
        a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(r1 - r0, c1 - c0)
        lon = lon0 + (np.arange(c0, c1) + 0.5) * px
        lat = top - (np.arange(r0, r1) + 0.5) * px
        return a, lon, lat

    _CNT = ", ".join(f"sum(CASE WHEN c = {code} THEN 1 ELSE 0 END) AS wc{code}" for code in WC_CODES)

    async def wc_fold(box, res):
        """Per res cell over the box: `nwc` (classified pixels sampled) and one
        count per class (`wc10` .. `wc100`). (table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        li = -1 if res >= MOSAIC_MIN_RES else min(5, AEF_LEVEL_FOR_RES[res])
        lats = range(int(math.floor(S_ / 3)) * 3, int(math.floor(N_ / 3)) * 3 + 1, 3)
        lons = range(int(math.floor(W_ / 3)) * 3, int(math.floor(E_ / 3)) * 3 + 1, 3)
        tiles = [(la, lo) for la in lats for lo in lons]
        if len(tiles) > WC_MAX_TILES:
            return None, f"land cover: {len(tiles)} tiles under the view; zoom in"
        parts = [p for p in await asyncio.gather(*(_read(la, lo, li, box) for la, lo in tiles)) if p is not None]
        if not parts:
            return None, "land cover: nothing under the view"
        t1 = time.time()

        def _run():
            name = f"wc_{next(_seq)}"
            cls = np.concatenate([p[0].ravel() for p in parts]).astype(np.int16)
            lon = np.concatenate([np.broadcast_to(p[1][None, :], p[0].shape).ravel() for p in parts])
            lat = np.concatenate([np.broadcast_to(p[2][:, None], p[0].shape).ravel() for p in parts])
            ctx.from_dataset(
                name,
                xr.Dataset({"c": (("i",), cls), "lat": (("i",), lat), "lon": (("i",), lon)}, coords={"i": np.arange(cls.size)}),
                chunks={"i": 262_144},
            )
            try:
                return ctx.sql(f"""
                    SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, count(*) AS nwc, {_CNT}
                    FROM {name}
                    WHERE c > 0 AND lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                    GROUP BY cell
                """).to_arrow_table()
            finally:
                ctx.deregister_table(name)

        out = await cpu(_run)
        lvl = "10 m" if li < 0 else f"ov{li} ({10 * 2 ** (li + 1)} m)"
        return out, f"land cover {lvl} {len(parts)} tiles read {t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"

    # ---- ESA WorldCover as map tiles, below HEX_ZOOM ------------------------------
    # One Web Mercator tile: every pixel's center looked up (nearest) in the
    # overview whose pixel is no bigger than the tile's, from each 3 degree
    # file under it, then colored with WC_TILE_COLORS. Kept in memory.
    _LUT = np.zeros((256, 4), np.uint8)
    for _code, _hx in WC_TILE_COLORS.items():
        _LUT[_code] = (int(_hx[0:2], 16), int(_hx[2:4], 16), int(_hx[4:6], 16), 225)
    _wc_png = {}

    async def wc_tile_png(z, x, y):
        """PNG bytes for Web Mercator tile (z, x, y) of WorldCover 2021, or None
        (below WC_TILE_MIN_Z, or no land under it)."""
        if (z, x, y) in _wc_png:
            return _wc_png[(z, x, y)]
        if z < WC_TILE_MIN_Z:
            return None
        T, n = RASTER_TILE, 2 ** z
        _lat = lambda v: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * v))))
        W_, E_ = -180 + x * 360 / n, -180 + (x + 1) * 360 / n
        N_, S_ = _lat(y / n), _lat((y + 1) / n)
        lon = W_ + (np.arange(T) + 0.5) * (E_ - W_) / T
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + (np.arange(T) + 0.5) / T) / n))))
        # native 10 m is 3 / 36000 degrees; overview i is 2^(i + 1) of that
        tpx = (E_ - W_) / T
        li = max(-1, min(5, int(math.floor(math.log2(tpx / (3 / 36000)))) - 1))
        S_c, N_c = max(S_, -60.0), min(N_, 84.0)
        if S_c >= N_c:
            return None
        lats = range(int(math.floor(S_c / 3)) * 3, int(math.floor(N_c / 3)) * 3 + 1, 3)
        lons = range(int(math.floor(W_ / 3)) * 3, int(math.floor(E_ / 3)) * 3 + 1, 3)
        parts = [p for p in await asyncio.gather(*(_read(la, lo, li, (W_, S_, E_, N_)) for la in lats for lo in lons)) if p is not None]
        if not parts:
            _wc_png[(z, x, y)] = None
            return None

        def _paint():
            out = np.zeros((T, T), np.uint8)
            for a, lonc, latr in parts:
                px = (lonc[1] - lonc[0]) if lonc.size > 1 else 3.0 / (36000 / 2 ** (li + 1))
                ci = np.floor((lon - (lonc[0] - px / 2)) / px).astype(np.int64)
                ri = np.floor(((latr[0] + px / 2) - lat) / px).astype(np.int64)
                cs, rs = np.nonzero((ci >= 0) & (ci < a.shape[1]))[0], np.nonzero((ri >= 0) & (ri < a.shape[0]))[0]
                if cs.size and rs.size:
                    sub = a[np.ix_(ri[rs], ci[cs])]
                    view = out[np.ix_(rs, cs)]
                    out[np.ix_(rs, cs)] = np.where(sub > 0, sub, view)
            if not out.any():
                return None
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(_LUT[out]), mode="RGBA").save(buf, format="PNG")
            return buf.getvalue()

        png = await cpu(_paint)
        _wc_png[(z, x, y)] = png
        while len(_wc_png) > 3000:
            _wc_png.pop(next(iter(_wc_png)))
        return png

    return WC_CODES, wc_fold, wc_tile_png


@app.cell
def _(WC_CLASSES, WC_CODES, change_resolution, con, np, pa):
    # ---- a FRAME: AlphaEarth over the window, the land cover beside it ----------
    # No quiet level this time (it came from WSF, which is out): the fill is
    # how much the fingerprint moved between the window's two ends, stretched
    # to this view's p2..p98. The change year is the year whose step stands
    # out most against THAT YEAR'S median step in view: the embeddings shift
    # as a whole between some years (measured over Lagos, 2026-09-25: the
    # 2024 to 2025 median step is 0.034, the others 0.012 to 0.022), so the
    # raw biggest step lands on 2025 almost everywhere. Each step is divided
    # by its year's median; the card shows those ratios against 1.
    #
    # CARRY THE PEAK: the years arrive folded at a finer res than the
    # hexagons (about a pixel of the read per finer cell). Steps, change and
    # change year are worked out per finer cell; each hexagon then takes the
    # finer cell that moved most, whole (its change, its year, its steps).
    # A hexagon reads "the most-changed patch in here", not "the average of
    # in here", so one changed site is not diluted by the quiet ground around it.
    _E = [f"e{i:02d}" for i in range(64)]
    _WC_NAME = dict(WC_CLASSES)

    def build_frame(aef_by_year, wc, y0, y1, res):
        years = [y for y in range(y0, y1 + 1) if aef_by_year.get(y) is not None]
        if len(years) < 2:
            return None
        for y in years:
            con.register(f"aef_{y}", aef_by_year[y])
        sel = ["cell"]
        for y in years:
            sel += [f"a{y}.{e} AS {e}_{y}" for e in _E]
        joins = [f"LEFT JOIN aef_{y} a{y} USING (cell)" for y in years[1:]]
        j = con.execute(f"SELECT {', '.join(sel)} FROM aef_{years[0]} a{years[0]} {' '.join(joins)} ORDER BY cell").arrow().read_all()
        nfine = j.num_rows

        def _V(y):
            V = np.stack([j[f"{e}_{y}"].to_numpy(zero_copy_only=False) for e in _E], axis=1).astype(np.float32)
            nrm = np.linalg.norm(V, axis=1)
            V = V / np.maximum(nrm, 1e-9)[:, None]
            V[~np.isfinite(nrm) | (nrm == 0)] = np.nan
            return V

        Vs = {y: _V(y) for y in years}
        step_years = list(zip(years[:-1], years[1:]))
        steps = np.stack([(1.0 - np.einsum("ij,ij->i", Vs[a], Vs[b])).astype(np.float32) for a, b in step_years], 0)
        disp = (1.0 - np.einsum("ij,ij->i", Vs[years[0]], Vs[years[-1]])).astype(np.float32)
        has = ~np.isnan(steps).all(0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(steps, axis=1)
        med = np.where(np.isfinite(med) & (med > 0), med, np.nan).astype(np.float32)
        rel = steps / med[:, None]
        k = np.argmax(np.where(np.isnan(rel), -np.inf, rel), 0)
        yb = np.array([b for _, b in step_years], np.int64)
        big = np.where(has, yb[k], -2).astype(np.int64)

        # the peak: per hexagon, its finer cell with the biggest change
        fine = j["cell"].to_numpy().astype(np.uint64)
        par = pa.array(change_resolution(fine, res)).to_numpy(zero_copy_only=False).astype(np.uint64)
        o = np.lexsort((-np.where(np.isnan(disp), -np.inf, disp), par))
        ps = par[o]
        first = np.r_[True, ps[1:] != ps[:-1]] if len(ps) else np.zeros(0, bool)
        pk = o[first]
        cellid = ps[first]
        n = len(cellid)
        nkids = np.diff(np.r_[np.flatnonzero(first), len(ps)]).astype(np.int32)
        disp, big, steps, rel = disp[pk], big[pk], steps[:, pk], rel[:, pk]

        scored = ~np.isnan(disp)
        if scored.sum() >= 2:
            lo, hi = (float(q) for q in np.percentile(disp[scored], [2, 98]))
            hi = hi if hi > lo else lo + 1e-6
        else:
            lo, hi = 0.0, 1.0
        level = np.where(scored, np.clip((np.nan_to_num(disp) - lo) / (hi - lo), 0, 1), np.nan).astype(np.float32)

        nwc = np.zeros(n)
        share = np.zeros((n, len(WC_CODES)))
        if wc is not None and wc.num_rows:
            wcell = wc["cell"].to_numpy().astype(np.uint64)
            ow = np.argsort(wcell)
            wcell = wcell[ow]
            pos = np.clip(np.searchsorted(wcell, cellid), 0, len(wcell) - 1)
            m = wcell[pos] == cellid
            wn = np.nan_to_num(wc["nwc"].to_numpy(zero_copy_only=False).astype(np.float64))[ow]
            nwc = np.where(m, wn[pos], 0.0)
            C = np.stack([np.nan_to_num(wc[f"wc{c}"].to_numpy(zero_copy_only=False).astype(np.float64))[ow][pos] for c in WC_CODES], 1)
            share = np.where(m[:, None], C, 0.0) / np.maximum(nwc, 1)[:, None]
        top = np.where(nwc > 0, share.argmax(1), -1)
        top_share = np.where(nwc > 0, share.max(1), 0.0)

        cells = pa.table({
            "cell": pa.array(cellid),
            "disp": pa.array(disp),
            "level": pa.array(level),
            "big_year": pa.array(big.astype(np.int16)),
            "finer_cells": pa.array(nkids),
            **{f"step_{b}": pa.array(steps[i]) for i, (_, b) in enumerate(step_years)},
            **{f"rel_{b}": pa.array(rel[i]) for i, (_, b) in enumerate(step_years)},
            "landcover": pa.array([_WC_NAME[WC_CODES[t]] if t >= 0 else None for t in top]),
            "landcover_share": pa.array(top_share.astype(np.float32)),
            **{f"wc_{_WC_NAME[c].replace(' ', '_')}": pa.array(share[:, i].astype(np.float32)) for i, c in enumerate(WC_CODES)},
        })
        return {
            "cells": cells, "cellid": cellid, "res": res, "disp": disp, "level": level, "big": big,
            "steps": steps, "rel": rel, "med": med, "step_years": step_years, "years": years, "y0": y0, "y1": y1,
            "shift_lo": lo, "shift_hi": hi, "share": share, "nwc": nwc, "top": top, "top_share": top_share,
            "score": f"AEF {years[0]} to {years[-1]}: {int(scored.sum()):,} of {n:,} cells scored, peak of {nfine:,} finer cells, median steps "
                     + ", ".join(f"{b} {m:.3f}" for (_, b), m in zip(step_years, med)),
        }

    return (build_frame,)


@app.cell
def _(anywidget, asyncio, time, traitlets):
    class ChangeMap(anywidget.AnyWidget):
        """The map: the AlphaEarth hexagons in viridis on a plain basemap; press
        and hold for the Sentinel-2 imagery (the hexagons go while you hold),
        scroll while holding to change its year; the year card at the top
        right with the view's changes by year and the clicked hexagon's account.

        Kernel -> browser: `cells` (uint64 LE) with `hattrs` (4 bytes per
        hexagon: the year of its biggest step, 0 none, else year - 2000; how
        much it moved 1..255, 0 none; its main land cover, 0 none, else class
        index + 1; that class's share 0..255) and `hmeta` (JSON); `card`
        (JSON); `status`; `config`. Browser -> kernel: `view`, `pick`, `ctl`.
        Tiles are custom messages: `s2`, a year's mosaic."""

        cells = traitlets.Bytes(b"").tag(sync=True)
        hattrs = traitlets.Bytes(b"").tag(sync=True)
        hmeta = traitlets.Unicode("{}").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        card = traitlets.Unicode("").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)
        ctl = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None  # async (src, z, x, y, year) -> PNG bytes or None
            self.tile_times = {}  # (src, z, x, y, year) -> {"wait", "run"} ms, set by tile_fn
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict) or content.get("kind") != "tile":
                return
            try:
                asyncio.get_running_loop().create_task(self._tile(content, time.time()))
            except RuntimeError as e:
                self.send({"kind": "tile", "id": content.get("id"), "err": f"no loop: {e}"})

        async def _tile(self, c, t_recv=None):
            """A FAILURE IS AN ERROR, never an empty tile (deck caches an empty
            tile as loaded and the area stays blank for good)."""
            if self.tile_fn is None:
                self.send({"kind": "tile", "id": c["id"], "err": "no tile_fn (re-run the wiring cell)"})
                return
            key = (c.get("src", "s2"), int(c["z"]), int(c["x"]), int(c["y"]), int(c["year"]))
            t_run = time.time()
            try:
                png = await self.tile_fn(*key)
            except Exception as e:
                self.tile_times.pop(key, None)
                self.send({"kind": "tile", "id": c["id"], "err": f"{type(e).__name__}: {e}"})
                return
            # timings for the tests: wall clock at receipt and at send, the
            # loop's delay before the tile started, and tile_fn's own split
            kt = {"recv": t_recv, "sent": time.time(), "loop": 1e3 * (t_run - t_recv) if t_recv else None, **self.tile_times.pop(key, {})}
            if png is None:
                self.send({"kind": "tile", "id": c["id"], "empty": True, "kt": kt})
            else:
                self.send({"kind": "tile", "id": c["id"], "kt": kt}, buffers=[png])

        _css = r"""
        .at{--glass:rgba(255,255,255,.9);--glass-hi:#fff;--line:rgba(24,32,40,.13);--text:#1b2127;--muted:#5d6873;--faint:rgba(24,32,40,.18);--cool:#0072b2;--sel:rgba(24,32,40,.08);
          position:relative;width:100%;background:#eef0f1;color:var(--text);font:14px/1.45 "Instrument Sans",ui-sans-serif,system-ui,sans-serif;font-variant-numeric:tabular-nums;overflow:hidden;border-radius:10px;-webkit-font-smoothing:antialiased}
        .at.fit{position:fixed;inset:0;z-index:9999;border-radius:0}
        .at *{box-sizing:border-box}
        .at-pane{position:relative;width:100%}
        .at-map{position:absolute;inset:0}
        .at-map.holding{cursor:ns-resize}
        .at-map.holding.key{cursor:grab}
        .at-glass{background:var(--glass);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:12px;box-shadow:0 6px 22px rgba(20,30,40,.14)}
        .at button{font:inherit;color:inherit}
        .at button:focus-visible,.at input:focus-visible{outline:2px solid var(--cool);outline-offset:2px}
        .at-top{position:absolute;left:12px;top:12px;z-index:6;display:flex;flex-direction:column;gap:8px;align-items:flex-start;max-width:calc(100% - 360px)}
        .at-search{position:relative;z-index:2;display:flex;align-items:center;gap:8px;padding:0 12px;height:40px;width:270px}
        .at-search svg{flex:0 0 auto;opacity:.6}
        .at-search input{flex:1;min-width:0;background:none;border:0;color:var(--text);font:inherit;outline:none}
        .at-search input::placeholder{color:var(--muted)}
        .at-hits{position:absolute;left:-1px;right:-1px;top:46px;display:none;padding:4px;background:var(--glass-hi)}
        .at-hit{padding:7px 10px;border-radius:8px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .at-hit small{display:block;color:var(--muted);font-size:12px}
        .at-hit.sel{background:var(--sel)}
        .at-panel{display:flex;flex-direction:column;gap:8px;padding:9px 11px}
        .at-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
        .at-lab{font-size:12.5px;color:var(--muted);min-width:64px}
        .seg-s{display:flex;gap:2px;padding:2px;border:1px solid var(--line);border-radius:9px;width:max-content}
        .seg-s button{border:0;background:none;color:var(--muted);padding:3px 10px;border-radius:7px;cursor:pointer}
        .seg-s button:hover{color:var(--text)}
        .seg-s button.on{background:var(--text);color:#fff}
        .seg-s.col{flex-direction:column;align-items:stretch}
        .seg-s.col button{text-align:left}
        .at-row.top{align-items:flex-start}
        .at-row.top .at-lab{padding-top:5px}
        .at-hd{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:-2px -4px -2px 0}
        .at-hd .t{font-size:12.5px;font-weight:600}
        .at-cb{flex:0 0 auto;border:0;background:none;color:var(--muted);cursor:pointer;width:26px;height:26px;border-radius:7px;display:inline-flex;align-items:center;justify-content:center;padding:0}
        .at-cb:hover{background:var(--sel);color:var(--text)}
        .at-cb svg{transition:transform .15s}
        .collapsed .at-cb svg{transform:rotate(-90deg)}
        .at-panel.collapsed{padding:2px 3px 2px 10px;gap:0;border-radius:10px}
        .at-panel.collapsed .at-hd .t{font-size:12px}
        .at-panel.collapsed .at-cb{width:22px;height:22px}
        .at-panel.collapsed .at-row{display:none}
        .at-yc .yr .at-cb{margin-left:auto;align-self:flex-start;margin-top:-4px;margin-right:-8px}
        .at-yc.collapsed{width:auto}
        .at-yc.collapsed .yr span{max-width:120px}
        .at-yc.collapsed>:not(.yr){display:none}
        .at-key{display:inline-flex;align-items:center;flex-wrap:wrap;gap:4px 8px;font-size:12.5px;color:var(--muted)}
        .at-ramp{height:10px;border-radius:3px;width:150px}
        .at-win{position:relative;width:170px;height:28px;flex:0 0 auto}
        .at-win input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}
        .at-win input:focus{outline:none}
        .at-win input::-webkit-slider-runnable-track{background:none;height:22px}
        .at-win input::-moz-range-track{background:none;height:22px}
        .at-win input::-webkit-slider-thumb{pointer-events:auto;-webkit-appearance:none;appearance:none;width:14px;height:14px;margin-top:4px;border-radius:50%;background:var(--text);border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.3);cursor:grab}
        .at-win input::-moz-range-thumb{pointer-events:auto;width:14px;height:14px;border-radius:50%;background:var(--text);border:2px solid #fff;cursor:grab}
        .at-win .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:var(--faint);border-radius:2px}
        .at-win .spn{position:absolute;top:9px;height:4px;background:var(--text);border-radius:2px}
        .at-win .tks{position:absolute;left:8px;right:8px;top:19px;display:flex;justify-content:space-between;font-size:9px;color:var(--muted);line-height:1}
        .at-win .tks span{width:0;display:flex;justify-content:center}
        .at-win .tks i{font-style:normal}
        .at-wtxt{font-size:12.5px;white-space:nowrap}
        .at-tools{position:absolute;right:12px;top:12px;z-index:7;display:flex;gap:8px}
        .at-btn{height:40px;min-width:40px;padding:0 13px;display:inline-flex;align-items:center;justify-content:center;gap:7px;cursor:pointer;white-space:nowrap}
        .at-btn:hover{border-color:rgba(24,32,40,.3)}
        .at-btn.on{background:var(--text);color:#fff;border-color:var(--text)}
        .at-bar{position:absolute;left:0;right:0;top:0;height:3px;z-index:9;overflow:hidden;pointer-events:none;opacity:0;transition:opacity .3s}
        .at-bar.busy{opacity:1}
        .at-bar i{position:absolute;top:0;height:3px;width:28%;background:linear-gradient(90deg,transparent,var(--text),transparent);animation:at-run 1.2s ease-in-out infinite}
        @keyframes at-run{0%{left:-28%}100%{left:100%}}
        .at-msg{position:absolute;left:50%;transform:translateX(-50%);bottom:16px;z-index:5;font-size:13px;color:var(--muted);padding:6px 11px;display:none;max-width:min(520px,calc(100% - 24px))}
        .at-msg.err{color:#8a4b00}
        .at-yc{position:absolute;right:12px;top:60px;z-index:6;width:320px;max-width:calc(100% - 24px);padding:14px 16px 12px;transform-origin:top right}
        .at-yc .yr{display:flex;align-items:flex-end;gap:12px}
        .at-yc .yr b{font-size:56px;line-height:.86;font-weight:600;letter-spacing:-.035em;font-stretch:88%}
        .at-yc .yr span{font-size:12.5px;color:var(--muted);line-height:1.35;padding-bottom:2px}
        .at-yc .yr.quiet{align-items:center}
        .at-yc .yr.quiet span{padding-bottom:0}
        .at-yc .yr.quiet .at-cb{margin-top:-2px;align-self:center}
        .at-yc.holding .yr span{color:var(--text)}
        .at-yc h4{margin:14px 0 2px;font-size:13.5px;font-weight:600}
        .at-yc .sub{color:var(--muted);font-size:12.5px;margin:0 0 6px}
        .at-yc .hex{border-top:1px solid var(--line);margin-top:12px;padding-top:12px;position:relative}
        .at-yc .hex .place{color:var(--text);font-size:13px;font-weight:600;margin-bottom:2px;padding-right:28px}
        .at-yc .hex .place span{font-weight:400;color:var(--muted)}
        .at-yc .hex h3{margin:0 0 6px;font-size:16px;font-weight:600;letter-spacing:-.005em;padding-right:28px}
        .at-yc .hex p{margin:0 0 8px}
        .at-yc .coords{display:grid;grid-template-columns:1fr auto;gap:3px 8px;align-items:center;margin:6px 0 10px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
        .at-yc .coords code{user-select:all;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .at-yc .coords button{border:1px solid var(--line);background:none;color:var(--muted);cursor:pointer;border-radius:6px;padding:1px 7px;font:12px/1.5 "Instrument Sans",ui-sans-serif,system-ui,sans-serif}
        .at-yc .coords button:hover{background:var(--sel);color:var(--text)}
        .at-yc .x{position:absolute;right:-6px;top:6px;border:0;background:none;color:var(--muted);cursor:pointer;width:28px;height:28px;border-radius:8px;font-size:17px;line-height:1}
        .at-yc .x:hover{background:var(--sel);color:var(--text)}
        .at-yc svg text{font-size:10.5px;fill:var(--muted)}
        .at-yc svg .lbl{fill:var(--text);font-weight:600}
        .at-lc{display:grid;grid-template-columns:1fr 120px 36px;gap:4px 8px;align-items:center;font-size:12.5px;margin:4px 0 6px}
        .at-lc i{display:block;height:8px;border-radius:0 4px 4px 0;background:var(--text);opacity:.72}
        .at-lc span:nth-child(3n){text-align:right;color:var(--muted)}
        .at-tip{position:absolute;z-index:10;pointer-events:none;background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12.5px;line-height:1.4;display:none;box-shadow:0 4px 14px rgba(20,30,40,.12);max-width:260px}
        .at-more{position:absolute;right:12px;top:60px;z-index:8;width:300px;padding:8px;display:none}
        .at-more .item{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:9px 10px;border-radius:9px}
        .at-more .item:hover{background:var(--sel)}
        .at-more .item small{display:block;color:var(--muted);font-size:12px}
        .at-more hr{border:0;border-top:1px solid var(--line);margin:4px 6px}
        .at-chip{border:1px solid var(--line);background:#fff;border-radius:999px;padding:4px 12px;cursor:pointer}
        .at-chip:hover{border-color:rgba(24,32,40,.35)}
        .at-sw{position:relative;width:34px;height:20px;flex:0 0 auto;border-radius:999px;background:rgba(24,32,40,.2);border:0;cursor:pointer;transition:background .2s}
        .at-sw::after{content:"";position:absolute;left:3px;top:3px;width:14px;height:14px;border-radius:50%;background:#fff;transition:left .2s}
        .at-sw.on{background:var(--cool)}
        .at-sw.on::after{left:17px}
        .at-more input[type=range]{width:120px;accent-color:var(--cool)}
        .at-about{position:absolute;inset:0;z-index:20;display:none;align-items:center;justify-content:center;background:rgba(20,30,40,.35)}
        .at-about .box{width:min(620px,calc(100% - 32px));max-height:calc(100% - 64px);overflow:auto;padding:22px 26px;line-height:1.55;background:#fff}
        .at-about h2{margin:0 0 10px;font-size:22px;font-weight:600;letter-spacing:-.01em}
        .at-about p{margin:0 0 10px;max-width:66ch}
        .at-about small{color:var(--muted)}
        .at .maplibregl-ctrl-group{border:1px solid var(--line);box-shadow:0 6px 22px rgba(20,30,40,.14);border-radius:10px}
        @media (max-width:760px){.at-top{max-width:calc(100% - 24px)}.at-search{width:calc(100vw - 48px)}.at-yc{top:auto;bottom:12px;max-height:45%;overflow:auto}.at-tools{top:108px}}
        @media (prefers-reduced-motion:reduce){.at-bar i{animation:none;left:0;width:100%}}
        """

        _esm = r"""
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {TileLayer, H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {latLngToCell, getResolution, cellToBoundary, cellToLatLng, isValidCell} from "https://esm.sh/h3-js@4.5.0";
        import {Protocol as PMProtocol} from "https://esm.sh/pmtiles@4.5.0";
        maplibregl.addProtocol("pmtiles", new PMProtocol().tile);

        const STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";
        const FONTS = "https://fonts.googleapis.com/css2?family=Instrument+Sans:wdth,wght@75..100,400..700&display=swap";
        const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
        const fmt = (n) => Number(n).toLocaleString("en-US");
        const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
        const cap = (s) => s ? s[0].toUpperCase() + s.slice(1) : s;
        const INK = [27, 33, 39];

        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength);
          return null;
        }
        const copyOf = (u8) => u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
        const el_ = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
        const ICON = {
          search: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
          more: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg>',
          expand: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
          chev: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9l6 6 6-6"/></svg>',
          shrink: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>',
        };
        // how much a hexagon moved, in words, from its 0..1 level in this view
        const howMuch = (t) => t >= 0.75 ? "a lot" : t >= 0.4 ? "a fair amount" : t >= 0.15 ? "a little" : "barely";
        const FAIR = 1 + Math.round(254 * 0.4);  // the level byte at "a fair amount"

        function render({model, el}) {
          let cfg = {};
          try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
          if (!document.getElementById("at-fonts")) {
            const f = document.createElement("link"); f.id = "at-fonts"; f.rel = "stylesheet"; f.href = FONTS; document.head.appendChild(f);
          }
          const mlcss = document.createElement("link");
          mlcss.rel = "stylesheet"; mlcss.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css";
          el.appendChild(mlcss);

          const S2Y = cfg.s2_years || [2022, 2023, 2024, 2025];
          const VIR = (cfg.viridis || "440154fde725").match(/.{6}/g).map((h) => [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)));
          const vir = (t) => { t = Math.max(0, Math.min(1, t)) * (VIR.length - 1); const i = Math.min(VIR.length - 2, Math.floor(t)), f = t - i; return VIR[i].map((v, j) => Math.round(v + (VIR[i + 1][j] - v) * f)); };
          const virCss = (n) => Array.from({length: n}, (_, i) => `rgb(${vir(i / (n - 1)).join(",")})`).join(",");
          // the year of the biggest change: YlOrBr less its near-white end, light
          // yellow for the first year to dark brown for the last (Stephen,
          // 2026-09-25: "use ylorbr for year changed"); a lightness ramp on the
          // orange leg, readable without red, and apart from how-much's viridis
          const YOB = ["fee391", "fec44f", "fe9929", "ec7014", "cc4c02", "993404", "662506"].map((h) => [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)));
          const yob = (t) => { t = Math.max(0, Math.min(1, t)) * (YOB.length - 1); const i = Math.min(YOB.length - 2, Math.floor(t)), f = t - i; return YOB[i].map((v, j) => Math.round(v + (YOB[i + 1][j] - v) * f)); };
          const yobCss = (n) => Array.from({length: n}, (_, i) => `rgb(${yob(i / (n - 1)).join(",")})`).join(",");
          const A_FILL = cfg.alpha_fill || 235, A_QUIET = cfg.alpha_quiet || 70;
          const HEXZ = cfg.hex_zoom || 9, HOLD_MS = cfg.hold_ms || 200, SLOP = cfg.hold_slop || 5;
          const st = {
            gmode: "much", y0: cfg.aef_from || 2022, y1: cfg.aef_to || 2025,
            imgYear: cfg.s2_year || S2Y[S2Y.length - 1], labels: true, s2scale: Number(cfg.s2_scale) || 1,
            fit: !!cfg.fit, holding: false,
          };

          // ---- the frame ----------------------------------------------------
          const root = el_("div", "at");
          const pane = el_("div", "at-pane");
          const mapEl = el_("div", "at-map");
          const bar = el_("div", "at-bar", "<i></i>");
          const msg = el_("div", "at-msg at-glass");
          pane.append(mapEl, bar, msg);
          root.appendChild(pane);
          el.appendChild(root);

          // top left: search, and the one control row for the hexagons
          const top = el_("div", "at-top");
          const search = el_("div", "at-search at-glass", ICON.search);
          const gc = el_("input"); gc.type = "search"; gc.placeholder = "Search a place or H3 string"; gc.autocomplete = "off"; gc.spellcheck = false;
          const hits = el_("div", "at-hits at-glass");
          search.append(gc, hits);
          const panel = el_("div", "at-panel at-glass");
          // every panel folds (Stephen, 2026-09-25); the fold is remembered in this browser
          const keep = (k, v) => { try { if (v === undefined) return localStorage.getItem("aef-lc-" + k) === "1"; localStorage.setItem("aef-lc-" + k, v ? "1" : "0"); } catch (e) {} return false; };
          const panelHd = el_("div", "at-hd", `<span class="t">AEF Change</span>`);
          const panelCb = el_("button", "at-cb", ICON.chev);
          panelHd.appendChild(panelCb);
          panel.appendChild(panelHd);
          const foldPanel = (on) => { panel.classList.toggle("collapsed", on); panelCb.title = on ? "show the controls" : "fold the controls"; keep("panel", on); };
          panelCb.onclick = (e) => { e.stopPropagation(); foldPanel(!panel.classList.contains("collapsed")); };
          foldPanel(keep("panel"));
          const rowOf = (label) => { const r = el_("div", "at-row"); if (label) r.appendChild(el_("span", "at-lab", label)); panel.appendChild(r); return r; };
          const segOf = (row, items, isOn, onClick) => {
            const seg = el_("div", "seg-s col");
            const bs = items.map(([k, label, title]) => { const b = el_("button", "", label); b.title = title || ""; b.onclick = () => onClick(k); seg.appendChild(b); return b; });
            row.appendChild(seg);
            return () => items.forEach(([k], i) => bs[i].classList.toggle("on", isOn(k)));
          };
          const rFill = rowOf("Color by");
          rFill.classList.add("top");
          const styleFill = segOf(rFill, [["much", "AEF Change", "how far the ground's AlphaEarth numbers moved between the first and last year read (S)"],
                                          ["year", "AEF Change Year", "the year each hexagon's change stood out most against the usual change that year, faded where the ground barely moved (D)"]],
                                  (k) => k === st.gmode, (k) => { st.gmode = k; recolorHex(); styleRows(); update(); });
          const rKey = rowOf("");
          rKey.classList.add("keep");
          const keyEl = el_("span", "at-key");
          rKey.appendChild(keyEl);
          // the window: the years AlphaEarth is read over; drag either end, it rereads on release
          const rWin = rowOf("Years read");
          const aefYears = cfg.aef_years || [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
          const win = el_("span", "at-win");
          const wTrk = el_("span", "trk"), wSpn = el_("span", "spn"), wTks = el_("span", "tks");
          for (const y of aefYears) wTks.appendChild(el_("span", "", `<i>’${String(y).slice(-2)}</i>`));
          const mkR = () => { const r = el_("input"); r.type = "range"; r.min = 0; r.max = aefYears.length - 1; r.step = 1; r.title = "the years AlphaEarth is read over: drag either end, it rereads when you let go"; return r; };
          const rFrom = mkR(), rTo = mkR();
          const wTxt = el_("span", "at-wtxt");
          win.append(wTrk, wSpn, wTks, rFrom, rTo);
          rWin.append(win, wTxt);
          function styleWin() {
            const i0 = Math.max(0, aefYears.indexOf(st.y0)), i1 = Math.max(0, aefYears.indexOf(st.y1)), n = Math.max(1, aefYears.length - 1);
            rFrom.value = i0; rTo.value = i1;
            rFrom.style.zIndex = i0 === n ? 3 : 2; rTo.style.zIndex = i1 === 0 ? 3 : 2;
            const usable = (win.clientWidth || 170) - 16;
            wSpn.style.left = (8 + usable * i0 / n) + "px"; wSpn.style.width = (usable * (i1 - i0) / n) + "px";
            wTxt.textContent = `${st.y0} to ${st.y1}`;
          }
          const onDrag = (which) => {
            let a = Number(rFrom.value), b = Number(rTo.value);
            if (a >= b) { if (which === "from") a = b - 1; else b = a + 1; }
            a = Math.max(0, a); b = Math.min(aefYears.length - 1, b);
            st.y0 = aefYears[a]; st.y1 = aefYears[b]; styleWin();
          };
          let winSent = [st.y0, st.y1];
          const winRelease = () => { if (st.y0 !== winSent[0] || st.y1 !== winSent[1]) { winSent = [st.y0, st.y1]; send("aef"); } };
          rFrom.addEventListener("input", () => onDrag("from")); rTo.addEventListener("input", () => onDrag("to"));
          rFrom.addEventListener("change", winRelease); rTo.addEventListener("change", winRelease);
          try { new ResizeObserver(styleWin).observe(win); } catch (e) {}
          function styleKey() {
            const y0 = hmeta.y0 || st.y0, y1 = hmeta.y1 || st.y1;
            const out_ = map && map.getZoom() < HEXZ;
            // the title is what is drawn, open or folded (Stephen, 2026-09-25)
            panelHd.querySelector(".t").textContent = out_ ? "Land cover" : st.gmode === "year" ? "AEF Change Year" : "AEF Change";
            // no hexagons zoomed out, so nothing to color (Stephen, 2026-09-25)
            rFill.style.display = out_ ? "none" : "";
            if (out_) {
              keyEl.innerHTML = `land cover, ESA WorldCover 2021: ` + (cfg.wc_key || []).map(([nm, hx]) => `<span style="display:inline-flex;align-items:center;gap:4px"><i style="width:10px;height:10px;border-radius:2px;background:#${hx}"></i>${esc(nm)}</span>`).join(" ");
              return;
            }
            keyEl.innerHTML = st.gmode === "much"
              ? `barely <i class="at-ramp" style="background:linear-gradient(90deg,${virCss(8)})"></i> a lot, ${y0} to ${y1}`
              : `${y0 + 1} <i class="at-ramp" style="background:linear-gradient(90deg,${yobCss(8)})"></i> ${y1}, faded where it barely moved`;
          }
          function styleRows() { styleFill(); styleWin(); styleKey(); }
          top.append(search, panel);
          pane.appendChild(top);

          // top right: settings and fill the window, then the year card
          const tools = el_("div", "at-tools");
          const bMore = el_("button", "at-btn at-glass", ICON.more); bMore.title = "settings and about";
          const bFit = el_("button", "at-btn at-glass", ICON.expand); bFit.title = "fill the window (X)";
          tools.append(bMore, bFit);
          pane.appendChild(tools);
          const yc = el_("div", "at-yc at-glass");
          pane.appendChild(yc);
          const tip = el_("div", "at-tip");
          pane.appendChild(tip);

          const more = el_("div", "at-more at-glass");
          const item = (title, sub, ctl) => { const r = el_("div", "item"); r.append(el_("div", "", `${title}${sub ? `<small>${sub}</small>` : ""}`), ctl); more.appendChild(r); return r; };
          const sw = (get, set) => { const b = el_("button", "at-sw"); b.setAttribute("role", "switch"); const sty = () => { b.classList.toggle("on", !!get()); b.setAttribute("aria-checked", String(!!get())); }; b.onclick = () => { set(!get()); sty(); }; sty(); b.sty = sty; return b; };
          const gam = el_("input"); gam.type = "range"; gam.min = 0.3; gam.max = 2.5; gam.step = 0.1; gam.value = st.s2scale;
          gam.title = "imagery brightness (gamma); double-click for 1.0";
          let gamT = null;
          gam.oninput = () => { st.s2scale = Number(gam.value); clearTimeout(gamT); gamT = setTimeout(() => send("s2scale"), 250); };
          gam.ondblclick = () => { gam.value = 1; gam.oninput(); };
          item("Imagery brightness", "; and ' step it", gam);
          const swLab = sw(() => st.labels, (v) => { st.labels = v; labels(v); });
          item("Place names", "", swLab);
          more.appendChild(el_("hr"));
          const bAbout = el_("button", "at-chip", "About this map"); bAbout.style.margin = "4px 10px 6px";
          more.appendChild(bAbout);
          pane.appendChild(more);

          const about = el_("div", "at-about");
          about.innerHTML = `<div class="box at-glass">
            <h2>Where the ground changed</h2>
            <p><b>AlphaEarth</b> describes every 10 m of ground with 64 numbers a year, 2017 to 2025. The hexagons show how far those numbers moved between the first and last year read, in viridis, stretched to what is in view: yellow moved most. Switch to <b>AEF Change Year</b> to color each hexagon by the year its change stood out most, light yellow for the first year to dark brown for the last. Each year is judged against the usual change that year in view, because the embeddings shift as a whole between some years (2024 to 2025 most of all). Hexagons fade where they barely moved.</p>
            <p><b>Hold space</b> to see the Sentinel-2 yearly imagery (Earth Genome, 2022 to 2025) instead of the hexagons. It opens on ${S2Y[0]} the first time, then on whichever year you left it at. The mouse stays free: move it off what you want to see, drag the map to look around, or click a cell for its H3 string and lat, long. <b>Scroll</b> while holding to step through the years; let go and the hexagons come back.</p>
            <p><b>Click</b> a hexagon for its account: each year-to-year step, and what the ground is by <b>ESA WorldCover</b> 2021. WorldCover is one map of one year, so it says what a place is, not when it changed.</p>
            <p><small>Keys: hold space for the imagery, scroll for its year; S AEF Change, D AEF Change Year; [ and ] the imagery year; ; and ' its brightness; - = and _ + the years read; L place names; X fill the window; / search (a place, or paste an H3 string); Esc close.</small></p>
            <p><small>AlphaEarth Foundations by Google and Google DeepMind (CC BY 4.0). ESA WorldCover 10 m 2021 v200, contains modified Copernicus Sentinel data processed by the ESA WorldCover consortium (CC BY 4.0). Sentinel-2 mosaics by Earth Genome (CC BY 4.0). Place names from Overture Maps divisions (ODbL), the PMTiles and, via Source Cooperative, fused/overture. Search by Photon over OpenStreetMap (ODbL). Basemap by Carto.</small></p>
            <div style="margin-top:12px"><button class="at-chip">Close</button></div></div>`;
          pane.appendChild(about);
          about.querySelector("button").onclick = () => { about.style.display = "none"; };
          about.onclick = (e) => { if (e.target === about) about.style.display = "none"; };
          bAbout.onclick = () => { more.style.display = "none"; about.style.display = "flex"; };

          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({act, s2scale: st.s2scale, y0: st.y0, y1: st.y1, n: Date.now()}, extra || {})));
            model.save_changes();
          };

          // ---- status ------------------------------------------------------------
          const ERR = /failed|error|no match|search:|timed? ?out|zoom in|^(deck|map|load|boot|\w+ tile):/i;
          let msgT = null;
          const note = (t, ms) => { msg.textContent = t; msg.style.display = t ? "block" : "none"; msg.classList.toggle("err", ERR.test(t)); clearTimeout(msgT); if (ms) msgT = setTimeout(() => { msg.style.display = "none"; }, ms); };
          const say = (t) => {
            t = (t || "").replace(/​/g, "");
            if (ERR.test(t)) { note(t); bar.classList.remove("busy"); return; }
            const busy = t.split(" | ").filter((p) => p.includes("…"));
            const names = [];
            for (const p of busy) {
              if (/AlphaEarth/i.test(p) && !names.includes("AlphaEarth")) names.push("AlphaEarth");
              if (/land cover/i.test(p) && !names.includes("land cover")) names.push("land cover");
            }
            bar.classList.toggle("busy", names.length > 0);
            note(names.length ? "Loading " + names.join(" and ") + "…" : "");
          };

          // ---- tiles ----------------------------------------------------------------
          const pending = new Map();
          let tseq = 0;
          const tstat = {asked: 0, got: 0, empty: 0, err: 0, abort: 0};
          const tlog = [];  // per tile, for the tests: asked, arrived, kernel times, bytes
          const tlogOf = new Map();
          model.on("msg:custom", (m, buffers) => {
            if (!m || m.kind !== "tile") return;
            const p = pending.get(m.id);
            if (!p) return;
            pending.delete(m.id);
            const lg = tlogOf.get(m.id);
            if (lg) { lg.got = Date.now(); lg.kt = m.kt || null; lg.bytes = buffers && buffers.length ? (buffers[0].byteLength || 0) : 0; lg.err = m.err || null; tlogOf.delete(m.id); }
            if (m.err) { tstat.err++; p.reject(new Error(m.err)); return; }
            if (m.empty || !buffers || !buffers.length) { tstat.empty++; p.resolve(null); return; }
            tstat.got++;
            p.resolve(bytesOf(buffers[0]));
          });
          const ask = (src, year, index, signal) => new Promise((resolve, reject) => {
            const id = ++tseq; tstat.asked++;
            pending.set(id, {resolve, reject});
            const lg = {id, src, year, z: index.z, x: index.x, y: index.y, asked: Date.now()};
            tlog.push(lg); tlogOf.set(id, lg); if (tlog.length > 4000) tlog.splice(0, 1000);
            model.send({kind: "tile", id, src, year, x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => { if (pending.has(id)) lg.aborted = Date.now(); tlogOf.delete(id); pending.delete(id); tstat.abort++; const e = new Error("aborted"); e.name = "AbortError"; reject(e); });
          });
          const pngBitmap = (u8) => createImageBitmap(new Blob([u8], {type: "image/png"}));

          // ---- the hexagons -----------------------------------------------------------
          let hexes = [], N = 0, res = -1, hexIndex = new Map(), hattrs = null, hmeta = {}, hcol = null, hcol32 = null, hexSeq = 0, hover = null, picked = null, imgPick = null;
          function recolorHex() {
            if (!N || !hattrs || hattrs.length !== 4 * N) { hcol = null; hcol32 = null; return; }
            hcol = new Uint8Array(4 * N);
            hcol32 = new Uint32Array(hcol.buffer);
            const y0 = hmeta.y0 || st.y0, y1 = hmeta.y1 || st.y1, span = Math.max(1, y1 - y0 - 1);
            for (let i = 0; i < N; i++) {
              const o = 4 * i, yc_ = hattrs[o], lv = hattrs[o + 1];
              if (!lv) continue;
              const t = (lv - 1) / 254;
              let col, a;
              // quiet ground faint, change in full ink (Stephen, 2026-09-25)
              if (st.gmode === "much") { col = vir(t); a = Math.round(A_QUIET + (A_FILL - A_QUIET) * t); }
              else if (yc_) { col = yob(y1 - y0 > 1 ? (2000 + yc_ - (y0 + 1)) / span : 1); a = Math.round(A_QUIET + (A_FILL - A_QUIET) * t); }
              else continue;
              hcol[o] = col[0]; hcol[o + 1] = col[1]; hcol[o + 2] = col[2]; hcol[o + 3] = a;
            }
            hexSeq++;
          }
          const hexAt = (ll) => { if (res < 0 || !map || map.getZoom() < HEXZ) return -1; try { const h = latLngToCell(ll.lat, ll.lng, res); const i = hexIndex.get(h); return i == null ? -1 : i; } catch (e) { return -1; } };
          function hexWords(i) {
            const o = 4 * i, yb = hattrs[o], lv = hattrs[o + 1], lc = hattrs[o + 2], sh = hattrs[o + 3];
            if (!lv) return "No AlphaEarth data here.";
            let s = `<b>Changed ${howMuch((lv - 1) / 254)}</b>${yb ? `, most in ${2000 + yb}` : ""}`;
            const names = hmeta.classes || [];
            if (lc && names[lc - 1]) s += `<br>${cap(names[lc - 1])}, ${Math.round(100 * sh / 255)}% of it`;
            return s;
          }

          // ---- the year card ------------------------------------------------------------
          let cardData = null;
          // one bar per year you can scroll to, plus any other year the window
          // dates; an imagery year the window cannot date is an empty slot
          function viewCounts() {
            const out = {}, dated = {};
            const y0 = hmeta.y0 || st.y0, y1 = hmeta.y1 || st.y1;
            const ys = [...new Set([...S2Y, ...Array.from({length: Math.max(0, y1 - y0)}, (_, i) => y0 + 1 + i)])].sort((a, b) => a - b);
            for (const y of ys) { out[y] = 0; dated[y] = y > y0 && y <= y1; }
            let total = 0;
            if (hattrs) for (let i = 0; i < N; i++) { const yb = hattrs[4 * i], lv = hattrs[4 * i + 1]; if (lv) total++; if (yb && lv >= FAIR && out[2000 + yb] != null) out[2000 + yb]++; }
            return {years: out, dated, total};
          }
          // bars per year: one series, ink on a baseline; the imagery year in
          // full ink with its count, the others lighter; a tooltip per bar
          function yearBars(c) {
            const ys = Object.keys(c.years).map(Number);
            if (!ys.length) return "";
            const W = 286, H = 78, base = 60, gap = 6, bw = Math.min(56, (W - gap * (ys.length - 1)) / ys.length);
            const x0 = (W - (bw * ys.length + gap * (ys.length - 1))) / 2;
            const max = Math.max(1, ...ys.map((y) => c.years[y]));
            let s = `<svg width="${W}" height="${H}" role="img" aria-label="hexagons in view that changed a fair amount or more, by the year of their biggest change">`;
            ys.forEach((y, i) => {
              const v = c.years[y], h = v ? Math.max(3, (base - 14) * v / max) : 0, x = x0 + i * (bw + gap), cur = y === st.imgYear;
              const r = Math.min(4, h / 2);
              if (h) s += `<path d="M${x},${base} v${-(h - r)} q0,${-r} ${r},${-r} h${bw - 2 * r} q${r},0 ${r},${r} v${h - r} z" fill="${rgba(INK, cur ? 0.9 : 0.28)}"/>`;
              const why = c.dated[y] ? `${fmt(v)} hexagon${v === 1 ? "" : "s"} changed most between the ${y - 1} and ${y} pictures` : `${y} is outside the years read: widen them to ${y - 1} to date changes into ${y}`;
              s += `<rect x="${x - gap / 2}" y="0" width="${bw + gap}" height="${H}" fill="transparent" data-tip="${why}"/>`;
              if (!c.dated[y]) s += `<line x1="${x}" x2="${x + bw}" y1="${base - 1}" y2="${base - 1}" stroke="rgba(24,32,40,.3)" stroke-dasharray="2 3"/>`;
              if (cur && v) s += `<text class="lbl" x="${x + bw / 2}" y="${base - h - 4}" text-anchor="middle">${fmt(v)}</text>`;
              s += `<text x="${x + bw / 2}" y="${H - 4}" text-anchor="middle"${cur ? ' class="lbl"' : ""}>${y}</text>`;
            });
            s += `<line x1="0" x2="${W}" y1="${base + 0.5}" y2="${base + 0.5}" stroke="rgba(24,32,40,.25)"/></svg>`;
            return s;
          }
          // the clicked hexagon's steps, each as a multiple of that year's
          // median step in view: the biggest in full ink, a dashed line at 1
          function stepBars(rel, steps, years, big) {
            if (!rel || !rel.length) return "";
            const W = 286, H = 74, base = 56, gap = 6, n = rel.length, bw = Math.min(46, (W - gap * (n - 1)) / n);
            const x0 = (W - (bw * n + gap * (n - 1))) / 2;
            const vmax = Math.max(2, ...rel.filter((v) => v != null));
            const yOf = (v) => base - (base - 10) * Math.min(1, v / vmax);
            let s = `<svg width="${W}" height="${H}" role="img" aria-label="AlphaEarth year-to-year change for this hexagon">`;
            rel.forEach((v, i) => {
              const x = x0 + i * (bw + gap), y = years[i];
              if (v != null) {
                const top_ = yOf(v), h = base - top_, r = Math.min(4, h / 2);
                if (h > 0.5) s += `<path d="M${x},${base} v${-(h - r)} q0,${-r} ${r},${-r} h${bw - 2 * r} q${r},0 ${r},${r} v${h - r} z" fill="${rgba(INK, y === big ? 0.9 : 0.28)}"/>`;
              }
              s += `<rect x="${x - gap / 2}" y="0" width="${bw + gap}" height="${H}" fill="transparent" data-tip="${y - 1} to ${y}: ${v == null ? "no data" : `${v.toFixed(1)} times the usual step here that year (${steps[i].toFixed(3)})`}"/>`;
              s += `<text x="${x + bw / 2}" y="${H - 4}" text-anchor="middle"${y === big ? ' class="lbl"' : ""}>’${String(y - 1).slice(-2)} to ’${String(y).slice(-2)}</text>`;
            });
            const ty = yOf(1); s += `<line x1="0" x2="${W}" y1="${ty}" y2="${ty}" stroke="rgba(24,32,40,.55)" stroke-dasharray="3 3"/><text x="${W}" y="${ty - 3}" text-anchor="end">usual step here</text>`;
            s += `<line x1="0" x2="${W}" y1="${base + 0.5}" y2="${base + 0.5}" stroke="rgba(24,32,40,.25)"/></svg>`;
            return s;
          }
          function hexSection(c) {
            if (!c || !c.kind) return "";
            let h = `<div class="hex"><button class="x" title="close (Esc)" aria-label="close">×</button>`;
            if (c.place && c.place.length) h += `<div class="place">${c.place.map((q) => typeof q === "string" ? esc(q) : esc(q.name) + (q.tag ? ` <span>(${esc(q.tag)})</span>` : "")).join(", ")}</div>`;
            if (c.kind === "note") return h + `<h3>${esc(c.title || "")}</h3></div>`;
            h += `<h3>This hexagon</h3>`;
            // picked on the imagery: its H3 string and center, each copyable
            if (c.cell && c.cell === imgPick) {
              let ll = null; try { ll = cellToLatLng(c.cell); } catch (e) {}
              const lat_lon = ll ? `${ll[0].toFixed(6)}, ${ll[1].toFixed(6)}` : "";
              h += `<div class="coords"><code title="H3 string">${esc(c.cell)}</code><button data-copy="${esc(c.cell)}">copy</button>`;
              if (lat_lon) h += `<code title="lat, long of the cell's center">${lat_lon}</code><button data-copy="${lat_lon}">copy</button>`;
              h += `</div>`;
            }
            if (c.level == null) h += `<p>No AlphaEarth data here.</p>`;
            else {
              h += `<p>AlphaEarth: the ground changed <b>${howMuch(c.level)}</b> from ${c.y0} to ${c.y1}, compared with the rest of the view. Its year-to-year change stood out most between the <b>${c.big - 1} and ${c.big}</b> pictures.</p>`;
              h += stepBars(c.rel, c.steps, c.step_years, c.big);
              if (S2Y.includes(c.big) && S2Y.includes(c.big - 1)) h += `<p class="sub">Hold space with the pointer near it and scroll between ${c.big - 1} and ${c.big} to see what happened.</p>`;
              else if (c.big) h += `<p class="sub">The imagery starts in ${S2Y[0]}, so there is no picture from before ${c.big} to compare.</p>`;
            }
            if (c.landcover && c.landcover.length) {
              h += `<h4>What the ground is, ESA WorldCover 2021</h4><div class="at-lc">`;
              for (const [nm, sh] of c.landcover) h += `<span>${esc(cap(nm))}</span><span><i style="width:${Math.max(2, 120 * sh)}px"></i></span><span>${Math.round(100 * sh)}%</span>`;
              h += `</div>`;
            } else h += `<p class="sub">No land cover read here.</p>`;
            if (c.km2) h += `<p class="sub">${c.km2 < 0.1 ? `${Math.round(c.km2 * 1e6).toLocaleString("en-US")} m²` : `${c.km2.toFixed(2)} km²`} hexagon.</p>`;
            return h + `</div>`;
          }
          let ycFolded = keep("card"), ycOpenedFor = null;
          function renderYear() {
            yc.classList.toggle("holding", st.holding);
            const c = viewCounts();
            // the imagery year only while the imagery shows (Stephen, 2026-09-25); otherwise the hint
            const cbH = `<button class="at-cb" title="${ycFolded ? "show the card" : "fold the card"}">${ICON.chev}</button>`;
            let h = st.holding
              ? `<div class="yr"><b>${st.imgYear}</b><span>Scroll for another year. Let go to see the hexagons.</span>${cbH}</div>`
              : `<div class="yr quiet"><span>Hold space for the Sentinel-2 imagery</span>${cbH}</div>`;
            if (N && hattrs) {
              h += `<h4>Where it changed, by year</h4><p class="sub">Hexagons in view that changed a fair amount or more, by the year their change stood out most</p>`;
              h += yearBars(c);
            } else h += `<p class="sub" style="margin-top:12px">${map && map.getZoom() < HEXZ ? `The map is land cover (ESA WorldCover 2021) out here. Zoom in to ${HEXZ} for the AlphaEarth change hexagons.` : "Reading AlphaEarth for this view…"}</p>`;
            h += hexSection(cardData);
            yc.innerHTML = h;
            const x = yc.querySelector(".x");
            if (x) x.onclick = () => closeCard();
            yc.classList.toggle("collapsed", ycFolded);
            const cb = yc.querySelector(".yr .at-cb");
            if (cb) cb.onclick = (e) => { e.stopPropagation(); ycFolded = !ycFolded; keep("card", ycFolded); renderYear(); };
            fitCard();
          }
          // the card never scrolls (Stephen, 2026-09-25: "that right panel
          // needs to fit to view no scrolling"): when its content is taller
          // than the pane below it, it is scaled down from its top right
          // corner to fit. On a narrow screen it keeps its own scroll.
          function fitCard() {
            yc.style.transform = "";
            if (window.matchMedia("(max-width:760px)").matches) return;
            const avail = pane.clientHeight - yc.offsetTop - 12, need = yc.offsetHeight;
            if (avail > 0 && need > avail) yc.style.transform = `scale(${avail / need})`;
          }
          yc.addEventListener("pointermove", (e) => {
            const t = e.target && e.target.getAttribute && e.target.getAttribute("data-tip");
            if (!t) { tip.style.display = "none"; return; }
            const p = pane.getBoundingClientRect();
            tip.textContent = t;
            tip.style.display = "block";
            tip.style.left = Math.max(8, e.clientX - p.left - tip.offsetWidth - 14) + "px";
            tip.style.top = (e.clientY - p.top + 12) + "px";
          });
          yc.addEventListener("pointerleave", () => { tip.style.display = "none"; });
          yc.addEventListener("click", (e) => {
            const b = e.target && e.target.closest && e.target.closest("[data-copy]");
            if (!b) return;
            e.stopPropagation();
            const t = b.getAttribute("data-copy");
            const done = () => { b.textContent = "copied"; setTimeout(() => { b.textContent = "copy"; }, 1200); };
            // the notebook may sit in an iframe without clipboard permission (molab): fall back to a selection copy
            const fallback = () => { const ta = document.createElement("textarea"); ta.value = t; ta.style.position = "fixed"; ta.style.opacity = "0"; root.appendChild(ta); ta.select(); try { document.execCommand("copy"); done(); } catch (e2) {} ta.remove(); };
            if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done, fallback); else fallback();
          });
          function renderCard() {
            try { cardData = JSON.parse(model.get("card") || "null"); } catch (e) { cardData = null; }
            picked = cardData && cardData.cell ? cardData.cell : null;
            // a new click opens a folded card once; folding it again stays folded
            if (cardData && cardData.n != null && cardData.n !== ycOpenedFor) { ycOpenedFor = cardData.n; if (ycFolded) { ycFolded = false; keep("card", false); } }
            renderYear(); update();
          }
          function closeCard() { model.set("pick", JSON.stringify({close: true, n: ++seq})); model.save_changes(); cardData = null; picked = null; imgPick = null; renderYear(); update(); }
          // a pick: the same cell again clears it (Stephen, 2026-09-25: the
          // selected hexagon stays "unless it's clicked again")
          function pickCell(cell, ll, pt, onImagery) {
            if (cell && cell === picked) { closeCard(); return; }
            imgPick = onImagery ? cell : null;
            model.set("pick", JSON.stringify({cell, lon: ll.lng, lat: ll.lat, admin: adminAt(pt), n: ++seq}));
            model.save_changes();
          }

          // ---- the layers --------------------------------------------------------------
          let map = null, ov = null;
          const slot = () => { const want = cfg.labels_slot || "watername_ocean"; const s = map && map.getStyle && map.getStyle(); if (!s || !s.layers || s.layers.some((x) => x.id === want)) return want; const l = s.layers.find((x) => x.type === "symbol"); return (l && l.id) || want; };
          // every imagery year stays mounted once the view is close enough, the
          // ones not shown at opacity 0, so their tiles load ahead and a scroll
          // while holding is instant
          const s2Layer = (year) => new TileLayer({
            id: "s2-" + year + "-g" + (cfg.s2_gen || 0),
            getTileData: async ({index, signal}) => { const u8 = await ask("s2", year, index, signal); return u8 ? pngBitmap(u8) : null; },
            onTileError: (e) => { if (!e || e.name !== "AbortError") say("s2 tile: " + ((e && e.message) || e)); },
            tileSize: cfg.tile || 256, minZoom: cfg.s2_min_z || 7, maxZoom: 14, refinementStrategy: "best-available", debounceTime: 120, beforeId: slot(),
            opacity: st.holding && year === st.imgYear ? 1 : 0,
            renderSubLayers: (p) => { if (!p.data) return null; const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]}); },
          });
          // below HEXZ: ESA WorldCover 2021 as tiles, read from the same COGs
          const wcLayer = () => new TileLayer({
            id: "wc",
            getTileData: async ({index, signal}) => { const u8 = await ask("wc", 2021, index, signal); return u8 ? pngBitmap(u8) : null; },
            onTileError: (e) => { if (!e || e.name !== "AbortError") say("land cover tile: " + ((e && e.message) || e)); },
            tileSize: cfg.tile || 256, minZoom: cfg.wc_min_z || 4, maxZoom: Math.ceil(HEXZ) - 1, refinementStrategy: "best-available", debounceTime: 120, beforeId: slot(),
            renderSubLayers: (p) => { if (!p.data) return null; const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]}); },
          });
          // the hexagons: tiles of cell numbers from the kernel (1 + the row in
          // this frame, 0 none), colored here from hcol, so a mode or window
          // change repaints without a round trip. A tile from an older frame
          // keeps its last picture until the new frame's tile replaces it.
          const unz = async (u8) => new Uint32Array(await new Response(new Blob([u8]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
          // hexagon edges drawn from the H3 boundary, not the tile's pixels
          // (Stephen, 2026-09-25: jagged at z19, where the z17 tiles are
          // stretched 8x). The tile says which hexagons are near a pixel (its
          // 3x3 texels, as local indices); each one's ring (h3-js
          // cellToBoundary, in tile pixel units) gives the fragment's signed
          // distance to that hexagon, and the hexagon covers the fragment by
          // that distance over one screen pixel. Smooth at any zoom; an edge
          // against no hexagon fades to clear.
          const GEO_W = 64, COL_W = 256;  // hexagons per row of the ring and color textures
          class HexEdgeLayer extends BitmapLayer {
            getShaders() {
              const s = super.getShaders();
              s.fs = s.fs.replace("uniform sampler2D bitmapTexture;", "uniform sampler2D bitmapTexture;\nuniform highp sampler2D hexGeom;\nuniform sampler2D hexCol;");
              s.fs = s.fs.replace("vec4 bitmapColor = texture(bitmapTexture, uv);", `
                ivec2 tsz = textureSize(bitmapTexture, 0);
                vec2 tp = uv * vec2(tsz);
                ivec2 tc = clamp(ivec2(floor(tp)), ivec2(0), tsz - 1);
                float pw = max(0.7071 * length(fwidth(tp)), 1e-5);
                int seen[9]; int ns = 0;
                vec4 acc = vec4(0.0);
                float cs = 0.0;  // coverage summed: near a corner the edge distances overlap past one pixel
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                  vec4 t = texelFetch(bitmapTexture, clamp(tc + ivec2(dx, dy), ivec2(0), tsz - 1), 0);
                  int k = int(t.r * 255.0 + 0.5) + 256 * int(t.g * 255.0 + 0.5) - 1;
                  if (k < 0) continue;
                  bool dup = false;
                  for (int j = 0; j < 9; j++) { if (j >= ns) break; if (seen[j] == k) dup = true; }
                  if (dup) continue;
                  seen[ns] = k; ns++;
                  vec2 v[10];
                  for (int j = 0; j < 5; j++) { vec4 g = texelFetch(hexGeom, ivec2(5 * (k % ${GEO_W}) + j, k / ${GEO_W}), 0); v[2 * j] = g.xy; v[2 * j + 1] = g.zw; }
                  vec2 ctr = vec2(0.0);
                  for (int j = 0; j < 10; j++) ctr += v[j];
                  ctr /= 10.0;
                  float sd = 1e9;
                  for (int j = 0; j < 10; j++) {
                    vec2 a = v[j], e = v[(j + 1) % 10] - a;
                    float L = length(e);
                    if (L < 1e-6) continue;
                    vec2 nr = vec2(-e.y, e.x) / L;
                    if (dot(ctr - a, nr) < 0.0) nr = -nr;
                    sd = min(sd, dot(tp - a, nr));
                  }
                  vec4 c = texelFetch(hexCol, ivec2(k % ${COL_W}, k / ${COL_W}), 0);
                  float cov = clamp(sd / pw + 0.5, 0.0, 1.0);
                  acc += vec4(c.rgb * c.a, c.a) * cov; cs += cov;
                }
                if (cs > 1.0) acc /= cs;
                vec4 bitmapColor = acc.a > 1e-4 ? vec4(acc.rgb / acc.a, min(acc.a, 1.0)) : vec4(0.0);`);
              return s;
            }
            updateState(params) {
              super.updateState(params);
              const {props, oldProps} = params, dev = this.context.device;
              const mk = (format, width, height, data) => dev.createTexture({format, width, height, data, mipmaps: false, sampler: {minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge"}});
              const st_ = this.state;
              if (props.pic !== oldProps.pic && props.pic) {
                st_.idTex && st_.idTex.destroy(); st_.geoTex && st_.geoTex.destroy();
                const q = props.pic;
                st_.idTex = mk("rg8unorm", q.n, q.n, q.idx);
                st_.geoTex = mk("rgba32float", 5 * GEO_W, q.gh, q.geo);
              }
              if (props.col !== oldProps.col && props.col) {
                st_.colTex && st_.colTex.destroy();
                st_.colTex = mk("rgba8unorm", COL_W, props.col.length / (4 * COL_W), props.col);
              }
            }
            finalizeState(ctx) {
              super.finalizeState(ctx);
              for (const k of ["idTex", "geoTex", "colTex"]) if (this.state[k]) this.state[k].destroy();
            }
            draw(opts) {
              const {model, coordinateConversion, bounds, idTex, geoTex, colTex} = this.state;
              if (!model || !idTex || !geoTex || !colTex || opts.shaderModuleProps.picking.isActive) return;
              model.setBindings({hexGeom: geoTex, hexCol: colTex});
              model.shaderInputs.setProps({bitmap: {bitmapTexture: idTex, bounds, coordinateConversion, desaturate: 0, tintColor: [1, 1, 1], transparentColor: [0, 0, 0, 0]}});
              model.draw(this.context.renderPass);
            }
          }
          HexEdgeLayer.layerName = "HexEdgeLayer";
          const ptimes = [];  // per hexagon tile painted, for the tests
          // once per tile and frame: the tile's hexagons as local indices (1 +,
          // 0 none) and their rings in tile pixels; the colors per repaint
          function tilePic(d) {
            if (d.seq !== hmeta.seq || !hcol32) return d.col ? d : null;  // an older frame's tile keeps its last picture
            if (d.col && d.cseq === hexSeq) return d;
            const tp = performance.now();
            if (!d.pic) {
              const n = d.side, ids = d.ids, loc = new Map(), rows = [], idx = new Uint8Array(2 * n * n);
              let last = 0, lastL = 0;
              for (let i = 0; i < ids.length; i++) {
                const k = ids[i];
                if (!k) continue;
                if (k !== last) { const l = loc.get(k); if (l === undefined) { lastL = rows.length; loc.set(k, lastL); rows.push(k - 1); } else lastL = l; last = k; }
                const v = Math.min(lastL + 1, 65535);
                idx[2 * i] = v & 255; idx[2 * i + 1] = v >> 8;
              }
              const K = rows.length, gh = Math.max(1, Math.ceil(K / GEO_W)), geo = new Float32Array(5 * GEO_W * gh * 4);
              const Z = 2 ** d.z, lonC = (d.x + 0.5) / Z * 360 - 180;
              for (let j = 0; j < K; j++) {
                const r = ring(hexes[rows[j]]);
                if (!r) continue;
                const m = Math.min(10, r.length - 1), o = (Math.floor(j / GEO_W) * 5 * GEO_W + 5 * (j % GEO_W)) * 4;
                for (let q = 0; q < 10; q++) {
                  let [lng, lat] = r[Math.min(q, m - 1)];
                  if (lng - lonC > 180) lng -= 360; else if (lng - lonC < -180) lng += 360;
                  const sn = Math.sin(lat * Math.PI / 180);
                  geo[o + 2 * q] = ((lng + 180) / 360 * Z - d.x) * n;
                  geo[o + 2 * q + 1] = ((0.5 - Math.log((1 + sn) / (1 - sn)) / (4 * Math.PI)) * Z - d.y) * n;
                }
              }
              d.pic = {n, idx, geo, gh, rows};
            }
            const rows = d.pic.rows, col = new Uint8Array(COL_W * Math.max(1, Math.ceil(rows.length / COL_W)) * 4), c32 = new Uint32Array(col.buffer);
            for (let j = 0; j < rows.length; j++) c32[j] = hcol32[rows[j]];
            d.col = col; d.cseq = hexSeq;
            ptimes.push({t: Date.now(), ms: performance.now() - tp, unz: d.unz, ready: d.done}); if (ptimes.length > 4000) ptimes.splice(0, 1000);
            return d;
          }
          const hexLayer = (visible) => new TileLayer({
            id: "hexes", visible,
            getTileData: async ({index, signal}) => {
              const seq = hmeta.seq, u8 = await ask("hex", seq, index, signal);
              if (!u8) return null;
              const t0 = performance.now(), ids = await unz(u8);
              return {ids, seq, side: Math.round(Math.sqrt(ids.length)), z: index.z, x: index.x, y: index.y, unz: performance.now() - t0, done: Date.now()};
            },
            onTileError: (e) => { if (!e || (e.name !== "AbortError" && !/stale/.test(e.message || ""))) say("hexagon tile: " + ((e && e.message) || e)); },
            tileSize: 256, minZoom: Math.floor(HEXZ), maxZoom: 17, refinementStrategy: "best-available", debounceTime: 60, beforeId: slot(),
            updateTriggers: {getTileData: [hmeta.seq], renderSubLayers: [hexSeq, hmeta.seq]},
            renderSubLayers: (p) => {
              const t = p.data ? tilePic(p.data) : null;
              if (!t) return null;
              const {west, south, east, north} = p.tile.bbox;
              return new HexEdgeLayer(p, {data: null, image: null, pic: t.pic, col: t.col, bounds: [west, south, east, north]});
            },
          });
          const ring = (h) => { try { return cellToBoundary(h, true); } catch (e) { return null; } };
          const outline = (id, h, color, width) => { const r = h ? ring(h) : null; return r ? new PathLayer({id, data: [r], getPath: (d) => d, getColor: color, widthUnits: "pixels", getWidth: width, beforeId: slot()}) : null; };
          function layers() {
            const out = [];
            const z = map ? map.getZoom() : 0;
            if (!st.holding && z < HEXZ) out.push(wcLayer());
            if (st.holding || z >= 9) for (const y of S2Y) out.push(s2Layer(y));  // preloaded from zoom 9 only: below it the tiles are decimated from L5 (slow)
            // while holding: the imagery, and over it only the two outlines
            // (Stephen, 2026-09-25: the selected hexagon "should appear on the
            // satellite", white on hover and gold when picked, as in the pair)
            // kept in the stack while hidden (holding, zoomed out) so its tiles stay cached
            if (hmeta.seq) out.push(hexLayer(!st.holding && !!hcol && z >= HEXZ));
            const hv = hover != null && hover >= 0 ? outline("hover", hexes[hover], [255, 255, 255, 235], 2) : null;
            if (hv) out.push(hv);
            const sc = searched ? outline("searched", searched, [0, 114, 178, 255], 3) : null;
            if (sc) out.push(sc);
            const pk = picked ? outline("picked", picked, [255, 200, 40, 255], 3) : null;
            if (pk) out.push(pk);
            return out;
          }
          function update() { if (ov) ov.setProps({layers: layers()}); }
          function labels(on) {
            if (!map || !map.isStyleLoaded()) return;
            (map.getStyle().layers || []).forEach((l) => { if (l.layout && l.layout["text-field"] !== undefined) map.setLayoutProperty(l.id, "visibility", on ? "visible" : "none"); });
          }

          // ---- hold: the imagery ------------------------------------------------------------
          // hold SPACE, only (the mouse stays free, so it can rest off the
          // building you are looking at: Stephen, 2026-09-25, "when my mouse is
          // over a tiny building, the mouse obscures the building"; the press
          // and hold on the map is gone: "that should just be space"). The
          // imagery opens on the year the last
          // hold left off at (2022 the first time); while holding, the wheel anywhere over the map or the card
          // steps the imagery year instead of zooming; letting go of whichever
          // started it ends it
          let holdT = null, holdAt = null, holdBy = null, wheelAcc = 0, lastStep = 0, suppressClick = false, lastPt = null;
          const stepImg = (d) => { const i = S2Y.indexOf(st.imgYear); const n = S2Y[Math.max(0, Math.min(S2Y.length - 1, (i < 0 ? S2Y.length - 1 : i) + d))]; if (n !== st.imgYear) { st.imgYear = n; renderYear(); update(); } };
          function beginHold(x, y, by) {
            holdT = null;
            if (!map || st.holding) return;
            st.holding = true;
            holdBy = by;
            if (by === "mouse") suppressClick = true;
            mapEl.classList.add("holding");
            mapEl.classList.toggle("key", by === "key");
            // the wheel is the year while holding (the capture listener on root
            // keeps it from the map, so scrollZoom stays on: disabling it
            // mid-zoom left maplibre's zoom marked active, and scroll froze
            // after the hold); a space hold leaves the map free to drag ("i'd
            // like to be able to move the map when space is pressed"), a mouse
            // hold cannot (the press is the hold)
            if (by === "mouse") map.dragPan.disable();
            tip.style.display = "none";
            renderYear(); update();
          }
          function endHold(by) {
            if (by !== "key") { clearTimeout(holdT); holdT = null; holdAt = null; }
            if (!st.holding || (by && by !== holdBy)) return;
            st.holding = false; holdBy = null;
            mapEl.classList.remove("holding");
            if (map) map.dragPan.enable();
            wheelAcc = 0;
            renderYear(); update();
            if (lastPt && map) showHexTip(lastPt);
          }
          mapEl.addEventListener("pointermove", (e) => {
            lastPt = {x: e.clientX, y: e.clientY};
            if (holdT && holdAt && Math.hypot(e.clientX - holdAt.x, e.clientY - holdAt.y) > SLOP) { clearTimeout(holdT); holdT = null; holdAt = null; }
          }, true);
          const endMouse = () => endHold("mouse"), endAny = () => endHold(null);
          window.addEventListener("pointerup", endMouse, true);
          window.addEventListener("pointercancel", endMouse, true);
          window.addEventListener("blur", endAny);
          // the space bar: down starts a hold at the pointer (or the map's
          // center before the pointer has been over it), up ends it
          function spaceDown() {
            if (st.holding || !map) return;
            const r = mapEl.getBoundingClientRect();
            const inMap = lastPt && lastPt.x >= r.left && lastPt.x <= r.right && lastPt.y >= r.top && lastPt.y <= r.bottom;
            const pt = inMap ? lastPt : {x: r.left + r.width / 2, y: r.top + r.height / 2};
            beginHold(pt.x, pt.y, "key");
          }
          const onKeyUp = (e) => { if (e.key === " ") endHold("key"); };
          window.addEventListener("keyup", onKeyUp);
          root.addEventListener("wheel", (e) => {
            if (!st.holding) return;
            e.preventDefault(); e.stopPropagation();
            wheelAcc += e.deltaY;
            const now = performance.now();
            if (Math.abs(wheelAcc) >= 40 && now - lastStep > 140) { stepImg(wheelAcc > 0 ? 1 : -1); wheelAcc = 0; lastStep = now; }
          }, {capture: true, passive: false});
          root.addEventListener("pointermove", (e) => { lastPt = {x: e.clientX, y: e.clientY}; }, true);
          function showHexTip(pt) {
            const r = mapEl.getBoundingClientRect();
            const i = hexAt(map.unproject([pt.x - r.left, pt.y - r.top]));
            if (i !== hover) { hover = i; update(); }
            if (i < 0 || !hattrs || map.getZoom() < HEXZ) { tip.style.display = "none"; return; }
            const p = pane.getBoundingClientRect();
            tip.innerHTML = hexWords(i);
            tip.style.display = "block";
            tip.style.left = (pt.x - p.left + 14) + "px";
            tip.style.top = (pt.y - p.top + 14) + "px";
          }

          // ---- search ------------------------------------------------------------------------
          const PHOTON = "https://photon.komoot.io/api/";
          let gcHits = [], gcSel = -1, gcTimer = null, gcSeq = 0, searched = null;
          // an H3 string in the box is a cell, not a place (Stephen, 2026-09-25)
          const h3Of = (q) => { const h = q.trim().toLowerCase(); try { return /^[0-9a-f]{15}$/.test(h) && isValidCell(h) ? h : null; } catch (e) { return null; } };
          const hitName = (f) => { if (f.h3) return "H3 " + f.h3; const p = f.properties || {}; return [p.name, p.street && !p.name ? p.street : null, p.city && p.city !== p.name ? p.city : null, p.state, p.country].filter(Boolean).join(", "); };
          const hitKind = (f) => { if (f.h3) { const [la, lo] = cellToLatLng(f.h3); return `res ${getResolution(f.h3)}, ${la.toFixed(5)}, ${lo.toFixed(5)}`; } const p = f.properties || {}; return [p.osm_value, p.type].filter((x) => x && x !== "yes").join(", "); };
          const gcHide = () => { hits.style.display = "none"; hits.replaceChildren(); gcSel = -1; };
          const gcShow = () => {
            hits.replaceChildren();
            if (!gcHits.length) { gcHide(); return; }
            gcHits.forEach((f, i) => { const r = el_("div", "at-hit" + (i === gcSel ? " sel" : "")); r.textContent = hitName(f); const k = el_("small"); k.textContent = hitKind(f); r.appendChild(k); r.onmousedown = (e) => { e.preventDefault(); gcFly(f); }; r.onmouseenter = () => { gcSel = i; gcShow(); }; hits.appendChild(r); });
            hits.style.display = "block";
          };
          const gcAsk = async () => {
            const q = gc.value.trim();
            if (!q && searched) { searched = null; update(); }
            if (q.length < 2) { gcHits = []; gcHide(); return; }
            const s = ++gcSeq;
            const h3 = h3Of(q);
            if (h3) { gcHits = [{h3}]; gcSel = 0; gcShow(); return; }
            const params = new URLSearchParams({q, limit: "6", lang: "en"});
            if (map) { const c = map.getCenter(); params.set("lon", c.lng.toFixed(4)); params.set("lat", c.lat.toFixed(4)); }
            try { const r = await fetch(PHOTON + "?" + params.toString()); const d = await r.json(); if (s !== gcSeq) return; gcHits = (d.features || []).filter((f) => f.geometry && f.geometry.coordinates); gcSel = gcHits.length ? 0 : -1; gcShow(); }
            catch (e) { if (s === gcSeq) note("search: " + e.message, 4000); }
          };
          const gcFly = (f) => {
            if (f.h3) {
              // zoomed so the cell is about 80 px across, never out past the hexagons
              const [lat, lon] = cellToLatLng(f.h3), edge = 1281256 / Math.pow(Math.sqrt(7), getResolution(f.h3));
              const zoom = Math.max(HEXZ, Math.min(17, Math.log2(78271.5 * Math.cos(lat * Math.PI / 180) * 80 / (2 * edge))));
              searched = f.h3; gcHits = []; gcHide(); gc.blur(); update();
              if (map) map.flyTo({center: [lon, lat], zoom, duration: 2200, essential: true});
              return;
            }
            searched = null;
            const [lon, lat] = f.geometry.coordinates;
            const ext = (f.properties || {}).extent;
            let zoom = 12;
            if (ext && ext.length === 4) { const span = Math.max(Math.abs(ext[2] - ext[0]), Math.abs(ext[1] - ext[3]) * 2, 0.01); zoom = Math.log2(360 * ((mapEl.clientWidth || 1200) / 512) / span) - 0.3; }
            zoom = Math.max(HEXZ, Math.min(16, zoom));
            gc.value = hitName(f); gcHits = []; gcHide(); gc.blur();
            if (map) map.flyTo({center: [lon, lat], zoom, duration: 2200, essential: true});
          };
          gc.addEventListener("input", () => { clearTimeout(gcTimer); gcTimer = setTimeout(gcAsk, 250); });
          gc.addEventListener("focus", () => { if (gcHits.length) gcShow(); });
          gc.addEventListener("blur", () => setTimeout(gcHide, 120));
          gc.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "ArrowDown" && gcHits.length) { gcSel = (gcSel + 1) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "ArrowUp" && gcHits.length) { gcSel = (gcSel - 1 + gcHits.length) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "Enter") { e.preventDefault(); if (gcHits.length) gcFly(gcHits[Math.max(0, gcSel)]); else { clearTimeout(gcTimer); gcAsk().then(() => { if (gcHits.length) gcFly(gcHits[0]); else note("no match: " + gc.value.trim(), 4000); }); } }
            else if (e.key === "Escape") { gcHide(); gc.blur(); }
          });

          // ---- fill the window --------------------------------------------------------------
          const FIT_CLS = "at-fit-on";
          if (!document.getElementById("at-fit-style")) {
            const s = document.createElement("style"); s.id = "at-fit-style";
            s.textContent = ["notebook-actions-dropdown", "cell-actions-button", "drag-button", "expand-output-button", "fullscreen-output-button", "chrome-sidebar", "chrome-footer", "chrome-controls-top-right", "chrome-controls-bottom-right"].map((t) => "html." + FIT_CLS + " [data-testid='" + t + "']").join(",") + ",html." + FIT_CLS + " div[class*='top-[25vh]']{display:none!important}html." + FIT_CLS + "{overflow:hidden}";
            document.head.appendChild(s);
          }
          function sizes() {
            root.classList.toggle("fit", st.fit);
            document.documentElement.classList.toggle(FIT_CLS, st.fit);
            pane.style.height = st.fit ? "100vh" : (cfg.height || 780) + "px";
            bFit.innerHTML = st.fit ? ICON.shrink : ICON.expand; bFit.title = st.fit ? "back to the notebook (X or Esc)" : "fill the window (X)";
            setTimeout(() => { try { map && map.resize(); } catch (e) {} }, 30);
          }
          bFit.onclick = () => { st.fit = !st.fit; sizes(); };
          bMore.onclick = (e) => { e.stopPropagation(); const open = more.style.display !== "block"; more.style.display = open ? "block" : "none"; yc.style.visibility = open ? "hidden" : ""; bMore.classList.toggle("on", open); };
          more.addEventListener("click", (e) => e.stopPropagation());
          const closeMore = () => { more.style.display = "none"; yc.style.visibility = ""; bMore.classList.remove("on"); };
          root.addEventListener("click", () => { if (more.style.display === "block") closeMore(); });
          window.addEventListener("resize", () => { if (st.fit) sizes(); });

          // ---- keys ---------------------------------------------------------------------------
          root.tabIndex = 0;
          const onKey = (e) => {
            const path = e.composedPath ? e.composedPath() : [];
            if (!st.fit && !path.includes(root)) return;
            const tgt = path[0] || e.target;
            if (tgt && /^(INPUT|SELECT|TEXTAREA)$/.test(tgt.tagName)) return;
            const k = e.key, lo = st.y0, hi = st.y1;
            if (k === " ") { if (!e.repeat) spaceDown(); }
            else if (k === "s" || k === "S" || k === "d" || k === "D") { const m = (k === "s" || k === "S") ? "much" : "year"; if (m !== st.gmode) { st.gmode = m; recolorHex(); styleRows(); update(); } }
            else if (k === "[" || k === "]") stepImg(k === "]" ? 1 : -1);
            else if (k === ";" || k === "'") { st.s2scale = Math.round(10 * Math.max(0.3, Math.min(2.5, st.s2scale + (k === "'" ? 0.1 : -0.1)))) / 10; gam.value = st.s2scale; clearTimeout(gamT); gamT = setTimeout(() => send("s2scale"), 250); }
            else if (k === "-" || k === "=") { const v = Math.max(aefYears[0], Math.min(hi - 1, lo + (k === "=" ? 1 : -1))); if (v !== lo) { st.y0 = v; winSent = [st.y0, st.y1]; styleWin(); send("aef"); } }
            else if (k === "_" || k === "+") { const v = Math.max(lo + 1, Math.min(aefYears[aefYears.length - 1], hi + (k === "+" ? 1 : -1))); if (v !== hi) { st.y1 = v; winSent = [st.y0, st.y1]; styleWin(); send("aef"); } }
            else if (k === "l" || k === "L") { st.labels = !st.labels; labels(st.labels); swLab.sty(); }
            else if (k === "x" || k === "X") { st.fit = !st.fit; sizes(); }
            else if (k === "/") gc.focus();
            else if (k === "Escape") { if (about.style.display === "flex") about.style.display = "none"; else if (more.style.display === "block") closeMore(); else if (cardData) closeCard(); else if (st.fit) { st.fit = false; sizes(); } }
            else return;
            e.preventDefault();
          };
          window.addEventListener("keydown", onKey);

          // ---- camera and click ----------------------------------------------------------------
          let seq = 0, lastView = "";
          function sendView() {
            if (!map) return;
            const c = map.getCenter();
            const v = {longitude: c.lng, latitude: c.lat, zoom: map.getZoom(), w: mapEl.clientWidth, h: mapEl.clientHeight};
            const key = JSON.stringify(v);
            if (key === lastView) return;
            lastView = key; v.n = ++seq;
            model.set("view", JSON.stringify(v)); model.save_changes();
          }
          const adminAt = (pt) => {
            const out = {};
            const one = (k) => { const id = "ov-div-" + k; if (!map.getLayer(id)) return null; const fs = map.queryRenderedFeatures(pt, {layers: [id]}); return fs && fs.length ? fs[0].properties : null; };
            try { const r = one("region"); if (r) out.region = r["@name"] || r.names || null; const c = one("county"); if (c) out.county = c["@name"] || c.names || null; const l = one("locality"); if (l) out.locality = l["@name"] || l.names || null; } catch (e) {}
            return out;
          };
          function boot() {
            const home = cfg.home || {longitude: 3.6, latitude: 6.46, zoom: 11};
            map = new maplibregl.Map({container: mapEl, style: STYLE, center: [home.longitude, home.latitude], zoom: home.zoom, attributionControl: {compact: true}});
            map.keyboard.disable();
            map.doubleClickZoom.enable();
            map.addControl(new maplibregl.NavigationControl({showCompass: false}), "bottom-left");
            ov = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck: " + (e && e.message ? e.message : e))});
            map.addControl(ov);
            map.on("load", () => {
              labels(st.labels);
              if (cfg.div_pm && !map.getSource("ov-div")) {
                try {
                  map.addSource("ov-div", {type: "vector", url: "pmtiles://" + cfg.div_pm});
                  for (const k of ["region", "county", "locality"]) map.addLayer({id: "ov-div-" + k, type: "fill", source: "ov-div", "source-layer": "division_area", filter: ["all", ["==", ["get", "subtype"], k], ["==", ["get", "class"], "land"]], paint: {"fill-opacity": 0}}, slot());
                } catch (e) { console.error("divisions", e); }
              }
              update(); sendView(); renderYear();
            });
            map.on("moveend", sendView);
            map.on("zoomend", () => { update(); renderYear(); styleKey(); });
            map.on("mousemove", (e) => {
              if (holdT) return;
              // over the imagery: the white outline follows the pointer, no tooltip
              if (st.holding) { const i = hexAt(e.lngLat); if (i !== hover) { hover = i; update(); } return; }
              showHexTip({x: e.originalEvent.clientX, y: e.originalEvent.clientY});
            });
            map.on("mouseout", () => { tip.style.display = "none"; if (hover != null && hover >= 0) { hover = null; update(); } });
            map.on("click", (e) => {
              if (suppressClick) { suppressClick = false; return; }
              const i = hexAt(e.lngLat);
              pickCell(i >= 0 ? hexes[i] : null, e.lngLat, e.point, st.holding);
            });
            map.on("error", (ev) => { if (ev && ev.error && ev.error.message && !/tile|404/i.test(ev.error.message)) say("map: " + ev.error.message); });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} fitCard(); }).observe(mapEl);
            window.__cmMaps = () => [map];
            window.__cmState = () => ({st: Object.assign({}, st), hex: N, res, hmeta, tiles: tstat, card: cardData, status: model.get("status")});
            window.__cmTiles = () => ({log: tlog, paints: ptimes, frames: flog});
            // for tests: the center of the first hexagon whose biggest step is year y and that moved a fair amount
            window.__cmHexAt = (y) => { const b = map.getBounds(); for (let i = 0; i < N; i++) if (hattrs && hattrs[4 * i] === y - 2000 && hattrs[4 * i + 1] >= FAIR) { const r = cellToBoundary(hexes[i], true); const c = r.slice(0, -1).reduce((a, p) => [a[0] + p[0] / (r.length - 1), a[1] + p[1] / (r.length - 1)], [0, 0]); if (b.contains(c) && map.project(c).x < mapEl.clientWidth - 360 && map.project(c).y > 200) return c; } return null; };
          }

          // ---- the kernel's data -----------------------------------------------------------------
          const flog = [];  // per frame received, for the tests
          const loadHex = () => {
            const tl = performance.now();
            const cb = bytesOf(model.get("cells")), ab = bytesOf(model.get("hattrs"));
            try { hmeta = JSON.parse(model.get("hmeta") || "{}"); } catch (e) { hmeta = {}; }
            if (!cb || !cb.length) { hexes = []; N = 0; hexIndex = new Map(); res = -1; hattrs = null; hcol = null; hcol32 = null; renderYear(); styleKey(); update(); return; }
            const ids = new BigUint64Array(copyOf(cb));
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            try { res = getResolution(hexes[0]); } catch (e) { res = -1; }
            hattrs = ab && ab.length === 4 * N ? new Uint8Array(copyOf(ab)) : null;
            hover = null;
            recolorHex(); renderYear(); styleKey(); update();
            flog.push({seq: hmeta.seq, n: N, t: Date.now(), ms: performance.now() - tl});
          };
          let pendHex = null;
          const hexSoon = () => { clearTimeout(pendHex); pendHex = setTimeout(loadHex, 0); };
          model.on("change:cells", hexSoon);
          model.on("change:hattrs", hexSoon);
          model.on("change:hmeta", hexSoon);
          model.on("change:card", renderCard);
          model.on("change:status", () => say(model.get("status")));
          model.on("change:config", () => { try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; } update(); });
          try {
            sizes(); boot(); styleRows(); loadHex(); renderCard(); say(model.get("status"));
          } catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("keyup", onKeyUp); window.removeEventListener("pointerup", endMouse, true); window.removeEventListener("pointercancel", endMouse, true); window.removeEventListener("blur", endAny); document.documentElement.classList.remove(FIT_CLS); try { map && map.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (ChangeMap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How it works

    **AlphaEarth, folded to H3.** Every 10 m pixel of the AlphaEarth
    Foundations embedding is 64 numbers describing the ground for one year.
    For the view on screen, the notebook reads each year in the window
    (2021 to 2025 by default, 2017 to 2025 available) from Source
    Cooperative: the COG overviews at coarser hexagons, the zarr mosaic
    from res 11 in. Each pixel's lon/lat goes through an h3ronpy UDF inside
    DataFusion (via xarray-sql), and the pixels are averaged per cell,
    one fold per year. The hexagon size follows the zoom.

    **The brightest patch, not the average.** The earlier notebooks (and
    this one at first) averaged every pixel in a hexagon into one vector a
    year and measured the change of that average. A small site that changed
    a lot, inside a hexagon of quiet ground, was averaged away: it read dark
    zoomed out and only lit up zoomed in. Here it runs in three steps:

    1. The fold runs one H3 level finer than the hexagons on screen (res 9
       cells under res 8 hexagons at zoom 9), about one pixel of the read
       per finer cell. Averaging only happens inside a finer cell.
    2. The change is worked out per finer cell, from that cell's own
       vectors across the years.
    3. h3ronpy's `change_resolution` gives each finer cell its parent
       hexagon, and each hexagon takes its most-changed finer cell whole:
       its change, its year, its steps.

    So a hexagon shows its strongest spot, not its average, and one changed
    site keeps a big hexagon bright zoomed out. `finer_cells` in the table
    under the map is how many finer cells each hexagon picked from.

    **Drawn as tiles.** The hexagons reach the browser as map tiles in
    which each pixel names the hexagons near it. The shader draws each
    hexagon's edge from its H3 boundary (h3-js), covering a fragment by its
    distance to the edge over one screen pixel, so edges stay smooth at any
    zoom. Colors come from a small table, so switching between AEF Change
    and AEF Change Year recolors without new tiles. Hover and click look
    the hexagon up from the pointer with h3-js. The browser draws images,
    however many hexagons there are.

    **AEF Change (viridis, `S`).** Each finer cell's yearly vector is
    normalized, and `disp` is 1 minus the cosine between the window's first
    and last year: 0 means the fingerprint did not move. A hexagon's `disp`
    is its most-changed finer cell's. The fill stretches
    `disp` to this view's 2nd to 98th percentile, so the colors rank the
    hexagons against their neighbors, not against the world. Hexagons
    that barely moved are drawn faint.

    **AEF Change Year (YlOrBr, `D`).** Every year-to-year step is scored
    the same way. The embeddings drift as a whole between some years (over
    Lagos the 2024 to 2025 median step is about twice the others), so the
    raw biggest step would land on 2025 almost everywhere. Each step is
    divided by that year's median step in view instead, and the change year
    is the step that stands out most. The card on a clicked hexagon shows
    those ratios against 1.

    **What is there (ESA WorldCover 2021).** The same fold, on the class
    raster read straight from ESA's bucket: a count of pixels per class per
    hexagon, shown as shares. It is one year only, so it describes the
    ground; it does not date anything.

    **Zoomed out (below zoom 9).** The map is WorldCover itself, drawn as
    tiles from the same COGs (the coarsest overview that still fills each
    tile), in a palette without red: built-up deep violet, cropland gold,
    vegetation in greens, water blue. It shows where the towns, farmland and
    water are; from zoom 9 the AlphaEarth change hexagons take over.

    **What happened (Sentinel-2).** Holding space swaps the hexagons for
    Earth Genome's yearly true-color mosaic, 2022 to 2025, so the change
    the hexagons point to can be checked against the imagery. Nothing
    colored is drawn over it, only the outlines of the hovered, picked and
    searched hexagons.
    """)
    return


@app.cell
def _(
    AEF_FROM0,
    AEF_TO0,
    AEF_YEARS_ALL,
    ALPHA_FILL,
    ALPHA_QUIET,
    ChangeMap,
    HEX_ZOOM,
    HOLD_MS,
    HOLD_SLOP_PX,
    HOME,
    LABELS_SLOT,
    OV_DIV_PM,
    RASTER_TILE,
    S2_SCALE0,
    S2_TILE_MIN_Z,
    S2_YEAR0,
    S2_YEARS,
    VIEW_H,
    VIRIDIS,
    WC_CLASSES,
    WC_TILE_COLORS,
    WC_TILE_MIN_Z,
    json,
    mo,
):
    # ---- the map: built ONCE, empty; never re-runs for a parameter ---------------
    # as an app (marimo run) it fills the window from the start; in the editor
    # it sits in the page (X fills the window, Esc brings it back)
    try:
        _fit = mo.app_meta().mode == "run"
    except Exception:
        _fit = False
    cmap = ChangeMap(config=json.dumps({
        "height": VIEW_H, "home": dict(HOME), "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
        "s2_year": S2_YEAR0, "s2_scale": S2_SCALE0, "s2_gen": 0, "s2_years": list(S2_YEARS), "s2_min_z": S2_TILE_MIN_Z,
        "aef_from": AEF_FROM0, "aef_to": AEF_TO0, "aef_years": list(AEF_YEARS_ALL),
        "hex_zoom": HEX_ZOOM, "div_pm": OV_DIV_PM, "fit": _fit, "hold_ms": HOLD_MS, "hold_slop": HOLD_SLOP_PX,
        "viridis": VIRIDIS, "alpha_fill": ALPHA_FILL, "alpha_quiet": ALPHA_QUIET,
        "wc_min_z": WC_TILE_MIN_Z, "wc_key": [[_nm, WC_TILE_COLORS[_c]] for _c, _nm in WC_CLASSES if _c in (50, 40, 10, 30, 80)],
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "pending_force": False, "task": None, "loop": None,
        "s2scale": S2_SCALE0, "s2gen": 0, "y0": AEF_FROM0, "y1": AEF_TO0,
        "hit": None, "pick_n": None, "card": None, "memo": {}, "aef": {}, "wc": {},
        "h_cam": None, "h_ctl": None, "h_pick": None, "runs": 0, "hex_status": "", "place": None,
    }
    cmap
    return HOLD, cmap


@app.cell
def _(
    AEF_YEARS_ALL,
    CARRY_RES,
    CELL_KM2,
    HEX_TILE_PX,
    HEX_UP,
    HEX_ZOOM,
    HOLD,
    HOME,
    SETTLE,
    WC_CLASSES,
    WC_CODES,
    aef_fold,
    asyncio,
    build_frame,
    cmap,
    contains,
    coordinates_to_cells,
    cpu,
    division_at,
    json,
    np,
    pa,
    pad_box,
    re,
    res_for_view,
    s2_set_scale,
    s2_tile_png,
    time,
    traceback,
    view_to_bbox,
    wc_fold,
    wc_tile_png,
    zlib,
):
    # ---- wiring: the camera loop, the click and the controls. Re-runs freely. -----
    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass
    HOLD["runs"] += 1
    _WC_NAME = dict(WC_CLASSES)

    def _hex_tile(fr, z, x, y):
        """A map tile of the frame's hexagons as cell numbers: HEX_TILE_PX a
        side, each pixel 1 + the frame row of the hexagon its center falls in
        (0 none), uint32 little-endian, deflated. The browser colors it."""
        T, n = HEX_TILE_PX, 2 ** z
        f = (np.arange(T) + 0.5) / T
        lon = (x + f) / n * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + f) / n))))
        LON, LAT = np.meshgrid(lon, lat)
        c = pa.array(coordinates_to_cells(LAT.ravel(), LON.ravel(), fr["res"])).to_numpy(zero_copy_only=False).astype(np.uint64)
        ids = fr["cellid"]
        if not len(ids):
            return None
        i = np.clip(np.searchsorted(ids, c), 0, len(ids) - 1)
        hit = ids[i] == c
        if not hit.any():
            return None
        return zlib.compress(np.where(hit, i + 1, 0).astype("<u4").tobytes(), 1)

    async def _tile_fn(src, z, x, y, year):
        t0 = time.time()
        if src == "wc":
            out = await wc_tile_png(z, x, y)
        elif src == "hex":
            fr = HOLD["frame"]
            if fr is None or fr.get("seq") != year:
                raise RuntimeError("stale hexagon frame")
            ts = {}

            def _job():
                ts["s"] = time.time()
                r = _hex_tile(fr, z, x, y)
                ts["e"] = time.time()
                return r

            out = await cpu(_job)
            cmap.tile_times[(src, z, x, y, year)] = {"wait": 1e3 * (ts["s"] - t0), "run": 1e3 * (ts["e"] - ts["s"])}
            return out
        else:
            out = await s2_tile_png(z, x, y, year)
        cmap.tile_times[(src, z, x, y, year)] = {"run": 1e3 * (time.time() - t0)}
        return out

    cmap.tile_fn = _tile_fn

    def _say(msg):
        # an unchanged status must still register in the browser, so a
        # zero-width space toggles on repeats
        try:
            if cmap.status == msg:
                msg = msg + "​" if not msg.endswith("​") else msg[:-1]
            cmap.status = msg
        except Exception:
            pass

    def _cfg(**kw):
        c = json.loads(cmap.config or "{}")
        c.update(kw)
        cmap.config = json.dumps(c)

    def _spawn(coro):
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

    def _vsd(vs):
        if vs is None:
            return dict(HOME)
        if isinstance(vs, str):
            try:
                vs = json.loads(vs)
            except Exception:
                return dict(HOME)
        out = {"longitude": float(vs["longitude"]), "latitude": float(vs["latitude"]), "zoom": float(vs["zoom"])}
        if vs.get("w") and vs.get("h"):
            out["w"], out["h"] = float(vs["w"]), float(vs["h"])
        return out

    # ---- the hexagons -------------------------------------------------------------
    def _paint():
        """Send the frame once: 4 bytes per hexagon, colored in the browser."""
        fr = HOLD["frame"]
        if fr is None or HOLD["sent"] is fr:
            return
        big, lv = fr["big"], fr["level"]
        yc = np.where(big > 0, big - 2000, 0).astype(np.uint8)
        lb = np.where(np.isnan(lv), 0, 1 + np.round(254 * np.nan_to_num(lv))).astype(np.uint8)
        tc = np.where(fr["top"] >= 0, fr["top"] + 1, 0).astype(np.uint8)
        ts = np.round(255 * np.clip(fr["top_share"], 0, 1)).astype(np.uint8)
        with cmap.hold_sync():
            cmap.cells = fr["cellid"].astype("<u8").tobytes()
            cmap.hattrs = np.ascontiguousarray(np.stack([yc, lb, tc, ts], 1)).tobytes()
            cmap.hmeta = json.dumps({
                "y0": int(fr["years"][0]), "y1": int(fr["years"][-1]), "km2": float(CELL_KM2.get(HOLD["res"], 0)),
                "seq": int(fr.get("seq", 0)), "carry": CARRY_RES, "timing": fr.get("timing"),
                "classes": [_WC_NAME[c] for c in WC_CODES],
            })
        HOLD["sent"] = fr

    async def _serve_hex(vsd, force=False):
        view = view_to_bbox(vsd)
        box = pad_box(view)
        fr0 = HOLD["frame"]
        if (fr0 is not None and HOLD["box"] is not None and contains(HOLD["box"], view) and not force
                and min(15, res_for_view(vsd, box) + HEX_UP) <= HOLD["res"] and (fr0["y0"], fr0["y1"]) == (HOLD["y0"], HOLD["y1"])):
            HOLD["hex_status"] = HOLD.get("hex_ready") or HOLD["hex_status"]
            return
        rres = res_for_view(vsd, box)
        res = min(15, rres + HEX_UP)
        fres = min(15, res + CARRY_RES)
        y0, y1 = HOLD["y0"], HOLD["y1"]
        rbox = tuple(round(v, 3) for v in box)
        key = (y0, y1, res, rbox)
        t0 = time.time()
        years = list(range(y0, y1 + 1))
        HOLD["hex_status"] = f"folding AlphaEarth {y0} to {y1} ({len(years)} years) and the land cover…"
        _say(HOLD["hex_status"])
        if key in HOLD["memo"]:
            fr, stats = HOLD["memo"][key]
        else:
            bkey = (res, rbox)
            need = [y for y in years if (y, bkey) not in HOLD["aef"]]
            wneed = bkey not in HOLD["wc"]
            got = await asyncio.gather(
                wc_fold(box, res) if wneed else asyncio.sleep(0, result=HOLD["wc"].get(bkey)),
                *(aef_fold(box, fres, y, read_res=rres) for y in need),
            )
            if wneed:
                HOLD["wc"][bkey] = got[0]
            for y, r in zip(need, got[1:]):
                HOLD["aef"][(y, bkey)] = r
            for k_ in ("aef", "wc"):
                while len(HOLD[k_]) > 40:
                    HOLD[k_].pop(next(iter(HOLD[k_])))
            wc, s_wc = HOLD["wc"][bkey]
            aef_by_year = {y: HOLD["aef"][(y, bkey)][0] for y in years if (y, bkey) in HOLD["aef"]}
            t1 = time.time()
            fr = await cpu(build_frame, aef_by_year, wc, y0, y1, res)
            if fr is not None:
                fr["timing"] = {
                    "reads": 1e3 * (t1 - t0), "frame": 1e3 * (time.time() - t1),
                    "aef": [HOLD["aef"][(y, bkey)][1] for y in years if (y, bkey) in HOLD["aef"]], "wc": s_wc,
                    "t_frame": time.time(),
                }
            if fr is None:
                HOLD["hex_status"] = f"hexagons: res {res}, AlphaEarth has fewer than two years here | " + " | ".join(HOLD["aef"][(y, bkey)][1] for y in years if (y, bkey) in HOLD["aef"])
                return
            HOLD["fseq"] = HOLD.get("fseq", 0) + 1
            fr["seq"] = HOLD["fseq"]
            # each year's AlphaEarth read and fold, "c" where it came from memory
            _rd = []
            for y in years:
                if (y, bkey) not in HOLD["aef"]:
                    continue
                m_ = re.search(r"([\d.]+) s · fold [\d,]+ ([\d.]+) s", HOLD["aef"][(y, bkey)][1] or "")
                _rd.append(f"{y} c" if y not in need else f"{y} {m_.group(1)}+{m_.group(2)} s" if m_ else f"{y} ?")
            stats = (
                f"read res {rres}, hexagons res {res}, peak of res {fres} | AEF read+fold {', '.join(_rd)} "
                f"(all {t1 - t0:.1f} s) | {s_wc} | frame {time.time() - t1:.1f} s"
            )
            HOLD["memo"][key] = (fr, stats)
            while len(HOLD["memo"]) > 12:
                HOLD["memo"].pop(next(iter(HOLD["memo"])))
        HOLD["frame"], HOLD["box"], HOLD["res"] = fr, box, res
        _paint()
        HOLD["hex_status"] = HOLD["hex_ready"] = f"hexagons: {stats} | {fr['score']} | {time.time() - t0:.1f} s"
        if HOLD.get("card_pick"):
            _card_send(HOLD["card_pick"])

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        if vsd["zoom"] < HEX_ZOOM:
            # the frame stays (hidden in the browser, its tiles cached there),
            # so zooming back in over the same ground is instant
            HOLD["hex_status"] = f"hexagons from zoom {HEX_ZOOM:g}"
        else:
            await _serve_hex(vsd, force)
        _say(HOLD["hex_status"])

    async def refresh(vs, force=False, settle=True):
        """ONE serve at a time; the latest request wins while one is in flight."""
        if HOLD["busy"]:
            HOLD["pending"] = vs
            HOLD["pending_force"] = HOLD["pending_force"] or force
            return
        HOLD["busy"] = True
        try:
            while True:
                if settle:
                    await asyncio.sleep(SETTLE)
                if HOLD["pending"] is not None:
                    vs, HOLD["pending"] = HOLD["pending"], None
                    force, HOLD["pending_force"] = HOLD["pending_force"], False
                    settle = True
                    continue
                await _serve(vs, force)
                vs = HOLD["pending"]
                if vs is None:
                    return
                force, HOLD["pending"], HOLD["pending_force"] = HOLD["pending_force"], None, False
                settle = False
        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            _say(f"failed: {type(exc).__name__}: {exc}{where}")
            raise
        finally:
            HOLD["busy"], HOLD["pending"], HOLD["pending_force"] = False, None, False

    def _request(force=False):
        vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
        HOLD["task"] = _spawn(refresh(vs, force, settle=False))

    def _on_camera(change):
        vs = change["new"]
        if not vs:
            return
        HOLD["vs"] = vs
        HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_cam") is not None:
        try:
            cmap.unobserve(HOLD["h_cam"], names="view")
        except ValueError:
            pass
    cmap.observe(_on_camera, names="view")
    HOLD["h_cam"] = _on_camera

    # ---- the click: the hexagon's account, as JSON the browser lays out ----------
    def _hex_card(p):
        fr = HOLD["frame"]
        cellh = p.get("cell")
        if fr is None or not cellh:
            return None
        cell = np.uint64(int(cellh, 16))
        ids = fr["cellid"]
        i = int(np.searchsorted(ids, cell))
        if i >= len(ids) or ids[i] != cell:
            return {"kind": "note", "title": "That hexagon is not in the current view's frame."}
        lv = float(fr["level"][i])
        lc = []
        if fr["nwc"][i] > 0:
            sh = fr["share"][i]
            lc = [[_WC_NAME[WC_CODES[k]], float(sh[k])] for k in np.argsort(-sh) if sh[k] >= 0.01][:5]
        return {
            "kind": "hex", "cell": cellh, "level": None if np.isnan(lv) else lv, "big": int(fr["big"][i]),
            "y0": int(fr["years"][0]), "y1": int(fr["years"][-1]),
            "steps": [None if np.isnan(v) else float(v) for v in fr["steps"][:, i]],
            "rel": [None if np.isnan(v) else float(v) for v in fr["rel"][:, i]],
            "step_years": [int(b) for _, b in fr["step_years"]],
            "landcover": lc, "km2": float(CELL_KM2.get(HOLD["res"], 0)),
        }

    def _card_send(p):
        card = _hex_card(p)
        if not card:
            HOLD["card"], HOLD["card_pick"] = None, None
            cmap.card = ""
            return
        adm = p.get("admin") or {}
        got = HOLD.get("place") or {}
        card["place"] = got["levels"] if got.get("n") == p.get("n") else [{"name": x} for x in (adm.get("locality"), adm.get("county"), adm.get("region")) if x]
        card["n"] = p.get("n")
        HOLD["card"], HOLD["card_pick"] = card, p
        cmap.card = json.dumps(card)

    def _place_later(p):
        """The whole ladder of divisions under the click, from GeoParquet;
        the card is resent with it if the click is still the latest."""
        n = p.get("n")

        async def _later():
            try:
                d = await asyncio.to_thread(division_at, p["lon"], p["lat"])
            except Exception:
                return
            if HOLD.get("pick_n") != n or not d:
                return
            levels = []
            for lv in d:
                nm = lv.get("name_en") or lv.get("name")
                if not nm:
                    continue
                lt = lv.get("local_type")
                levels.append({"name": nm, "tag": lt if lt and lt != lv["subtype"] else lv["subtype"]})
            HOLD["place"] = {"n": n, "levels": levels}
            if HOLD.get("card_pick") is p:
                _card_send(p)

        _spawn(_later())

    def _on_pick(change):
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        try:
            HOLD["pick_n"] = p.get("n")
            if p.get("close"):
                HOLD["card"], HOLD["card_pick"] = None, None
                cmap.card = ""
                return
            _card_send(p)
            if HOLD.get("card") and p.get("lon") is not None:
                _place_later(p)
        except Exception as e:
            cmap.card = json.dumps({"kind": "note", "title": f"click: {type(e).__name__}: {e}"})


    if HOLD.get("h_pick") is not None:
        try:
            cmap.unobserve(HOLD["h_pick"], names="pick")
        except ValueError:
            pass
    cmap.observe(_on_pick, names="pick")
    HOLD["h_pick"] = _on_pick

    # ---- the controls -----------------------------------------------------------------
    def _on_ctl_body(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        act = c.get("act")
        if act == "s2scale":
            try:
                v = float(min(3.0, max(0.2, float(c.get("s2scale", HOLD["s2scale"])))))
            except (TypeError, ValueError):
                return
            if s2_set_scale(v):
                HOLD["s2scale"] = v
                HOLD["s2gen"] += 1
                _cfg(s2_scale=v, s2_gen=HOLD["s2gen"])
            return
        if act == "aef":
            a, b = int(c.get("y0", HOLD["y0"])), int(c.get("y1", HOLD["y1"]))
            if a in AEF_YEARS_ALL and b in AEF_YEARS_ALL and a < b and (a, b) != (HOLD["y0"], HOLD["y1"]):
                HOLD["y0"], HOLD["y1"] = a, b
                _cfg(aef_from=a, aef_to=b)
                _request(force=True)
            return

    def _on_ctl(change):
        try:
            _on_ctl_body(change)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            _say(f"control failed: {type(e).__name__}: {e}{where}")

    if HOLD.get("h_ctl") is not None:
        try:
            cmap.unobserve(HOLD["h_ctl"], names="ctl")
        except ValueError:
            pass
    cmap.observe(_on_ctl, names="ctl")
    HOLD["h_ctl"] = _on_ctl

    # the first fold waits for the browser's own view (its real size); a
    # re-run of this cell with a view already known serves it again
    if HOLD["frame"] is None and not HOLD["busy"]:
        if HOLD["vs"] is not None:
            _request()
    else:
        HOLD["sent"] = None
        _paint()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Under the map

    Press the button once the map has settled to query the current view's
    hexagons with DuckDB, one row per hexagon: `disp` (how far the
    AlphaEarth fingerprint moved between the first and last year read, 1
    minus the cosine), `level` (the same, stretched to this view's p2 to p98),
    `big_year` (the year whose step stands out most against that year's
    median step in view, -2 no data), `finer_cells` (how many finer cells
    it holds; the row's numbers are its most-changed one), one `step_YYYY` column per step and
    one `rel_YYYY` with that step over the year's median, `landcover` and `landcover_share` (the main
    ESA WorldCover 2021 class), and one `wc_` column per class with its share.
    """)
    return


@app.cell
def _(mo):
    tables_btn = mo.ui.run_button(label="table for the current view")
    tables_btn
    return (tables_btn,)


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None, mo.md("*no hexagons yet (zoom in past 9)*") if tables_btn.value else None)
    con.register("view_cells", HOLD["frame"]["cells"])
    view_table = mo.sql(
        """
        SELECT * FROM view_cells ORDER BY disp DESC NULLS LAST
        """,
        engine=con,
    )
    return


if __name__ == "__main__":
    app.run()
