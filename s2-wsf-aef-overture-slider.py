# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "datafusion>=54.0.0",
#     "xarray-sql>=0.3.3",
#     "xarray",
#     "zarr>=3",
#     "h3ronpy>=0.22.0",
#     "pyarrow>=25.0.0",
#     "obstore>=0.9.2",
#     "async-geotiff>=0.4",
#     "anywidget>=0.9",
#     "numpy",
#     "duckdb>=1.5.5",
#     "pyproj",
#     "pillow",
#     "shapely>=2.0",
#     "rasterio",
# ]
# ///


"""Overture buildings, one map, the other datasets as layers, S2 behind a slider.

The Overture building footprints are the map (from zoom 13, over a plain
vector basemap), each coloured by one thing at a time: the earliest
half-year WSF read the ground under it as built-up; the first year its own
AlphaEarth fingerprint jumped; the share of its pixels WSF reads as built;
the dataset it came from; the year that dataset last touched it. Layers
switched on and off on top or beneath: the WSF raster (built ground with
no footprint shows around them), the H3 hexagons of the settlement pair
(WSF and AlphaEarth over the year window), the Sentinel-2 mosaic.

Stephen, 2026-09-24: "These are the Overture footprints. And here are some
other data sets that could show us whether like when they were built, or
if there's structures that could be missing, we're kind of evaluating
Overture and we can use the source on top of that." And: "one map not
pair. toggle layers on and off."

Overture from Overture's own release bucket, PINNED (OV_PINNED), an
OVERTURE_RELEASE override, a fallback to the newest release, the per-file
bboxes from the release's STAC items (Overture's, then the Portolan mirror).
WSF, AlphaEarth and S2 stay on Source Cooperative. Doc: docs/16.

Run: uv run marimo edit s2-wsf-aef-overture-slider.py --sandbox
molab: https://molab.marimo.io/github/github.com/kentstephen/s2-wsf-aef-overture-pair/blob/main/s2-wsf-aef-overture-slider.py

Attribution: WSF Tracker (c) DLR and MindEarth, via source.coop
(mindearth/wsf, DOI 10.5281/zenodo.20424537). "The AlphaEarth Foundations
Satellite Embedding dataset is produced by Google and Google DeepMind" (CC
BY 4.0). Sentinel-2 yearly mosaics by Earth Genome (CC BY 4.0). Photon
(komoot) over OpenStreetMap data (ODbL). Overture Maps buildings and
divisions (ODbL). Basemap by Carto.
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
    import tempfile
    import time
    import traceback
    import urllib.parse
    import urllib.request

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
        tempfile,
        time,
        traceback,
        traitlets,
        udf,
        urllib,
        xr,
        zarr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/s2-wsf-aef-overture-pair/blob/main/s2-wsf-aef-overture-slider.py)
    <small>molab runs in the same region as the data and is the faster place to open this notebook.
    Locally: `uv run marimo edit s2-wsf-aef-overture-slider.py --sandbox`</small>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Overture buildings, with the other datasets as layers

    One map. **WSF** and **buildings** are on when it opens. From zoom 13 the
    Overture building footprints are drawn with a gold edge, each coloured by
    one thing you pick under **BUILDINGS** (key `Q` steps through them):

    - **WSF year**: the year WSF first read the ground under the footprint as
      built-up. Grey is already built when the record opens in 2016; near
      white is a footprint WSF has never read as built-up.
    - **AEF year**: the first year the footprint's own AlphaEarth fingerprint
      jumped past a quiet level set by the footprints that were standing the
      whole time. Read on demand for the view (all nine years), so the first
      time takes a moment. Same palette as WSF year: flip between the two and
      a footprint that changes colour is one the witnesses date differently.
    - **source**: the dataset Overture took the footprint from.

    **LAYERS**, listed top to bottom as drawn (keys `B` `W` `A` `S`):
    **buildings**; the **WSF** raster at 10 m, one colour per year of first
    detection, which shows built ground with no footprint around the
    buildings; **AEF**, the AlphaEarth embeddings folded to H3 hexagons from
    zoom 9, finer as you zoom (res 12 from zoom 14.6), with two fills, how
    much the fingerprint changed over the year window and the year it changed
    most, and the window slider; **S2**, the Sentinel-2 mosaic on the left of
    a divider you drag across the map, the data on its right, the footprints
    on both sides, with its year (`[` `]`, `F` for first or last) and a gamma
    slider. **VIEW** next to it switches between that divider (**slider**)
    and two maps side by side on one camera (**pair**: S2 alone on the left,
    the data on the right). **FIND** flies to a place. `X` fills the browser window, `Esc`
    brings it back.

    **Click** a footprint for its source, the date the source last touched
    it, the WSF pixels and first year under it, AlphaEarth's year once read,
    and the place from the same Overture release. Click between footprints
    for what WSF says about that ground. Click a hexagon for its own account.

    Overture is read from a pinned release on Overture's own bucket
    (override with `OVERTURE_RELEASE`; the status line names the release in
    play). WSF, AlphaEarth and S2 come from Source Cooperative.

    Attribution: WSF Tracker (c) DLR and MindEarth. AlphaEarth Foundations by
    Google and Google DeepMind (CC BY 4.0). Sentinel-2 mosaics by Earth
    Genome (CC BY 4.0). Overture Maps buildings and divisions (ODbL). Photon
    over OpenStreetMap (ODbL). Basemap by Carto.
    """)
    return


@app.cell
def _(os, tempfile):
    # ---- constants ----------------------------------------------------------
    # Each source offers what it has: Sentinel-2 yearly mosaics 2022-2025 (2025
    # is in the bucket, not yet in the STAC), AlphaEarth 2017-2025, WSF Tracker
    # half-years July 2016 .. January 2026. ONE window (from year, to year) is
    # read by both WSF and AlphaEarth; it opens at 2022..2025, the years the
    # S2 mosaics cover, so the picture on the left has a frame for every year
    # the window holds (Stephen, 2026-09-04).
    S2_YEARS = (2022, 2023, 2024, 2025)
    AEF_YEARS_ALL = tuple(range(2017, 2026))
    AEF_FROM0, AEF_TO0 = 2022, 2025
    S2_YEAR0 = 2022
    # the S2 mosaic's opening `scale`: a gain on the TCI bytes (1 = as served)
    S2_SCALE0 = 1.0

    # The zoom -> H3 ladder: BASE_RES at ZOOM0, one step finer every PER_RES zoom
    # units, clamped, then coarsened until the view's expected cell count fits
    # CELL_BUDGET. The pane is half the width of the old single map, so the same
    # zoom holds half the cells.
    # the settlement pair's ladder, kept as it was (Stephen, 2026-09-24: "I
    # liked it how it was set up in the initial pair notebook ... gradually
    # going up ... to 12"): res 8 at zoom 9, 9 at 10.4, 10 at 11.8, 11 at 13.2,
    # 12 from 14.6
    ZOOM0, PER_RES, BASE_RES = 6.2, 1.4, 6
    # res 12 (307 m2, ~3 WSF pixels, ~3 AEF mosaic pixels) is the top rung:
    # Stephen, 2026-09-02: "the detail from both these datasets could get us
    # to res 12 ... I want to try". The budget is what lets it in: zoom 14.6
    # holds ~108k res 12 cells in a padded pane, and the browser tessellates
    # them (highPrecision) in about a second.
    MIN_RES, MAX_RES = 5, 12
    CELL_BUDGET = 300_000
    MOSAIC_MIN_RES = 11
    AEF_LEVEL_FOR_RES = {5: 7, 6: 7, 7: 5, 8: 4, 9: 3, 10: 1}
    AEF_MAX_FILES = 2500

    S2_STAC = "https://stac.earthgenome.org/search"
    S2_COLLECTION = "sentinel2-yearly-mosaics"
    # Where the yearly mosaic is nodata, the same year's tile from Earth
    # Genome's OTHER composite fills the hole pixel by pixel, under the yearly:
    # `sentinel2-temporal-mosaics`, the SCL-masked yearly median (CC-BY 4.0,
    # 2022 and 2023 only on this STAC). Found 2026-09-03 over Nusantara: the
    # yearly 2022 tiles 50MME/50MMD hold 0 valid pixels over the city (a cloud
    # hole the size of Balikpapan Bay, good_pxl_pct 0.22), the temporal ones
    # 43% and 100%. The status line counts the filled pixels per year so the
    # reader knows which frames are a patchwork of two composites.
    S2_FILL_COLLECTION = "sentinel2-temporal-mosaics"
    # the mosaic pyramid ends at z9 (L5, 306 m); z7-8 are rendered from L5 by
    # decimation (a z7 tile reads up to nine 1024 px windows: slow, so no lower)
    S2_TILE_MIN_Z, S2_PYRAMID_Z, S2_TCI_MAX_Z = 7, 9, 14

    # ---- WSF Tracker: one GeoZarr (v3, sharded 8192 / 256 zstd), global 10 m --
    # `wsf_tracker` int8: 0 never built-up, k = 1..20 the half-year the pixel
    # first read as built-up. Index k's period ENDS at WSF_DATE[k]: 1 = already
    # built by 2016-07-01, 2 = July 2016 .. January 2017, 3 = January .. July
    # 2017, .. 20 = July 2025 .. January 2026. So the pixels built DURING
    # calendar year Y carry the two indices idx_h1(Y) = 2 (Y - 2016) + 1 and
    # idx_h2(Y) = idx_h1(Y) + 1 (for Y = 2016 the first of those is the
    # "already built" class). Levels 1..12 are a MIN pyramid over the nonzero
    # pixels (the earliest date under the window, a dilation of the built
    # share: measured 3.4% native vs 4.6% at level 2 over Chico) and are drawn,
    # never folded. The fold reads level 0 on a stride that fits WSF_MAX_PX
    # samples.
    WSF_BUCKET = "us-west-2.opendata.source.coop"
    # every S3 read (WSF, AlphaEarth COGs and mosaic, S2) from here to
    # us-west-2: a few requests in a batch stall until the client timeout and
    # are retried, and the default 30 s timeout made a 3 s AlphaEarth mosaic
    # read take 31 to 42 s (US East Coast, 2026-09-24: block median 1.5 s,
    # slowest good block 2.7 s). 6 s cuts a stall short and the retry lands
    S3_OPTS = {"timeout": "6s", "connect_timeout": "3s"}
    WSF_PREFIX = "mindearth/wsf/World_WSF_20160701-20260101.zarr"
    WSF_RES, WSF_X0, WSF_Y0 = 8.983152841195216e-05, -180.00001488697754, 78.0100585990529
    WSF_LEVELS = 13
    WSF_MAX_PX = 12_000_000
    WSF_NIDX = 20
    WSF_YEARS = tuple(range(2016, 2026))
    wsf_bounds = (-180.0, -60.01, 180.0, 78.01)

    VIEW_W, VIEW_H = 700, 720  # one pane
    # the strip under the map, minimal (Stephen, 2026-09-01): the legend and
    # the story stay; the status line (res, fold
    # timings, tile counts) and the keys hint are hidden. Flip to bring them
    # back; the kernel still writes them.
    STRIP_MINIMAL = True
    PAD = 1.3
    SETTLE = 0.35
    HEX_ZOOM = 9.0  # as the settlement pair
    LABELS_SLOT = "watername_ocean"
    RASTER_TILE = 256
    # home: Wuhan, where the Photon geocoder lands the user who types it
    # (Stephen, 2026-09-04): the hit's point and the zoom gcFly derives from
    # the city's extent at a 700 px pane, which is below HEX_ZOOM, so the right
    # pane opens on the WSF pyramid tiles and the hexagons fold on the first
    # zoom in. Earlier homes: Egypt's New Administrative Capital (31.75,
    # 30.01, zoom 10.2) and Paradise, California (-121.60, 39.76).
    HOME = {"longitude": 114.29, "latitude": 30.58, "zoom": 7.2}

    # a cell GREW when at least this share of its sampled pixels became
    # built-up inside the window; below it the cell counts as quiet for the
    # AlphaEarth baseline
    NEW_MIN = 0.01
    # the quiet level: D0 is the displacement quantile (1 - FA) of the view's
    # quiet cells (WSF saw no growth); a cell "moved" above it
    FA = 0.05
    MIN_STABLE_CELLS = 30
    # the keepers (Stephen, 2026-09-24: "keep the keepers, comment out the
    # rest for now"): the WSF fills repeat what the raster shows per pixel and
    # the click gives per cell
    FILLS = ("shift", "when")  # + "grew", "byear" (WSF grew, WSF build year: the raster has them)
    # FILLS = ("grew", "byear", "shift", "when")  # "built" is the WSF layer now, not a fill
    FILL_NAMES = {
        "built": "the WSF record itself: the half-year each 10 m pixel first read as built-up (the raster, not hexagons)",
        "grew": "share of the hexagon that became built-up inside the window (WSF)",
        "byear": "the year most of the new built-up ground arrived (WSF)",
        "shift": "how much the AlphaEarth fingerprint changed",
        "when": "the year the AlphaEarth fingerprint changed",
    }
    FILL_SHORT = {"built": "WSF built", "grew": "WSF grew", "byear": "WSF build year", "shift": "AEF changed", "when": "AEF change year"}
    ALPHA_FILL = 235
    ALPHA_QUIET = 70  # nothing built / never moved: drawn faintly so the grid stays legible
    VIRIDIS = "440154470d6048186a482374472e7c4538824241863e4a893a548c365d8d32658e2e6d8e2b758e287d8e25848e228c8d1f948c1e9c8920a38625ab822eb37c3aba7648c16e58c7656ccd5a7fd34e93d741a8db34c0df25d5e21aeae51afde725"
    # the grew fill: a warm lightness ramp (matplotlib YlOrBr less its white
    # end: the orange leg a protanope keeps, and not the blue of the basemap
    # water; Stephen, 2026-09-02: "blue is the wrong cmap for that"), zero
    # drawn quiet. "WSF built" is the raster (the year palette below), not a
    # hexagon fill ("wsf built should be the zarr buildings not h3").
    GREW_RAMP = ("#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506")

    # ---- Overture Maps: buildings and divisions from Overture's OWN bucket ----
    # (Stephen, 2026-09-24: "we dont need to use source for any overture now",
    # "i want to know if there's a way to keep a stable release"). Each
    # release lives under its own prefix and stays there, so the tag is the
    # pin. OV_PINNED is the release this notebook was built against; the
    # OVERTURE_RELEASE environment variable overrides it; if the pin is gone
    # from Overture's STAC the newest release is taken and the status line
    # says so. The STAC items (one per parquet file, with its bbox) are the
    # file index: Overture's canonical catalog first, the Portolan mirror
    # (nlebovits.github.io/overture-portolan, metadata only, same hrefs) as
    # the fallback. The PMTiles are the same release's, on Overture's extras
    # bucket: `building` layer to z14, `division_area` to z12.
    OV_PINNED = "2026-08-19.0"
    OV_STAC_ROOT = "https://stac.overturemaps.org/catalog.json"
    OV_PORTOLAN = "https://nlebovits.github.io/overture-portolan"
    OV_TILES = "https://overturemaps-extras-us-west-2.s3.us-west-2.amazonaws.com/tiles"
    # the footprints come in past this zoom (a 700 px pane is ~9.4 km wide
    # here; padded, ~160 km2, some 100k footprints in a dense city centre); a
    # view holding more than BLD_MAX rows is refused with "zoom in". Was 14
    # and 60k (Stephen, 2026-09-24: "we can load the buildings a little more
    # zoomed out").
    BLD_ZOOM = 13.0
    BLD_MAX = 150_000
    # ---- the buildings by themselves, coloured by what we know about them ----
    # (Stephen, 2026-09-24: "just visualize the buildings by themselves and
    # color by the information where we have with them"). Past BLD_ZOOM the
    # footprints are filled polygons on the right pane, one fill at a time:
    #   wyear   the earliest half-year WSF read the ground under the footprint
    #           as built-up (the year palette; 2016 = already built when the
    #           record opens; never = WSF has not seen it built)
    #   wshare  the share of the footprint's pixels WSF reads as built-up
    #   source  the dataset the footprint came from
    #   mapped  the year that source last touched the footprint
    # The WSF raster and the orange "built-up, no footprint" mark are off
    # here; they come back as overlays later, if they do.
    BFILLS = ("wyear", "ayear", "source")  # + "wshare", "mapped" (the click panel says both)
    # BFILLS = ("wyear", "ayear", "wshare", "source", "mapped")
    BFILL_SHORT = {"wyear": "WSF year", "ayear": "AEF year", "wshare": "WSF share", "source": "source", "mapped": "mapped"}
    BFILL_NAMES = {
        "wyear": "the earliest half-year WSF read the ground under the footprint as built-up",
        "ayear": "the first year the footprint's own AlphaEarth fingerprint jumped past the quiet level set by the pre-2016 footprints (read on demand, all years)",
        "wshare": "the share of the footprint's WSF pixels that read built-up (all_touched: a footprint claims every pixel it reaches)",
        "source": "the dataset Overture took the footprint from",
        "mapped": "the year that dataset last touched the footprint (a building cannot be newer than its map)",
    }
    BFILL0 = "wyear"
    NEVER_RGB = (240, 240, 240)  # a footprint WSF has never read as built-up: near white, the outline carries it
    # a single-hue blue ramp for the share (light to dark; luminance, not hue)
    BLUE_RAMP = ("#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594")
    # the sources: Okabe-Ito less the red, in order of how common each is in the view
    SOURCE_RGB = ((0, 114, 178), (230, 159, 0), (0, 158, 115), (204, 121, 167), (86, 180, 233), (240, 228, 66), (140, 86, 75), (45, 45, 45))
    BLD_STROKE = (255, 214, 120, 230)  # light gold: a footprint edge against the WSF raster under it
    # ONE map, the datasets as layers (Stephen, 2026-09-24: "We should just
    # have one map and look at the different data sets"). Which are on at
    # the start, and the switch labels.
    # WSF on first, the raster the viewer finds the buildings by (Stephen,
    # 2026-09-24: "WSF should just be on to help the viewer find the buildings")
    # listed top to bottom as drawn (Stephen, 2026-09-24: "buildings, WSF, AEF,
    # and then optionally the base maps"): footprints over the raster over the
    # hexagons, S2 the optional base under them all
    LAYERS0 = {"bld": True, "wsf": True, "hex": False, "s2": False}
    LAYER_DEFS = (("bld", "buildings", "Overture footprints, one fill at a time (from zoom 13)"),
                  ("wsf", "WSF", "the WSF Tracker raster: one colour per year of first detection"),
                  ("hex", "AEF", "the AlphaEarth embeddings (and WSF) folded to H3 hexagons over the year window (from zoom 9, finer as you zoom, res 12 from 14.6)"),
                  ("s2", "S2", "the Sentinel-2 yearly mosaic on the left of a divider you drag, the data on its right"))
    AEF_PREFIX = "tge-labs/aef-mosaic"
    AEF_RES, AEF_Y0, AEF_X0 = 8.983111749910169e-05, 83.68570533713473, -180.0
    AEF_NODATA = -128
    AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "aef-lcms")

    # ONE palette for a year, read by the WSF build-year fill, the AlphaEarth
    # change-year fill and the WSF pyramid tiles: Okabe-Ito less the red, plus
    # a teal, a brown and a near-black; 2016 (already built when the record
    # opens) mid grey. -1 built before the window / never moved, light grey;
    # -2 no embedding; -3 nothing built. No red, nothing hangs on red vs green.
    YEAR_RGB = {2016: (128, 128, 128), 2017: (0, 163, 152), 2018: (86, 180, 233), 2019: (0, 158, 115),
                2020: (240, 228, 66), 2021: (0, 114, 178), 2022: (230, 159, 0), 2023: (204, 121, 167),
                2024: (140, 86, 75), 2025: (45, 45, 45),
                -1: (222, 222, 222), -2: (150, 150, 150), -3: (236, 236, 236)}
    return (
        BFILL0,
        BFILLS,
        BFILL_NAMES,
        BFILL_SHORT,
        BLD_MAX,
        BLD_STROKE,
        BLD_ZOOM,
        BLUE_RAMP,
        LAYERS0,
        LAYER_DEFS,
        NEVER_RGB,
        SOURCE_RGB,
        OV_PINNED,
        OV_PORTOLAN,
        OV_STAC_ROOT,
        OV_TILES,
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
        CELL_BUDGET,
        FA,
        FILLS,
        FILL_NAMES,
        FILL_SHORT,
        GREW_RAMP,
        HEX_ZOOM,
        HOME,
        LABELS_SLOT,
        MAX_RES,
        MIN_RES,
        MIN_STABLE_CELLS,
        MOSAIC_MIN_RES,
        NEW_MIN,
        PAD,
        PER_RES,
        RASTER_TILE,
        S3_OPTS,
        S2_COLLECTION,
        S2_FILL_COLLECTION,
        S2_PYRAMID_Z,
        S2_SCALE0,
        S2_STAC,
        S2_TCI_MAX_Z,
        S2_TILE_MIN_Z,
        S2_YEAR0,
        S2_YEARS,
        SETTLE,
        STRIP_MINIMAL,
        VIEW_H,
        VIEW_W,
        VIRIDIS,
        WSF_BUCKET,
        WSF_LEVELS,
        WSF_MAX_PX,
        WSF_NIDX,
        WSF_PREFIX,
        WSF_RES,
        WSF_X0,
        WSF_Y0,
        YEAR_RGB,
        ZOOM0,
        wsf_bounds,
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
    Image,
    ObjectStore,
    RASTER_TILE,
    S3Store,
    S3_OPTS,
    WSF_BUCKET,
    WSF_LEVELS,
    WSF_MAX_PX,
    WSF_NIDX,
    WSF_PREFIX,
    WSF_RES,
    WSF_X0,
    WSF_Y0,
    YEAR_RGB,
    asyncio,
    cpu,
    ctx,
    io,
    itertools,
    math,
    np,
    time,
    xr,
    zarr,
):
    # ---- WSF Tracker: the GeoZarr, one fold per (box, res), tiles from the pyramid --
    # The grid is plate carree (EPSG:4326, WSF_RES degrees per pixel at every
    # latitude), so a lon/lat box IS a window: no projection leg. The fold
    # reads LEVEL 0 (the pyramid is a min over the nonzero pixels, a dilation,
    # and cannot be averaged) on a stride s so the window's samples fit
    # WSF_MAX_PX: a pixel every s in each axis, an unbiased sample of the
    # cell's share and date mix (measured: 308 Mpx at zoom 9 on a stride of 6
    # read in 1.0 s; the whole 308 MB in 0.8 s, the fold is what the budget
    # protects). The tiles (`wsf_tile_png`, the right pane below HEX_ZOOM) read
    # the level whose pixel is nearest the tile's own, which is what a browse
    # pyramid is for.
    _store = S3Store(WSF_BUCKET, region="us-west-2", skip_signature=True, prefix=WSF_PREFIX, client_options=S3_OPTS)
    _root = zarr.open_group(ObjectStore(_store, read_only=True), mode="r")
    _arr = {k: _root[str(k)]["wsf_tracker"] for k in range(WSF_LEVELS)}
    _win = {}
    _sem = asyncio.Semaphore(6)
    _seq = itertools.count()  # a table name per fold, so folds run side by side
    _png_cache = {}

    def idx_year(k):
        """The calendar year index k books its built-up date in (1 -> 2016)."""
        return 2016 + (int(k) - 1) // 2

    def idx_h1(y):
        return 2 * (int(y) - 2016) + 1

    def idx_h2(y):
        return 2 * (int(y) - 2016) + 2

    # the record is half-yearly but the story is yearly (Stephen, 2026-09-24):
    # index k reads as its calendar year, so 20 (July 2025 .. January 2026) is 2025
    WSF_DATE = ["never"] + [str(idx_year(k)) for k in range(1, WSF_NIDX + 1)]

    _cmap = np.zeros((256, 4), np.uint8)
    for _k in range(1, WSF_NIDX + 1):
        _cmap[_k, :3] = YEAR_RGB[idx_year(_k)]
        _cmap[_k, 3] = 255

    def _px(k):
        return WSF_RES * (2 ** k)

    def _window_ix(k, box):
        W_, S_, E_, N_ = box
        px = _px(k)
        H, W = _arr[k].shape
        c0, c1 = max(0, int(math.floor((W_ - WSF_X0) / px))), min(W, int(math.ceil((E_ - WSF_X0) / px)))
        r0, r1 = max(0, int(math.floor((WSF_Y0 - N_) / px))), min(H, int(math.ceil((WSF_Y0 - S_) / px)))
        return c0, c1, r0, r1, px

    async def wsf_window(k, box, stride=1):
        """The level-k pixels under the box, every `stride`-th in each axis:
        (int8 (h, w), lon of the columns, lat of the rows) or None. zarr's
        sync read runs in a thread (it owns an event loop of its own)."""
        c0, c1, r0, r1, px = _window_ix(k, box)
        if c1 <= c0 or r1 <= r0:
            return None
        key = (k, stride, r0, r1, c0, c1)
        a = _win.get(key)
        if a is None:
            loop = asyncio.get_running_loop()
            async with _sem:
                a = await loop.run_in_executor(None, lambda: np.asarray(_arr[k][r0:r1:stride, c0:c1:stride]))
            _win[key] = a
            if len(_win) > 64:
                _win.pop(next(iter(_win)))
        lon = WSF_X0 + (c0 + stride * np.arange(a.shape[1]) + 0.5) * px
        lat = WSF_Y0 - (r0 + stride * np.arange(a.shape[0]) + 0.5) * px
        return a, lon, lat

    # above wsf_tile_png: marimo drops a cell's _private helper that is only
    # referenced by a function defined before it (NameError under marimo run)
    def _wsf_png(got, k, n, y, lon0, lon1):
        T = RASTER_TILE
        arr, lon, lat = got
        ys = np.pi * (1 - 2 * (y + (np.arange(T) + 0.5) / T) / n)
        lat_c = np.degrees(np.arctan(np.sinh(ys)))
        lon_c = lon0 + (np.arange(T) + 0.5) * (lon1 - lon0) / T
        px = _px(k)
        ci = np.floor((lon_c - (lon[0] - px / 2)) / px).astype(np.int64)
        ri = np.floor(((lat[0] + px / 2) - lat_c) / px).astype(np.int64)
        okc, okr = (ci >= 0) & (ci < arr.shape[1]), (ri >= 0) & (ri < arr.shape[0])
        pxv = arr[np.clip(ri, 0, arr.shape[0] - 1)[:, None], np.clip(ci, 0, arr.shape[1] - 1)[None, :]]
        pxv = np.where(okr[:, None] & okc[None, :], pxv, 0)
        rgba = _cmap[pxv.astype(np.uint8)]
        if not rgba[..., 3].any():
            return None
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    async def wsf_tile_png(z, x, y):
        """RGBA PNG bytes for Web Mercator tile (z, x, y) of the pyramid (the
        earliest built-up date under each pixel, the year's color), or None
        where nothing is built. The level is the one whose pixel is nearest the
        tile's own (in metres at the tile's latitude)."""
        key = (z, x, y)
        if key in _png_cache:
            return _png_cache[key]
        T = RASTER_TILE
        n = 2 ** z
        lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
        lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        if lat1 < -60.01 or lat0 > 78.01:
            _png_cache[key] = None
            return None
        m_tile = 2 * math.pi * 6378137.0 / (n * T) * math.cos(math.radians((lat0 + lat1) / 2))
        k = max(0, min(WSF_LEVELS - 1, int(round(math.log2(max(m_tile, 10.0) / 10.0)))))
        got = await wsf_window(k, (lon0, lat0, lon1, lat1))
        if got is None:
            _png_cache[key] = None
            return None
        _png_cache[key] = await cpu(_wsf_png, got, k, n, y, lon0, lon1)
        if len(_png_cache) > 4000:
            _png_cache.pop(next(iter(_png_cache)))
        return _png_cache[key]

    _CNT = ", ".join(f"sum(CASE WHEN idx = {k} THEN 1 ELSE 0 END) AS c{k:02d}" for k in range(1, WSF_NIDX + 1))

    async def wsf_fold(box, res):
        """Per res cell over the box: the number of sampled pixels (`npx`),
        how many are built-up (`built`) and one count per date index (`c01`..
        `c20`). Every cell with a sample is present, built or not, so the grid
        is whole. The window's shares are derived from the counts in the frame,
        so a window change is a frame, never a refold. (table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        c0, c1, r0, r1, _ = _window_ix(0, box)
        if c1 <= c0 or r1 <= r0:
            return None, "WSF: nothing under the view (off the record, 60 S to 78 N)"
        full = (c1 - c0) * (r1 - r0)
        stride = max(1, int(math.ceil(math.sqrt(full / WSF_MAX_PX))))
        got = await wsf_window(0, box, stride)
        if got is None:
            return None, "WSF: nothing under the view"
        arr, lon, lat = got
        tr = time.time()
        h, w = arr.shape

        def _run():
            name = f"wsf_{next(_seq)}"
            LON, LAT = np.meshgrid(lon, lat)
            ctx.from_dataset(
                name,
                xr.Dataset(
                    {"idx": (("y", "x"), arr.astype(np.int16)), "lat": (("y", "x"), LAT), "lon": (("y", "x"), LON)},
                    coords={"y": np.arange(h), "x": np.arange(w)},
                ),
                chunks={"y": 512},
            )
            try:
                return ctx.sql(f"""
                    SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell,
                           count(*) AS npx,
                           sum(CASE WHEN idx > 0 THEN 1 ELSE 0 END) AS built,
                           {_CNT}
                    FROM {name}
                    WHERE lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                    GROUP BY cell
                """).to_arrow_table()
            finally:
                ctx.deregister_table(name)

        out = await cpu(_run)
        return out, (
            f"WSF {w:,}x{h:,} samples (stride {stride}, {10 * stride} m) read {tr - t0:.1f} s · fold {out.num_rows:,} {time.time() - tr:.1f} s"
        )

    return WSF_DATE, idx_h1, idx_h2, wsf_fold, wsf_tile_png, wsf_window


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

    async def aef_fold(box, res, year):
        """Mean AlphaEarth vector per res cell over the box for one year.
        Returns (arrow table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        if res >= MOSAIC_MIN_RES:
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
        li = AEF_LEVEL_FOR_RES[res]
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
def _(
    BLD_MAX,
    CACHE_DIR,
    HOME,
    OV_PINNED,
    OV_PORTOLAN,
    OV_STAC_ROOT,
    OV_TILES,
    duckdb,
    json,
    np,
    os,
    pa,
    pq,
    urllib,
):
    # ---- Overture: the release, the file index, the footprints, the place ------
    # Everything Overture in one cell, all of it against Overture's own bucket.
    # The release is resolved once at import; the file index per (theme, type)
    # is 512 small STAC item reads the first time, then a parquet under tmp;
    # a query names the files whose bbox meets the box and lets DuckDB prune
    # the row groups inside them by the bbox column (1 to 3 s warm for a city
    # block; the cold footer read that took six minutes over the whole
    # partition never happens).
    import threading as _th
    from concurrent.futures import ThreadPoolExecutor as _Pool
    import pyarrow.compute as _pc
    import pyarrow.fs as _pafs
    import shapely
    from rasterio import features as _rf
    from rasterio.transform import from_origin as _from_origin

    def _get_json(url, timeout=60):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)

    def _resolve_release():
        """(release tag, how): the override, else the pin if Overture's STAC
        still lists it, else the newest release listed; the pin alone when
        the catalog does not answer."""
        env = os.environ.get("OVERTURE_RELEASE")
        if env:
            return env, "OVERTURE_RELEASE override"
        try:
            root = _get_json(OV_STAC_ROOT)
            rels = sorted({l["href"].rstrip("/").split("/")[-2] for l in root["links"] if l.get("rel") == "child"})
        except Exception as e:
            return OV_PINNED, f"pinned (Overture's STAC did not answer: {type(e).__name__})"
        if OV_PINNED in rels:
            return OV_PINNED, "pinned"
        if not rels:
            return OV_PINNED, "pinned (Overture's STAC lists no release)"
        return rels[-1], f"newest; the pinned {OV_PINNED} is gone from Overture's STAC"

    OV_RELEASE, OV_RELEASE_HOW = _resolve_release()
    OV_BLD_PM = f"{OV_TILES}/{OV_RELEASE}/buildings.pmtiles"
    OV_DIV_PM = f"{OV_TILES}/{OV_RELEASE}/divisions.pmtiles"
    _STAC_BASE = OV_STAC_ROOT.rsplit("/", 1)[0]

    def _s3(href):
        """An item's asset href as DuckDB reads it (s3://bucket/key)."""
        if href.startswith("s3://"):
            return href
        u = urllib.parse.urlparse(href)
        bucket = u.netloc.split(".s3.")[0] if ".s3." in u.netloc else u.netloc.split(".")[0]
        return f"s3://{bucket}{u.path}"

    _IX, _ix_lock = {}, _th.Lock()

    def _index(theme, typ):
        """The release's files for one type: dict of numpy arrays (xmin, ymin,
        xmax, ymax, path). Built from the STAC items once, kept as parquet."""
        key = (theme, typ)
        with _ix_lock:
            if key in _IX:
                return _IX[key]
            path = os.path.join(CACHE_DIR, f"overture_{OV_RELEASE}_{theme}_{typ}.parquet")
            if os.path.exists(path):
                t = pq.read_table(path)
            else:
                coll = None
                for base in (f"{_STAC_BASE}/{OV_RELEASE}/{theme}/{typ}/collection.json",
                             f"{OV_PORTOLAN}/{theme}/{typ}/collection.json"):
                    try:
                        coll = (_get_json(base), base)
                        break
                    except Exception:
                        continue
                if coll is None:
                    raise RuntimeError(f"no STAC collection for {theme}/{typ} in release {OV_RELEASE}")
                c, base = coll
                items = [urllib.parse.urljoin(base, l["href"]) for l in c["links"] if l.get("rel") == "item"]

                def one(u):
                    it = _get_json(u)
                    a = it["assets"].get("aws") or it["assets"].get("data") or next(iter(it["assets"].values()))
                    return (it["id"], *[float(v) for v in it["bbox"][:4]], _s3(a["href"]))

                with _Pool(16) as ex:
                    rows = [r for r in ex.map(one, items) if f"/release/{OV_RELEASE}/" in r[5]]
                if not rows:
                    raise RuntimeError(f"the STAC items for {theme}/{typ} do not point at release {OV_RELEASE}")
                t = pa.table({"id": [r[0] for r in rows], "xmin": [r[1] for r in rows], "ymin": [r[2] for r in rows],
                              "xmax": [r[3] for r in rows], "ymax": [r[4] for r in rows], "path": [r[5] for r in rows]})
                os.makedirs(CACHE_DIR, exist_ok=True)
                pq.write_table(t, path)
            _IX[key] = {k: t[k].to_numpy(zero_copy_only=False) for k in t.column_names}
            return _IX[key]

    def _files(theme, typ, box):
        ix = _index(theme, typ)
        W_, S_, E_, N_ = box
        hit = (ix["xmax"] >= W_) & (ix["xmin"] <= E_) & (ix["ymax"] >= S_) & (ix["ymin"] <= N_)
        return [str(p) for p in ix["path"][hit]]

    # one DuckDB connection for Overture, a cursor per call, the object cache
    # on so a file's footer is read once
    _dv = {"con": None, "err": None}
    _lock = _th.Lock()

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
                    c.execute("SET GLOBAL s3_region='us-west-2'; SET GLOBAL enable_object_cache=true")
                    # the stalls S3_OPTS cuts short, on DuckDB's own client: a
                    # socket timeout (no bytes for 6 s), not a cap on a long read,
                    # and a flat retry (the default backoff of 4 waits 25.6 s by
                    # the fifth try: a 2-row-group, 7 MB read took 90 s)
                    c.execute("SET GLOBAL http_timeout=6; SET GLOBAL http_retries=5; SET GLOBAL http_retry_backoff=1.5; SET GLOBAL http_retry_wait_ms=200")
                    _dv["con"] = c
                except Exception as e:
                    _dv["err"] = e
            return _dv["con"]

    def _cur():
        c = _connect()
        if c is None:
            raise _dv["err"]
        return c.cursor()

    def _lst(files):
        return "[" + ", ".join(f"'{f}'" for f in files) + "]"

    # The footprints are read with pyarrow, not DuckDB (2026-09-24): DuckDB's
    # HTTP client stalled at random on this bucket (a 7 MB, two-row-group read
    # took 3 s one run and 67 to 105 s the next, whatever the query, and its
    # http_timeout did not cut the stall short). Here each request is cut at
    # 6 s and retried; the footer's bbox statistics pick the row groups, and
    # each one is read in its own thread with its column ranges pre-buffered
    # (0.9 to 3.1 s for the same boxes, twelve reads, no stall).
    _fs = _pafs.S3FileSystem(anonymous=True, region="us-west-2", request_timeout=6, connect_timeout=3,
                             retry_strategy=_pafs.AwsStandardS3RetryStrategy(max_attempts=6))
    _rg_boxes = {}  # file -> [(row group, xmin, ymin, xmax, ymax)], from its footer, once
    _io = _Pool(16, thread_name_prefix="overture")
    _BCOLS = ["id", "sources", "subtype", "class", "height", "num_floors", "geometry", "bbox"]

    def _row_groups(path):
        if path not in _rg_boxes:
            md = pq.ParquetFile(path.removeprefix("s3://"), filesystem=_fs).metadata
            names = [md.schema.column(j).path for j in range(md.num_columns)]
            J = {k: names.index(f"bbox.{k}") for k in ("xmin", "ymin", "xmax", "ymax")}
            _rg_boxes[path] = [
                (i, *(getattr(md.row_group(i).column(J[k]).statistics, "min" if k in ("xmin", "ymin") else "max")
                      for k in ("xmin", "ymin", "xmax", "ymax")))
                for i in range(md.num_row_groups)
            ]
        return _rg_boxes[path]

    def _read_rg(path, i):
        pf = pq.ParquetFile(path.removeprefix("s3://"), filesystem=_fs, pre_buffer=True)
        return pf.read_row_group(i, columns=_BCOLS)

    def bld_rows(box):
        """The building footprints whose bbox meets the box: arrow table
        (id, dataset, updated, subtype, class, height, num_floors, wkb,
        xmin, ymin, xmax, ymax), at most BLD_MAX + 1 rows (the caller reads
        the overflow as "zoom in"). Empty when no file covers the box."""
        W_, S_, E_, N_ = box
        files = _files("buildings", "building", box)
        jobs = [
            (p, i) for p, rgs in zip(files, _io.map(_row_groups, files))
            for (i, x0, y0, x1, y1) in rgs
            if x0 <= E_ and x1 >= W_ and y0 <= N_ and y1 >= S_
        ]
        if not jobs:
            return pa.table({"id": pa.array([], pa.string())})
        t = pa.concat_tables(list(_io.map(lambda j: _read_rg(*j), jobs)))
        bb = t["bbox"]
        f = _pc.struct_field
        t = t.filter(_pc.and_(
            _pc.and_(_pc.less_equal(f(bb, "xmin"), E_), _pc.greater_equal(f(bb, "xmax"), W_)),
            _pc.and_(_pc.less_equal(f(bb, "ymin"), N_), _pc.greater_equal(f(bb, "ymax"), S_)),
        )).slice(0, BLD_MAX + 1)
        src0 = _pc.list_element(t["sources"], 0)
        bb = t["bbox"]
        return pa.table({
            "id": t["id"], "dataset": f(src0, "dataset"), "updated": f(src0, "update_time"),
            "subtype": t["subtype"], "class": t["class"], "height": t["height"], "num_floors": t["num_floors"],
            "wkb": t["geometry"],
            "xmin": f(bb, "xmin"), "ymin": f(bb, "ymin"), "xmax": f(bb, "xmax"), "ymax": f(bb, "ymax"),
        })

    def bld_geoarrow(rows):
        """The footprints as Arrow IPC stream bytes: one record batch, one
        column `geometry`, native GeoArrow (geoarrow.polygon, or
        geoarrow.multipolygon when any footprint is multi-part), interleaved
        float64 xy. Row i is footprint i (the order of `rows`, which `bcolors`
        follows). Anything not polygonal becomes an empty polygon so the rows
        stay aligned."""
        if rows.num_rows == 0:
            return b""
        geoms = shapely.from_wkb(rows["wkb"].to_numpy(zero_copy_only=False))
        tid = shapely.get_type_id(geoms)
        bad = (tid != 3) & (tid != 6)
        if bad.any():
            geoms = geoms.copy()
            geoms[bad] = shapely.Polygon()
        gtype, coords, offs = shapely.to_ragged_array(geoms, include_z=False)
        xy_t = pa.list_(pa.field("xy", pa.float64(), nullable=False), 2)
        arr = pa.FixedSizeListArray.from_arrays(pa.array(coords.ravel()), type=xy_t)
        for name, o in zip(("vertices", "rings", "polygons"), offs):
            arr = pa.ListArray.from_arrays(pa.array(o, pa.int32()), arr, type=pa.list_(pa.field(name, arr.type, nullable=False)))
        ext = "geoarrow.multipolygon" if gtype == shapely.GeometryType.MULTIPOLYGON else "geoarrow.polygon"
        field = pa.field("geometry", arr.type, metadata={"ARROW:extension:name": ext, "ARROW:extension:metadata": '{"crs":"OGC:CRS84"}'})
        tbl = pa.Table.from_arrays([arr], schema=pa.schema([field]))
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, tbl.schema) as w:
            w.write_batch(tbl.combine_chunks().to_batches()[0])
        return sink.getvalue().to_pybytes()

    def bld_cover(rows, shape, lon0, lat0, px):
        """int32 (h, w) on the WSF window's grid (lon0, lat0 its north-west
        corner, px its pixel): 0 under no footprint, else 1 + the row index
        of the footprint on the pixel (the last drawn wins where two overlap).
        all_touched: a footprint smaller than a pixel still claims the pixel
        it sits in, so the gaps are pixels no footprint reaches at all."""
        if rows.num_rows == 0:
            return np.zeros(shape, np.int32)
        geoms = shapely.from_wkb(rows["wkb"].to_numpy(zero_copy_only=False))
        return _rf.rasterize(
            zip(geoms, range(1, len(geoms) + 1)),
            out_shape=shape, transform=_from_origin(lon0, lat0, px, px),
            fill=0, dtype="int32", all_touched=True,
        )

    _ORDER = {"locality": 0, "localadmin": 1, "county": 2, "region": 3, "country": 4}

    def division_at(lon, lat):
        """The divisions holding the point, smallest first: a list of
        {subtype, name, name_en, local_type, population}, locality up to
        country. The files are the ones whose bbox holds the point; inside
        them DuckDB prunes by the bbox column and runs ST_Contains on the
        rest. Raises on a failed read so the caller can say so."""
        pt = (float(lon), float(lat), float(lon), float(lat))
        files = _files("divisions", "division_area", pt)
        if not files:
            return []
        cur = _cur()
        rows = cur.execute(
            "SELECT subtype, names.primary, names.common['en'], country, division_id "
            f"FROM read_parquet({_lst(files)}, hive_partitioning=0) "
            "WHERE bbox.xmin <= $x AND bbox.xmax >= $x AND bbox.ymin <= $y AND bbox.ymax >= $y "
            "AND class = 'land' AND ST_Contains(geometry, ST_Point($x, $y))",
            {"x": float(lon), "y": float(lat)},
        ).fetchall()
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
                dfiles = _files("divisions", "division", pt)
                extra = {i: (lt, pop) for i, lt, pop in cur.execute(
                    "SELECT id, local_type['en'], population "
                    f"FROM read_parquet({_lst(dfiles)}, hive_partitioning=0) "
                    "WHERE country = $country AND list_contains($ids, id)",
                    {"country": country, "ids": ids},
                ).fetchall()} if dfiles else {}
            except Exception:
                extra = {}
            for d in out:
                d["local_type"], d["population"] = extra.get(d["id"], (None, None))
        return out

    # the indexes and the footers, read now rather than on the first view,
    # off the main thread
    def _warm():
        try:
            _index("buildings", "building")
            division_at(HOME["longitude"], HOME["latitude"])
        except Exception:
            pass

    _th.Thread(target=_warm, daemon=True).start()
    return OV_BLD_PM, OV_DIV_PM, OV_RELEASE, OV_RELEASE_HOW, bld_cover, bld_geoarrow, bld_rows, division_at


@app.cell
def _(
    ALPHA_FILL,
    ALPHA_QUIET,
    FA,
    GREW_RAMP,
    MIN_STABLE_CELLS,
    NEW_MIN,
    VIRIDIS,
    WSF_DATE,
    WSF_NIDX,
    YEAR_RGB,
    con,
    idx_h1,
    idx_h2,
    np,
    pa,
):
    # ---- a FRAME: the join, the window's shares, the displacement, the quiet level, five fills --
    _stops = np.array([[int(VIRIDIS[i + j:i + j + 2], 16) for j in (0, 2, 4)] for i in range(0, len(VIRIDIS), 6)], np.float64)
    RAMP = np.stack(
        [np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_stops)), _stops[:, k]) for k in range(3)], 1
    ).round().astype(np.uint8)
    RAMP_HEX = ["#%02x%02x%02x" % tuple(int(v) for v in RAMP[i]) for i in range(0, 256, 17)]
    _bstops = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in GREW_RAMP], np.float64)
    BRAMP = np.stack(
        [np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_bstops)), _bstops[:, k]) for k in range(3)], 1
    ).round().astype(np.uint8)
    BRAMP_HEX = ["#%02x%02x%02x" % tuple(int(v) for v in BRAMP[i]) for i in range(0, 256, 17)]
    _E = [f"e{i:02d}" for i in range(64)]
    _GREY = np.array([128, 128, 128], np.uint8)

    def build_frame(wsf_cells, aef_by_year, y0, y1):
        """Join the WSF fold with each AlphaEarth year in the window y0..y1
        (LEFT: a cell keeps its WSF counts with or without an embedding).
        From the 20 date counts: `p_built` (built-up
        by the end of y1), `p_new` (became built-up in y0+1..y1, the years the
        y0 and y1 composites straddle), `byear` (the year with most of the new
        pixels). `disp` is the displacement between the two ENDS (1 - cos of
        the y0 and y1 vectors): the "AEF changed" fill. The consecutive steps
        inside the window give D0 (the quantile of the largest step among the
        cells WSF says did not grow) and `when`, the year of the first step
        above D0: the "AEF change year" fill."""
        con.register("wsf_cells", wsf_cells)
        years = [y for y in range(y0, y1 + 1) if aef_by_year.get(y) is not None]
        sel = ["w.*"]
        joins = []
        for y in years:
            con.register(f"aef_{y}", aef_by_year[y])
            sel += [f"a{y}.{e} AS {e}_{y}" for e in _E]
            joins.append(f"LEFT JOIN aef_{y} a{y} USING (cell)")
        j = con.execute(f"SELECT {', '.join(sel)} FROM wsf_cells w {' '.join(joins)} ORDER BY cell").arrow().read_all()
        n = j.num_rows
        npx = j["npx"].to_numpy().astype(np.float64)
        C = np.stack([j[f"c{k:02d}"].to_numpy().astype(np.float64) for k in range(1, WSF_NIDX + 1)], 0) if n else np.zeros((WSF_NIDX, 0))
        k_end = min(WSF_NIDX, idx_h2(y1))
        k_lo = idx_h1(y0 + 1)
        p_built = (C[:k_end].sum(0) / np.maximum(npx, 1)).astype(np.float32)
        p_new = (C[k_lo - 1:k_end].sum(0) / np.maximum(npx, 1)).astype(np.float32)
        new_years = list(range(y0 + 1, y1 + 1))
        per_year = np.stack([C[idx_h1(y) - 1] + C[idx_h2(y) - 1] for y in new_years], 0) if n and new_years else np.zeros((max(1, len(new_years)), n))
        grew = p_new >= NEW_MIN
        top = per_year.argmax(0) if n else np.zeros(0, np.int64)
        byear = np.where(grew, np.array(new_years or [-1], np.int64)[top], np.where(p_built > 0, -1, -3)).astype(np.int64)
        # the first half-year the cell had any built-up pixel: the record's date
        first_idx = np.where(C.any(0), (C > 0).argmax(0) + 1, 0).astype(np.int64) if n else np.zeros(0, np.int64)

        def _V(y):
            V = np.stack([j[f"{e}_{y}"].to_numpy(zero_copy_only=False) for e in _E], axis=1).astype(np.float32)
            nrm = np.linalg.norm(V, axis=1)
            V = V / np.maximum(nrm, 1e-9)[:, None]
            V[~np.isfinite(nrm) | (nrm == 0)] = np.nan
            return V

        Vs = {y: _V(y) for y in years} if n else {}
        step_years = [(years[k], years[k + 1]) for k in range(len(years) - 1)]
        steps = np.full((max(1, len(step_years)), n), np.nan, np.float32)
        for k, (ya, yb) in enumerate(step_years):
            steps[k] = (1.0 - np.einsum("ij,ij->i", Vs[ya], Vs[yb])).astype(np.float32)
        if y0 in Vs and y1 in Vs and y0 != y1:
            disp = (1.0 - np.einsum("ij,ij->i", Vs[y0], Vs[y1])).astype(np.float32)
        else:
            disp = np.full(n, np.nan, np.float32)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            disp_max = np.nanmax(steps, axis=0) if n else np.zeros(0, np.float32)
        scored = ~np.isnan(disp)
        stepped = ~np.isnan(disp_max)
        stable_ok = stepped & ~grew
        when = np.full(n, -2, np.int64)
        if stable_ok.sum() >= MIN_STABLE_CELLS:
            ds = disp_max[stable_ok].astype(np.float64)
            D0 = float(np.quantile(ds, 1 - FA))
            above = steps > D0
            first = above.argmax(axis=0)
            yrs = np.array([yb for _, yb in step_years] or [-1], np.int64)
            when = np.where(above.any(axis=0), yrs[first], np.where(stepped, -1, -2)).astype(np.int64)
        else:
            D0 = float("nan")
        moved = when >= 0
        when_name = {
            -1: f"no single year stood out {y0} to {y1} (every step under the quiet level)",
            -2: "no AlphaEarth embedding here",
        }
        for ya, yb in step_years:
            when_name[yb] = f"AlphaEarth changed in {yb} (its {ya} vs {yb} fingerprints)"
        byear_name = {-1: f"built-up before {y0 + 1}, nothing new to {y1}", -3: "nothing built-up here"}
        for y in new_years:
            byear_name[y] = f"most new built-up ground arrived in {y}"
        cells = pa.table({
            "cell": j["cell"],
            "npx": j["npx"],
            "p_built": pa.array(p_built),
            "p_new": pa.array(p_new),
            "grew": pa.array(grew),
            "byear": pa.array(byear.astype(np.int16)),
            "byear_name": pa.array([byear_name[int(b)] for b in byear]),
            "first_date": pa.array([WSF_DATE[int(k)] for k in first_idx]),
            "disp": pa.array(disp.astype(np.float32)),
            "disp_max": pa.array(disp_max.astype(np.float32)),
            **{f"step_{yb}": pa.array(steps[k]) for k, (_, yb) in enumerate(step_years)},
            "moved": pa.array(moved),
            "when": pa.array(when.astype(np.int16)),
            "when_name": pa.array([when_name[int(w)] for w in when]),
        })
        cellid = cells["cell"].to_numpy().astype(np.uint64)
        ok = disp[scored]
        if len(ok) >= 2:
            lo, hi = (float(q) for q in np.percentile(ok, [2, 98]))
            if hi <= lo:
                hi = lo + 1e-6
        else:
            lo, hi = 0.0, 1.0

        def _stretch(v, mask):
            vv = v[mask]
            if len(vv) >= 2:
                a, b = float(np.percentile(vv, 2)), float(np.percentile(vv, 98))
                if b <= a:
                    b = a + 1e-6
            else:
                a, b = 0.0, 1.0
            t = np.clip((np.where(mask, v, a) - a) / (b - a), 0, 1)
            return BRAMP[(t * 255).round().astype(np.int64)], a, b

        has_new = p_new > 0
        rgb_new, n_lo, n_hi = _stretch(p_new, has_new)
        _t = np.clip((np.where(scored, disp, lo) - lo) / (hi - lo), 0, 1)
        rgb_shift = np.where(scored[:, None], RAMP[(_t * 255).round().astype(np.int64)], _GREY).astype(np.uint8)
        rgb_when = np.array([YEAR_RGB.get(int(w), (45, 45, 45)) for w in when], np.uint8) if n else np.zeros((0, 3), np.uint8)
        rgb_byear = np.array([YEAR_RGB.get(int(b), (45, 45, 45)) for b in byear], np.uint8) if n else np.zeros((0, 3), np.uint8)

        def fill(kind, hit=None):
            """(N, 4) uint8 rgba: the right pane's getFillColor. The picked cell
            keeps its color (the stroke is the widget's, gold, on both panes)."""
            if kind == "built":
                # the raster carries this one; the hexagons are not drawn
                c, a = rgb_new, np.zeros(n, np.int64)
            elif kind == "grew":
                c, a = rgb_new, np.where(has_new, ALPHA_FILL, ALPHA_QUIET)
            elif kind == "byear":
                c, a = rgb_byear, np.where(grew, ALPHA_FILL, ALPHA_QUIET)
            elif kind == "shift":
                c, a = rgb_shift, np.where(scored, ALPHA_FILL, ALPHA_QUIET)
            else:
                c, a = rgb_when, np.where(when >= 0, ALPHA_FILL, ALPHA_QUIET)
            return np.ascontiguousarray(np.concatenate([c, a[:, None].astype(np.uint8)], axis=1)).astype(np.uint8)

        def legend(kind):
            tot = max(1, n)
            if kind == "built":
                # the raster's palette: the year each pixel first read as built-up
                items = [{"name": "already built-up in 2016", "hex": "#%02x%02x%02x" % YEAR_RGB[2016]}]
                items += [{"name": str(y), "hex": "#%02x%02x%02x" % YEAR_RGB[y]} for y in range(2017, 2026)]
                return items
            if kind == "grew":
                return [{"ramp": BRAMP_HEX, "lo": f"new {y0 + 1} to {y1} {100 * n_lo:.1f}%", "hi": f"{100 * n_hi:.1f}%",
                         "title": f"share of the hexagon's sampled WSF pixels that first read built-up in {y0 + 1}..{y1}, stretched to this view's p2-p98 of the cells that grew; none drawn faint"}]
            if kind == "shift":
                return [{"ramp": RAMP_HEX, "lo": f"{y0} to {y1} shift {lo:.3f}", "hi": f"{hi:.3f}",
                         "title": f"1 - cos between the cell's AlphaEarth vectors in {y0} and {y1} (the two ends of the window, whatever happened between), stretched to this view's p2-p98"}]
            if kind == "when":
                items = []
                when_short = {-1: "no single year", -2: "no embedding"}
                for w in [yb for _, yb in step_years] + [-1, -2]:
                    m = when == w
                    if m.any():
                        items.append({"name": when_name[w], "short": when_short.get(w, str(w)),
                                      "hex": "#%02x%02x%02x" % YEAR_RGB.get(w, (45, 45, 45)), "pct": round(100 * int(m.sum()) / tot, 1)})
                return items
            items = []
            byear_short = {-1: f"before {y0 + 1}", -3: "nothing built"}
            for b in new_years + [-1, -3]:
                m = byear == b
                if m.any():
                    items.append({"name": byear_name[b], "short": byear_short.get(b, str(b)),
                                  "hex": "#%02x%02x%02x" % YEAR_RGB.get(b, (45, 45, 45)), "pct": round(100 * int(m.sum()) / tot, 1)})
            return items

        n_moved = int(moved.sum())
        n_grew = int(grew.sum())
        both = int((grew & moved).sum())
        score = (
            f"WSF: {n_grew:,} of {n:,} cells grew {y0 + 1}..{y1} · AEF D0 {D0:.3f}, {n_moved:,} of {int(stepped.sum()):,} scored cells moved · both {both:,}"
            if not np.isnan(D0)
            else f"WSF: {n_grew:,} of {n:,} cells grew {y0 + 1}..{y1} · AEF unscored (no embedding, one year only, or too few quiet cells)"
        )
        return {"cells": cells, "cellid": cellid, "p_built": p_built, "p_new": p_new, "byear": byear, "disp": disp, "when": when,
                "years": years, "new_years": new_years, "y0": y0, "y1": y1, "steps": steps, "step_years": step_years,
                "D0": D0, "shift_lo": lo, "shift_hi": hi, "fill": fill, "legend": legend, "score": score, "n_grew": n_grew}

    return (build_frame,)


@app.cell
def _(anywidget, asyncio, traitlets):
    class OneMap(anywidget.AnyWidget):
        """ONE maplibre map, one deck overlay, the datasets as layers switched on
        and off (Stephen, 2026-09-24: "We should just have one map and look at
        the different data sets"). Always: the Overture footprints past
        BLD_ZOOM, one fill at a time. Switchable on top or beneath: the WSF
        raster (the pyramid, one colour per year of first detection), the H3
        hexagons (WSF / AlphaEarth fills, the year window), the S2 mosaic.

        Kernel -> browser: `polys` (the footprints as an Arrow IPC stream,
        one batch, a native GeoArrow polygon column, row = footprint) with `bcolors`
        (rgba u8 per footprint, swapped per fill without resending the
        polygons); `cells` (uint64 LE) with `colors` (rgba u8) for the
        hexagons; `config` (JSON); `status` / `panel` / `legend` (strings for
        the strip). Browser -> kernel: `view` (JSON lon/lat/zoom + w/h on
        every moveend), `pick` (JSON: lon/lat, the cell at the frame's res or
        null, the region and county from the division tiles), `ctl` (JSON:
        the layer switches, fills, S2 year and scale, the window, labels).
        Tiles: custom messages, PNG bytes back, for `s2` and `wsf`."""

        cells = traitlets.Bytes(b"").tag(sync=True)
        colors = traitlets.Bytes(b"").tag(sync=True)
        polys = traitlets.Bytes(b"").tag(sync=True)
        bcolors = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("[]").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)
        ctl = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None  # async (src, z, x, y, year) -> PNG bytes or None; src "s2" | "wsf"
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict) or content.get("kind") != "tile":
                return
            try:
                asyncio.get_running_loop().create_task(self._tile(content))
            except RuntimeError as e:
                self.send({"kind": "tile", "id": content.get("id"), "err": f"no loop: {e}"})

        async def _tile(self, c):
            """A FAILURE IS AN ERROR, never an empty tile (deck caches an empty
            tile as loaded and the area stays blank for good)."""
            if self.tile_fn is None:
                self.send({"kind": "tile", "id": c["id"], "err": "no tile_fn (re-run the wiring cell)"})
                return
            try:
                png = await self.tile_fn(c.get("src", "s2"), int(c["z"]), int(c["x"]), int(c["y"]), int(c["year"]))
            except Exception as e:
                self.send({"kind": "tile", "id": c["id"], "err": f"{type(e).__name__}: {e}"})
                return
            if png is None:
                self.send({"kind": "tile", "id": c["id"], "empty": True})
            else:
                self.send({"kind": "tile", "id": c["id"]}, buffers=[png])

        _esm = r"""
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {TileLayer, H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {ClipExtension} from "https://esm.sh/@deck.gl/extensions@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import * as arrow from "https://esm.sh/apache-arrow@18.1.0";
        import {GeoArrowPolygonLayer} from "https://esm.sh/@geoarrow/deck.gl-layers@0.3.2?deps=@deck.gl/core@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/geo-layers@9.3.10,@deck.gl/aggregation-layers@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {latLngToCell, getResolution, cellToBoundary} from "https://esm.sh/h3-js@4.5.0";
        import {Protocol as PMProtocol} from "https://esm.sh/pmtiles@4.5.0";
        maplibregl.addProtocol("pmtiles", new PMProtocol().tile);

        const STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";  // Positron, as the settlement pair (Stephen, 2026-09-24)

        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength);
          return null;
        }
        function copyOf(u8) { return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength); }

        function render({model, el}) {
          let cfg = {};
          try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
          const css = document.createElement("link");
          css.rel = "stylesheet"; css.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css";
          const font = "font:11px ui-sans-serif,system-ui,sans-serif";
          const mono = "font:11px ui-monospace,Menlo,monospace";
          const root = document.createElement("div");
          root.className = "sp-root";
          root.style.cssText = "width:100%;background:#fff;color:#222;" + font;
          const pane = document.createElement("div");
          pane.className = "sp-pane sp-right";
          pane.style.cssText = "position:relative;width:100%;height:" + (cfg.height || 720) + "px;background:#f4f2ee";
          const mapEl = document.createElement("div");
          mapEl.className = "sp-map";
          mapEl.style.cssText = "position:absolute;inset:0";
          // PAIR (Stephen, 2026-09-24: "next to the s2 info ... a toggle
          // between pair and slider"): a second map on the left half draws the
          // S2 mosaic alone, the data map takes the right half, one camera.
          // Made the first time PAIR is picked
          const mapElS = document.createElement("div");
          mapElS.className = "sp-map sp-map-s2";
          mapElS.style.cssText = "position:absolute;top:0;bottom:0;left:0;width:50%;display:none;border-right:2px solid #fff;box-sizing:border-box";
          const head = document.createElement("div");
          head.className = "sp-head";
          head.style.cssText = "position:absolute;left:8px;top:8px;z-index:5;display:flex;flex-direction:column;gap:.18rem;align-items:flex-start;" +
            "max-width:calc(100% - 72px);background:rgba(255,255,255,.82);color:#1d1d1b;padding:3px 7px;border-radius:6px;" +
            "box-shadow:0 1px 3px rgba(0,0,0,.18);white-space:nowrap;font-variant-numeric:tabular-nums";
          // the swipe (Stephen, 2026-09-24: "lets try a swipe"): S2 left of the
          // divider, the data right of it, one map. Shown while S2 is on.
          const swipe = document.createElement("div");
          swipe.className = "sp-swipe";
          swipe.style.cssText = "position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);z-index:4;cursor:col-resize;display:none;touch-action:none";
          const grip = document.createElement("div");
          grip.style.cssText = "position:absolute;top:50%;left:50%;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.4);display:flex;align-items:center;justify-content:center;font:12px ui-sans-serif,system-ui,sans-serif;color:#1d1d1b;user-select:none";
          grip.textContent = "⇔";
          swipe.appendChild(grip);
          let swipeF = 0.5;  // the divider, as a share of the pane width
          const placeSwipe = () => { swipe.style.left = (swipeF * 100) + "%"; };
          placeSwipe();
          swipe.addEventListener("pointerdown", (e) => { swipe.setPointerCapture(e.pointerId); e.preventDefault(); });
          swipe.addEventListener("pointermove", (e) => {
            if (!swipe.hasPointerCapture(e.pointerId)) return;
            const r = pane.getBoundingClientRect();
            swipeF = Math.max(0.03, Math.min(0.97, (e.clientX - r.left) / r.width));
            placeSwipe(); update();
          });
          swipe.addEventListener("pointerup", (e) => { try { swipe.releasePointerCapture(e.pointerId); } catch (err) {} });
          pane.append(mapElS, mapEl, swipe, head);
          const strip = document.createElement("div");
          const stripCss = "display:flex;flex-direction:column;gap:.25rem;padding:.35rem .4rem;background:#fff;color:#222";
          strip.style.cssText = stripCss;
          const status = document.createElement("div");
          status.className = "sp-status";
          status.style.cssText = "font:13px ui-sans-serif,system-ui,sans-serif;color:#666;white-space:pre-wrap;max-width:80ch";
          const legend = document.createElement("div");
          legend.className = "sp-legend";
          legend.style.cssText = "display:flex;flex-wrap:wrap;gap:.3rem .9rem;align-items:center;font-size:14px";
          const panel = document.createElement("div");
          panel.className = "sp-panel";
          panel.style.cssText = "font-size:14px;max-width:110ch;line-height:1.45";
          strip.append(legend, panel, status);
          status.hidden = !!cfg.minimal;
          root.append(pane, strip);
          el.append(css, root);

          // ---- the controls ---------------------------------------------------
          const ACCENT = "#2a5db0";
          const btnCss = font + ";padding:.1rem .45rem;border:0;background:transparent;color:#1d1d1b;cursor:pointer;line-height:1.4;font-variant-numeric:tabular-nums";
          const onCss = (b, on) => { b.style.background = on ? ACCENT : "transparent"; b.style.color = on ? "#fff" : "#1d1d1b"; };
          let s2y = cfg.s2_year, fill = cfg.fill || "grew", bfill = cfg.bfill || "wyear", labelsOn = cfg.labels !== false;
          let s2scale = Number(cfg.s2_scale) || 1;
          let y0 = cfg.aef_from, y1 = cfg.aef_to;
          let on = Object.assign({bld: true, wsf: false, hex: false, s2: false}, cfg.layers || {});
          let map = null, ov = null, hover = null;
          let mapS = null, ovS = null, syncing = false;  // PAIR's S2 map (below)
          const send = (act) => {
            model.set("ctl", JSON.stringify({act, on, s2y, s2scale, fill, bfill, y0, y1, labels: labelsOn, n: Date.now()}));
            model.save_changes();
          };
          const newRow = (h) => { const r = document.createElement("span"); r.style.cssText = "display:inline-flex;gap:.6rem;align-items:center"; h.appendChild(r); return r; };
          const rowOf = (h) => h.lastElementChild || newRow(h);
          const mkGroup = (h, title, values, get, set, act, cls, isOn) => {
            const wrap = document.createElement("span");
            wrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
            const lab = document.createElement("span"); lab.textContent = title;
            lab.style.cssText = "font-size:9.5px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#6b6b68;min-width:5.4em";
            const seg = document.createElement("span");
            seg.style.cssText = "display:inline-flex;border:1px solid rgba(29,29,27,.28);border-radius:5px;overflow:hidden";
            const btns = values.map((v, i) => {
              const b = document.createElement("button"); b.textContent = String(v.label != null ? v.label : v); b.style.cssText = btnCss;
              if (i) b.style.borderLeft = "1px solid rgba(29,29,27,.18)";
              b.className = cls; b.dataset.value = String(v.value != null ? v.value : v); if (v.title) b.title = v.title;
              b.onclick = () => { set(v.value != null ? v.value : v); style(); send(act); };
              seg.appendChild(b); return b;
            });
            wrap.append(lab, seg);
            rowOf(h).appendChild(wrap);
            const style = () => btns.forEach((b) => onCss(b, isOn ? isOn(b) : b.dataset.value === String(get())));
            style();
            return {style, wrap};
          };
          // row 1: LAYERS (each on or off) and FIND
          const layerDefs = (cfg.layer_defs || []).map((f) => ({value: f[0], label: f[1], title: f[2]}));
          // S2 is the swipe's left half, so it no longer has to exclude AEF
          const toggleOn = (v) => { on[v] = !on[v]; };
          const gLayers = mkGroup(head, "layers", layerDefs, () => null, (v) => { toggleOn(v); rows(); update(); }, "layers", "sp-layer", (b) => !!on[b.dataset.value]);
          // rows 2..: one per layer, shown while it is on
          newRow(head);
          const bfills = (cfg.bfills || []).map((f) => ({value: f[0], label: f[1], title: f[2]}));
          const gBFill = mkGroup(head, "buildings", bfills, () => bfill, (v) => { bfill = v; }, "bfill", "sp-bfill");
          const rowBld = head.lastElementChild;
          newRow(head);
          const fills = (cfg.fills || []).map((f) => ({value: f[0], label: f[1], title: f[2]}));
          const gFill = mkGroup(head, "AEF", fills, () => fill, (v) => { fill = v; }, "fill", "sp-fill");
          const rowHex = head.lastElementChild;
          const sty = document.createElement("style");
          sty.textContent = [
            ".sp-range{position:relative;width:200px;height:26px}",
            ".sp-range input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}",
            ".sp-range input:focus{outline:none}",
            ".sp-range input::-webkit-slider-runnable-track{background:none;height:22px}",
            ".sp-range input::-moz-range-track{background:none;height:22px}",
            ".sp-range input::-webkit-slider-thumb{pointer-events:auto;-webkit-appearance:none;appearance:none;width:12px;height:12px;margin-top:5px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-range input::-moz-range-thumb{pointer-events:auto;width:12px;height:12px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-range .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:rgba(29,29,27,.22);border-radius:2px}",
            ".sp-range .spn{position:absolute;top:9px;height:4px;background:#2a5db0;border-radius:2px}",
            ".sp-range .tks{position:absolute;left:8px;right:8px;top:17px;display:flex;justify-content:space-between;font-size:8px;color:#6b6b68;line-height:1}",
            ".sp-range .tks span{width:0;display:flex;justify-content:center}",
            ".sp-scale{position:relative;width:90px;height:22px}",
            ".sp-scale input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;-webkit-appearance:none;appearance:none}",
            ".sp-scale input:focus{outline:none}",
            ".sp-scale input::-webkit-slider-runnable-track{background:none;height:22px}",
            ".sp-scale input::-moz-range-track{background:none;height:22px}",
            ".sp-scale input::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:12px;height:12px;margin-top:5px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-scale input::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-scale .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:rgba(29,29,27,.22);border-radius:2px}",
            ".sp-scale .spn{position:absolute;left:8px;top:9px;height:4px;background:#2a5db0;border-radius:2px}",
            ".sp-year{height:30px}",
            ".sp-year .tks{position:absolute;left:8px;right:8px;top:19px;display:flex;justify-content:space-between;font-size:9px;color:#6b6b68;line-height:1}",
            ".sp-year .tks span{width:0;display:flex;justify-content:center}",
          ].join("\n");
          el.appendChild(sty);
          const eyebrow = (t) => { const s = document.createElement("span"); s.textContent = t; s.style.cssText = "font-size:9.5px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#6b6b68"; return s; };
          // the window (hexagons row): a stepped two-handle slider
          const aefYears = cfg.aef_years || [];
          const aefWrap = document.createElement("span");
          aefWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const rng = document.createElement("span"); rng.className = "sp-range";
          const trk = document.createElement("span"); trk.className = "trk";
          const spn = document.createElement("span"); spn.className = "spn";
          const tks = document.createElement("span"); tks.className = "tks";
          for (const y of aefYears) { const t = document.createElement("span"); const i = document.createElement("i"); i.style.fontStyle = "normal"; i.textContent = "\u2019" + String(y).slice(-2); t.appendChild(i); tks.appendChild(t); }
          const mkRange = () => { const r = document.createElement("input"); r.type = "range"; r.min = 0; r.max = Math.max(0, aefYears.length - 1); r.step = 1; r.title = "the window the hexagon fills read: drag either end (whole years); release to rebuild"; return r; };
          const rFrom = mkRange(), rTo = mkRange();
          const aefTxt = document.createElement("span"); aefTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:5.5em";
          rng.append(trk, spn, tks, rFrom, rTo);
          aefWrap.append(eyebrow("window"), rng, aefTxt);
          rowHex.appendChild(aefWrap);
          const styleAef = () => {
            const i0 = Math.max(0, aefYears.indexOf(y0)), i1 = Math.max(0, aefYears.indexOf(y1)), n = Math.max(1, aefYears.length - 1);
            rFrom.value = i0; rTo.value = i1;
            rFrom.style.zIndex = i0 === n ? 3 : 2; rTo.style.zIndex = i1 === 0 ? 3 : 2;
            const usable = rng.clientWidth - 16;
            spn.style.left = (8 + usable * i0 / n) + "px"; spn.style.width = (usable * (i1 - i0) / n) + "px";
            aefTxt.textContent = y0 + " to " + y1;
          };
          const onDrag = (which) => {
            let a = Number(rFrom.value), b = Number(rTo.value);
            if (a >= b) { if (which === "from") a = b - 1; else b = a + 1; }
            a = Math.max(0, a); b = Math.min(aefYears.length - 1, b);
            y0 = aefYears[a]; y1 = aefYears[b]; styleAef();
          };
          rFrom.addEventListener("input", () => onDrag("from"));
          rTo.addEventListener("input", () => onDrag("to"));
          let aefSent = [y0, y1];
          const aefRelease = () => { if (y0 !== aefSent[0] || y1 !== aefSent[1]) { aefSent = [y0, y1]; send("aef"); } };
          rFrom.addEventListener("change", aefRelease);
          rTo.addEventListener("change", aefRelease);
          try { new ResizeObserver(styleAef).observe(rng); } catch (e) {}
          // the S2 row: year and gamma
          newRow(head);
          const rowS2 = head.lastElementChild;
          const s2Years = cfg.s2_years || [];
          const yrWrap = document.createElement("span"); yrWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const yr = document.createElement("span"); yr.className = "sp-scale sp-year";
          const yrTrk = document.createElement("span"); yrTrk.className = "trk";
          const yrSpn = document.createElement("span"); yrSpn.className = "spn";
          const yrTks = document.createElement("span"); yrTks.className = "tks";
          for (const y of s2Years) { const t = document.createElement("span"); const i = document.createElement("i"); i.style.fontStyle = "normal"; i.textContent = "\u2019" + String(y).slice(-2); t.appendChild(i); yrTks.appendChild(t); }
          const yri = document.createElement("input"); yri.type = "range"; yri.min = 0; yri.max = Math.max(0, s2Years.length - 1); yri.step = 1;
          yri.title = "which Sentinel-2 yearly mosaic is drawn ([ and ])";
          const yrTxt = document.createElement("span"); yrTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.6em;margin-right:.6rem";
          yr.append(yrTrk, yrSpn, yrTks, yri);
          yrWrap.append(eyebrow("S2"), yr, yrTxt);
          rowS2.appendChild(yrWrap);
          const styleS2 = () => {
            const i = Math.max(0, s2Years.indexOf(s2y)), n = Math.max(1, s2Years.length - 1);
            yri.value = i;
            yrSpn.style.width = Math.max(0, (yr.clientWidth - 16) * i / n) + "px";
            yrTxt.textContent = String(s2y);
          };
          let yrSent = s2y, yrTimer = null;
          const yrRelease = () => { if (yrTimer) { clearTimeout(yrTimer); yrTimer = null; } if (s2y !== yrSent) { yrSent = s2y; send("s2"); } };
          yri.addEventListener("input", () => { s2y = s2Years[Number(yri.value)]; styleS2(); if (yrTimer) clearTimeout(yrTimer); yrTimer = setTimeout(yrRelease, 150); });
          yri.addEventListener("change", yrRelease);
          try { new ResizeObserver(styleS2).observe(yr); } catch (e) {}
          const SC_MIN = 0.2, SC_MAX = 3;
          const scWrap = document.createElement("span"); scWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const scr = document.createElement("span"); scr.className = "sp-scale";
          const scTrk = document.createElement("span"); scTrk.className = "trk";
          const scSpn = document.createElement("span"); scSpn.className = "spn";
          const sc = document.createElement("input"); sc.type = "range"; sc.min = SC_MIN; sc.max = SC_MAX; sc.step = 0.1;
          sc.title = "gamma of the Sentinel-2 mosaic: above 1 lifts the midtones (double-click for 1.0)";
          const scTxt = document.createElement("span"); scTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.4em;margin-right:.6rem";
          scr.append(scTrk, scSpn, sc);
          scWrap.append(eyebrow("gamma"), scr, scTxt);
          rowS2.appendChild(scWrap);
          const styleSc = () => {
            sc.value = s2scale;
            scSpn.style.width = Math.max(0, (scr.clientWidth - 16) * (s2scale - SC_MIN) / (SC_MAX - SC_MIN)) + "px";
            scTxt.textContent = "γ " + s2scale.toFixed(1);
          };
          let scSent = s2scale, scTimer = null;
          const scRelease = () => { if (scTimer) { clearTimeout(scTimer); scTimer = null; } if (s2scale !== scSent) { scSent = s2scale; send("s2scale"); } };
          sc.addEventListener("input", () => { s2scale = Number(sc.value); styleSc(); if (scTimer) clearTimeout(scTimer); scTimer = setTimeout(scRelease, 150); });
          sc.addEventListener("change", scRelease);
          scr.addEventListener("dblclick", (e) => { e.preventDefault(); s2scale = 1; styleSc(); scRelease(); });
          try { new ResizeObserver(styleSc).observe(scr); } catch (e) {}
          let layout = cfg.layout === "pair" ? "pair" : "slider";
          const gLayout = mkGroup(head, "view", [
            {value: "slider", label: "slider", title: "one map: S2 left of a divider you drag, the data right of it"},
            {value: "pair", label: "pair", title: "two maps side by side, one camera: S2 on the left, the data on the right"},
          ], () => layout, (v) => { layout = v; applyLayout(); }, "layout", "sp-layout");
          // which rows show: one per layer that is on
          const rows = () => {
            rowBld.style.display = on.bld ? "" : "none";
            rowHex.style.display = on.hex ? "" : "none";
            rowS2.style.display = on.s2 ? "" : "none";
            gLayers.style();
            applyLayout();
            setTimeout(() => { styleAef(); styleS2(); styleSc(); }, 0);
          };
          rows();
          // the geocoder (Photon, browser only), on the first row
          const PHOTON = "https://photon.komoot.io/api/";
          const gcWrap = document.createElement("span");
          gcWrap.style.cssText = "position:relative;display:inline-flex;align-items:center;gap:.4rem";
          const gc = document.createElement("input");
          gc.type = "search"; gc.placeholder = "a place…"; gc.autocomplete = "off"; gc.spellcheck = false;
          gc.title = "Photon geocoder: type, pick a hit (arrows, Enter, or click) and the map flies there";
          gc.style.cssText = "width:11rem;" + font + ";padding:.15rem .45rem;border:1px solid rgba(29,29,27,.28);border-radius:5px;background:#fff;color:#1d1d1b";
          const gcList = document.createElement("div");
          gcList.className = "sp-hits";
          gcList.style.cssText = "position:absolute;left:0;top:calc(100% + 4px);z-index:9;display:none;min-width:100%;max-width:26rem;" +
            "background:#fff;color:#1d1d1b;border:1px solid rgba(29,29,27,.28);border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.18);overflow:hidden";
          gcWrap.append(eyebrow("find"), gc, gcList);
          head.firstElementChild.appendChild(gcWrap);
          let gcHits = [], gcSel = -1, gcTimer = null, gcSeq = 0;
          const hitName = (f) => {
            const p = f.properties || {};
            const parts = [p.name, p.street && !p.name ? p.street : null, p.city && p.city !== p.name ? p.city : null,
              p.county && p.county !== p.city && p.county !== p.name ? p.county : null, p.state, p.country];
            return parts.filter((x) => x).join(", ");
          };
          const hitKind = (f) => { const p = f.properties || {}; return [p.osm_value, p.type].filter((x) => x && x !== "yes").join(" · "); };
          const gcHide = () => { gcList.style.display = "none"; gcList.replaceChildren(); gcSel = -1; };
          const gcShow = () => {
            gcList.replaceChildren();
            if (!gcHits.length) { gcHide(); return; }
            gcHits.forEach((f, i) => {
              const row = document.createElement("div");
              row.style.cssText = "padding:.3rem .55rem;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.3;" + (i === gcSel ? "background:" + ACCENT + ";color:#fff" : "");
              const nm = document.createElement("div"); nm.textContent = hitName(f);
              const kd = document.createElement("div"); kd.textContent = hitKind(f); kd.style.cssText = "font-size:11px;opacity:" + (i === gcSel ? ".85" : ".6");
              row.append(nm, kd);
              row.onmousedown = (e) => { e.preventDefault(); gcFly(f); };
              row.onmouseenter = () => { gcSel = i; gcShow(); };
              gcList.appendChild(row);
            });
            gcList.style.display = "block";
          };
          const gcAsk = async () => {
            const q = gc.value.trim();
            if (q.length < 2) { gcHits = []; gcHide(); return; }
            const seq = ++gcSeq;
            const params = new URLSearchParams({q, limit: "6", lang: "en"});
            if (map) { const c = map.getCenter(); params.set("lon", c.lng.toFixed(4)); params.set("lat", c.lat.toFixed(4)); }
            try {
              const r = await fetch(PHOTON + "?" + params.toString());
              const data = await r.json();
              if (seq !== gcSeq) return;
              gcHits = (data.features || []).filter((f) => f.geometry && f.geometry.coordinates);
              gcSel = gcHits.length ? 0 : -1;
              gcShow();
            } catch (e) { if (seq === gcSeq) say("search: " + e.message); }
          };
          const gcFly = (f) => {
            const [lon, lat] = f.geometry.coordinates;
            const ext = (f.properties || {}).extent;
            const w = (mapEl.clientWidth || 1200);
            let zoom = 12;
            if (ext && ext.length === 4) {
              const span = Math.max(Math.abs(ext[2] - ext[0]), Math.abs(ext[1] - ext[3]) * 2, 0.01);
              zoom = Math.log2(360 * (w / 512) / span) - 0.3;
            }
            zoom = Math.max(3.5, Math.min(16, zoom));
            gc.value = hitName(f); gcHits = []; gcHide(); gc.blur();
            if (map) map.flyTo({center: [lon, lat], zoom, duration: 2000});
            say("→ " + hitName(f) + " · zoom " + zoom.toFixed(1));
          };
          gc.addEventListener("input", () => { if (gcTimer) clearTimeout(gcTimer); gcTimer = setTimeout(gcAsk, 250); });
          gc.addEventListener("focus", () => { if (gcHits.length) gcShow(); });
          gc.addEventListener("blur", () => { setTimeout(gcHide, 120); });
          gc.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "ArrowDown" && gcHits.length) { gcSel = (gcSel + 1) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "ArrowUp" && gcHits.length) { gcSel = (gcSel - 1 + gcHits.length) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "Enter") {
              e.preventDefault();
              if (gcHits.length) gcFly(gcHits[Math.max(0, gcSel)]);
              else { if (gcTimer) clearTimeout(gcTimer); gcAsk().then(() => { if (gcHits.length) gcFly(gcHits[0]); else say("no match: " + gc.value.trim()); }); }
            }
            else if (e.key === "Escape") { gcHide(); gc.blur(); }
          });
          // full screen (the widget lives in a shadow root: walk down to compare)
          const isFull = () => {
            let fe = document.fullscreenElement;
            while (fe && fe.shadowRoot && fe.shadowRoot.fullscreenElement) fe = fe.shadowRoot.fullscreenElement;
            return fe === root;
          };
          // "full screen in the window" (Stephen, 2026-09-24): the widget pinned
          // over the page, not the OS full screen (that stays on the map's button)
          let fit = false;
          // marimo's own buttons (the notebook "..." menu marimo run pins to the
          // top right; in edit mode the cell "..." actions, drag, expand) float
          // over the map's top-right controls while the widget fills the
          // window (Stephen, 2026-09-24: "can you hide this button"); a page
          // style, on only while fit is on
          const FIT_CLS = "sp-fit-on";
          if (!document.getElementById("sp-fit-style")) {
            const st = document.createElement("style");
            st.id = "sp-fit-style";
            st.textContent = ["notebook-actions-dropdown", "cell-actions-button", "drag-button", "expand-output-button", "fullscreen-output-button"]
              .map((t) => "html." + FIT_CLS + " [data-testid='" + t + "']").join(",") + "{display:none!important}";
            document.head.appendChild(st);
          }
          const paneHeight = () => {
            document.documentElement.classList.toggle(FIT_CLS, fit && !isFull());
            const full = isFull() || fit;
            root.style.position = fit && !isFull() ? "fixed" : (full ? "relative" : "");
            root.style.inset = fit && !isFull() ? "0" : "";
            root.style.zIndex = fit && !isFull() ? "9999" : "";
            root.style.height = full ? "100vh" : "";
            root.style.boxSizing = "border-box";
            pane.style.height = full ? "100vh" : (cfg.height || 720) + "px";
            strip.style.cssText = full
              ? stripCss + ";position:absolute;left:0;right:0;bottom:0;z-index:30;background:rgba(255,255,255,.8);backdrop-filter:blur(3px);-webkit-backdrop-filter:blur(3px);border-top:1px solid rgba(29,29,27,.18);max-height:40vh;overflow-y:auto;box-sizing:border-box"
              : stripCss;
            styleAef();
            setTimeout(() => { if (map) map.resize(); }, 30);
          };
          const toggleFit = () => { fit = !fit; paneHeight(); };
          const toggleFull = () => { if (isFull()) document.exitFullscreen(); else root.requestFullscreen().catch((e) => say("fullscreen: " + e.message)); };
          document.addEventListener("fullscreenchange", () => { setTimeout(paneHeight, 30); });
          window.addEventListener("resize", () => { paneHeight(); });
          const hint = document.createElement("div");
          hint.style.cssText = mono + ";opacity:.55;color:#666";
          hint.textContent = "keys: B buildings · W WSF · A AEF · S S2 · F first/last S2 year · [ ] S2 year · ; ' S2 gamma · 1-9 the fill of the last row shown · - = window from · _ + window to · Q next building fill · L labels · X fill the window (Esc back) · click for the account";
          strip.appendChild(hint);
          hint.hidden = !!cfg.minimal;
          const step = (arr, cur, d) => { const i = arr.indexOf(cur); return arr[Math.max(0, Math.min(arr.length - 1, (i < 0 ? 0 : i) + d))]; };
          root.tabIndex = 0;
          root.addEventListener("pointerup", (e) => {
            if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
            setTimeout(() => { try { root.focus({preventScroll: true}); } catch (err) {} }, 0);
          });
          const flip = (k) => { toggleOn(k); rows(); update(); send("layers"); };
          root.addEventListener("keydown", (e) => {
            if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
            const k = e.key;
            if (k === "[" || k === "]" || k === "ArrowLeft" || k === "ArrowRight") { s2y = step(s2Years, s2y, (k === "]" || k === "ArrowRight") ? 1 : -1); styleS2(); yrRelease(); }
            else if ((k === "f" || k === "F" || k === "\\") && s2Years.length) { const lo = s2Years[0], hi = s2Years[s2Years.length - 1]; s2y = (s2y === hi) ? lo : hi; styleS2(); yrRelease(); }  // first <-> last S2 year
            else if (k === ";" || k === "'") { s2scale = Math.round(10 * Math.max(SC_MIN, Math.min(SC_MAX, s2scale + (k === "'" ? 0.1 : -0.1)))) / 10; styleSc(); scRelease(); }
            else if (k >= "1" && k <= "9") {
              if (on.bld && bldZoomOk()) { const f = bfills[Number(k) - 1]; if (f) { bfill = f.value; gBFill.style(); send("bfill"); } }
              else if (on.hex) { const f = fills[Number(k) - 1]; if (f) { fill = f.value; gFill.style(); send("fill"); } }
            }
            else if (k === "-" || k === "=") { const v = step(aefYears, y0, k === "=" ? 1 : -1); if (v < y1) { y0 = v; styleAef(); aefRelease(); } }
            else if (k === "_" || k === "+") { const v = step(aefYears, y1, k === "+" ? 1 : -1); if (v > y0) { y1 = v; styleAef(); aefRelease(); } }
            else if (k === "b" || k === "B") { flip("bld"); }
            else if (k === "w" || k === "W") { flip("wsf"); }
            else if (k === "a" || k === "A" || k === "e" || k === "E") { flip("hex"); }
            else if (k === "s" || k === "S" || k === "i" || k === "I") { flip("s2"); }
            else if (k === "l" || k === "L") { labelsOn = !labelsOn; labels(labelsOn); send("labels"); }
            else if (k === "x" || k === "X") { toggleFit(); }
            else if (k === "Escape" && fit) { toggleFit(); }
            else if (k === "q" || k === "Q") { if (bfills.length) { const i = bfills.findIndex((f) => f.value === bfill); bfill = bfills[(i + 1) % bfills.length].value; gBFill.style(); send("bfill"); } }  // next building fill
            else return;
            e.preventDefault();
          });
          const say = (t) => {
            status.textContent = t || "";
            if (cfg.minimal) status.hidden = !/folding|reading|failed|zoom in|no match|search:/.test(t || "");  // only while working or failing; the finished tallies stay in the kernel (Stephen, 2026-09-24: "get rid of it")
          };
          const renderLegend = () => {
            legend.replaceChildren();
            let items = [];
            try { items = JSON.parse(model.get("legend") || "[]"); } catch (e) { items = []; }
            const rowsOk = items.length > 0 && items.every((it) => !it.ramp && (it.pct != null || !it.hex));
            if (rowsOk) {
              const card = document.createElement("div");
              card.className = "sp-rows";
              card.style.cssText = "display:flex;flex-wrap:wrap;gap:.2rem .9rem;font-size:13px;font-variant-numeric:tabular-nums";
              for (const it of items) {
                if (!it.hex) { const t = eyebrow(it.name); t.style.marginRight = ".2rem"; card.appendChild(t); continue; }
                const row = document.createElement("div");
                row.style.cssText = "display:flex;align-items:center;gap:.45rem;white-space:nowrap";
                const chip = document.createElement("span");
                chip.style.cssText = "display:inline-block;flex:0 0 12px;width:12px;height:12px;border-radius:2px;background:" + it.hex;
                const t = document.createElement("span"); t.textContent = it.name + " " + it.pct + "%";
                row.append(chip, t); card.appendChild(row);
              }
              legend.appendChild(card);
              return;
            }
            for (const it of items) {
              if (!it.hex && !it.ramp) { legend.appendChild(eyebrow(it.name)); continue; }
              const s = document.createElement("span");
              s.style.cssText = "display:inline-flex;align-items:center;gap:.35rem";
              if (it.ramp) {
                const bar = document.createElement("span");
                bar.style.cssText = "display:inline-block;width:11rem;height:12px;border-radius:2px;background:linear-gradient(90deg," + it.ramp.join(",") + ")";
                const lo = document.createElement("span"); lo.textContent = it.lo; lo.style.opacity = ".75";
                const hi = document.createElement("span"); hi.textContent = it.hi; hi.style.opacity = ".75";
                s.append(lo, bar, hi); s.title = it.title || "";
              } else {
                const chip = document.createElement("span");
                chip.style.cssText = "display:inline-block;width:12px;height:12px;border-radius:2px;background:" + it.hex;
                const t = document.createElement("span"); t.textContent = it.name + (it.pct != null ? " " + it.pct + "%" : "");
                s.append(chip, t);
              }
              legend.appendChild(s);
            }
          };
          model.on("change:status", () => say(model.get("status")));
          model.on("change:panel", () => { panel.innerHTML = model.get("panel") || ""; });
          model.on("change:legend", renderLegend);

          // ---- the data ------------------------------------------------------
          let hexes = [], N = 0, colors = null, res = -1, hexIndex = new Map(), dataObj = null;
          const raw = {cells: null, colors: null};
          const grab = (k) => {
            try { const u8 = bytesOf(model.get(k)); raw[k] = u8 && u8.length ? copyOf(u8) : null; }
            catch (e) { raw[k] = null; say("grab " + k + ": " + e.message); }
          };
          function loadCells() {
            const buf = raw.cells;
            if (!buf || !buf.byteLength) { hexes = []; N = 0; hexIndex = new Map(); res = -1; return; }
            const ids = new BigUint64Array(buf);
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            try { res = getResolution(hexes[0]); } catch (e) { res = -1; }
          }
          function loadAttrs() {
            const c8 = raw.colors;
            colors = c8 && c8.byteLength === N * 4 ? new Uint8Array(c8) : null;
            dataObj = N && colors ? {length: N} : null;
          }
          // the footprints: a GeoArrow table (one batch) and their fill as an
          // arrow FixedSizeList<u8, 4> vector over the same rows, rebuilt only
          // when either arrives, so a camera move never touches them
          let polys = null, bcolors = null, bvec = null, polysSeq = 0;
          const RGBA = new arrow.FixedSizeList(4, new arrow.Field("rgba", new arrow.Uint8(), false));
          const mkBVec = () => {
            const n = polys ? polys.numRows : 0;
            bvec = n && bcolors && bcolors.length === 4 * n
              ? arrow.makeVector(arrow.makeData({type: RGBA, length: n, nullCount: 0,
                  child: arrow.makeData({type: new arrow.Uint8(), length: 4 * n, nullCount: 0, data: bcolors})}))
              : null;
          };
          const loadPolys = () => {
            const u8 = bytesOf(model.get("polys"));
            polysSeq++;
            polys = null;
            if (u8 && u8.length) {
              try { polys = arrow.tableFromIPC(new Uint8Array(copyOf(u8))); } catch (e) { say("footprints: " + e.message); }
            }
            mkBVec(); update();
          };
          const loadBColors = () => {
            const u8 = bytesOf(model.get("bcolors"));
            bcolors = u8 && u8.length ? new Uint8Array(copyOf(u8)) : null;
            mkBVec(); update();
          };

          // ---- tiles: ask the kernel ------------------------------------------
          const pending = new Map();
          let tseq = 0;
          const tstat = {asked: 0, got: 0, empty: 0, err: 0, abort: 0};
          model.on("msg:custom", (msg, buffers) => {
            if (!msg || msg.kind !== "tile") return;
            const p = pending.get(msg.id);
            if (!p) return;
            pending.delete(msg.id);
            if (msg.err) { tstat.err++; say("tile: " + msg.err); p.reject(new Error(msg.err)); return; }
            if (msg.empty || !buffers || !buffers.length) { tstat.empty++; p.resolve(null); return; }
            const u8 = bytesOf(buffers[0]);
            createImageBitmap(new Blob([u8], {type: "image/png"})).then(
              (b) => { tstat.got++; p.resolve(b); },
              (e) => { tstat.err++; p.reject(e instanceof Error ? e : new Error("decode")); });
          });
          const getTileDataFor = (src, year) => ({index, signal}) => new Promise((resolve, reject) => {
            const id = ++tseq;
            tstat.asked++;
            pending.set(id, {resolve, reject});
            model.send({kind: "tile", id, src, year, x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => {
              pending.delete(id); tstat.abort++;
              const e = new Error("aborted"); e.name = "AbortError"; reject(e);
            });
          });

          // ---- the layers ----------------------------------------------------
          const slot = () => { const want = cfg.labels_slot || "watername_ocean"; const st = map && map.getStyle && map.getStyle(); if (!st || !st.layers || st.layers.some((x) => x.id === want)) return want; const l = st.layers.find((x) => x.type === "symbol"); return (l && l.id) || want; };
          const ring = (h) => { try { return cellToBoundary(h, true); } catch (e) { return null; } };
          const ADMIN = ["region", "county"];
          const aFill = (k) => "ov-div-" + k;
          const adminSync = (m) => {
            if (!m || !cfg.div_pm || m.getSource("ov-div")) return;
            try {
              m.addSource("ov-div", {type: "vector", url: "pmtiles://" + cfg.div_pm});
              for (const k of ADMIN) m.addLayer({id: aFill(k), type: "fill", source: "ov-div", "source-layer": "division_area",
                filter: ["all", ["==", ["get", "subtype"], k], ["==", ["get", "class"], "land"]], paint: {"fill-opacity": 0}}, slot());
            } catch (e) { console.error("divisions", e); }
          };
          const adminAt = (m, pt) => {
            const out = {};
            const one = (k) => {
              if (!m.getLayer(aFill(k))) return null;
              const fs = m.queryRenderedFeatures(pt, {layers: [aFill(k)]});
              return fs && fs.length ? fs[0].properties : null;
            };
            try {
              const r = one("region"); if (r) out.region = r["@name"] || r.names || null;
              const c = one("county"); if (c) out.county = c["@name"] || c.names || null;
            } catch (e) {}
            return out;
          };
          const outline = (id, h, color, width) => {
            const r = h ? ring(h) : null;
            if (!r) return null;
            return new PathLayer({id, data: [r], getPath: (d) => d, getColor: color, widthUnits: "pixels", getWidth: width, widthMinPixels: 1, beforeId: slot()});
          };
          const mkRaster = (src, year, maxZ, extent, visible, clip) => new TileLayer({
            ...(clip || {}),
            id: src + "-" + year + (src === "s2" && cfg.s2_gen ? "-s" + cfg.s2_gen : ""),
            getTileData: getTileDataFor(src, year),
            onTileError: (e) => { if (!e || e.name !== "AbortError") say(src + " tile: " + ((e && e.message) || e)); },
            tileSize: cfg.tile || 256, minZoom: 0, maxZoom: maxZ, extent: extent || null, visible,
            refinementStrategy: "best-available", beforeId: slot(),
            renderSubLayers: (p) => {
              if (!p.data) return null;
              const {west, south, east, north} = p.tile.bbox;
              return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]});
            },
          });
          const hexZoomOk = () => !!map && map.getZoom() >= (cfg.hex_zoom || 9);
          const bldZoomOk = () => !!map && map.getZoom() >= (cfg.bld_zoom || 13);
          const swipeLon = () => {
            if (!map) return null;
            const w = mapEl.clientWidth, h = mapEl.clientHeight;
            return map.unproject([swipeF * w, h / 2]).lng;
          };
          const clipTo = (side) => {
            if (!on.s2 || paired()) return {};
            const x = swipeLon();
            if (x == null) return {};
            const b = side === "left" ? [x - 360, -85, x, 85] : [x, -85, x + 360, 85];
            return {extensions: [new ClipExtension()], clipBounds: b};
          };
          function layers() {
            const out = [];
            swipe.style.display = on.s2 && !paired() ? "" : "none";
            // bottom to top: S2, hexagons, WSF raster, footprints, the rings. The
            // raster is above the hexagons (Stephen, 2026-09-24: "wsf has a big
            // building here i cant see"): its pixels are only where something is
            // built, so the hexagons show through everywhere else
            out.push(mkRaster("s2", cfg.s2_year, 14, null, !!on.s2 && !paired(), clipTo("left")));
            if (on.hex && dataObj && hexZoomOk()) out.push(new H3HexagonLayer({
              ...clipTo("right"),
              id: "hexes", data: {length: N},
              getHexagon: (_, {index}) => hexes[index],
              getFillColor: (_, {index}) => [colors[4 * index], colors[4 * index + 1], colors[4 * index + 2], colors[4 * index + 3]],
              updateTriggers: {getFillColor: [dataObj], getHexagon: [dataObj]},
              filled: true, stroked: false, extruded: false, highPrecision: true, pickable: false, beforeId: slot(),
            }));
            out.push(mkRaster("wsf", 0, 14, cfg.extent || null, !!on.wsf, clipTo("right")));
            const bld = on.bld && bldZoomOk() && polys && bvec;
            // the footprints span both sides of the swipe: the photo is what a
            // footprint is checked against (Stephen, 2026-09-24)
            if (bld) out.push(new GeoArrowPolygonLayer({
              id: "footprints-" + polysSeq, data: polys, filled: true, stroked: true,
              getFillColor: bvec,
              getLineColor: cfg.bld_stroke || [255, 214, 120, 230],
              lineWidthUnits: "pixels", getLineWidth: 1.0, lineWidthMinPixels: 0.8,
              pickable: false, beforeId: slot(),
            }));
            const h = on.hex ? outline("hover", hover, [255, 255, 255, 255], 2) : null;
            if (h) out.push(h);
            const pk = cfg.hit ? outline("picked", cfg.hit, [255, 200, 40, 255], 3) : null;
            if (pk) out.push(pk);
            return out;
          }
          function update() {
            if (ov) ov.setProps({layers: layers()});
            if (ovS) ovS.setProps({layers: paired() ? [mkRaster("s2", cfg.s2_year, 14, null, true, null)] : []});
          }
          function labels(onL) {
            for (const m of [map, mapS]) {
              if (!m || !m.isStyleLoaded()) continue;
              const st = m.getStyle();
              if (!st || !st.layers) continue;
              st.layers.forEach((l) => { if (l.layout && l.layout["text-field"] !== undefined) m.setLayoutProperty(l.id, "visibility", onL ? "visible" : "none"); });
            }
          }
          // ---- PAIR: the S2 map on the left, the camera shared -----------------
          function paired() { return !!on.s2 && layout === "pair"; }
          const follow = (a, b) => {
            if (syncing || !a || !b) return;
            syncing = true;
            try { b.jumpTo({center: a.getCenter(), zoom: a.getZoom(), bearing: a.getBearing(), pitch: a.getPitch()}); }
            finally { syncing = false; }
          };
          function ensureS2Map() {
            if (mapS || !map) return;
            mapS = new maplibregl.Map({container: mapElS, style: STYLE, center: map.getCenter(), zoom: map.getZoom(), attributionControl: false});
            mapS.keyboard.disable();
            ovS = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck: " + (e && e.message ? e.message : e))});
            mapS.addControl(ovS);
            mapS.on("load", () => { labels(labelsOn); update(); });
            mapS.on("move", () => { if (paired()) follow(mapS, map); });
            map.on("move", () => { if (paired()) follow(map, mapS); });
            new ResizeObserver(() => { try { mapS.resize(); } catch (e) {} }).observe(mapElS);
          }
          function applyLayout() {
            const p = paired();
            mapEl.style.left = p ? "50%" : "0";
            mapElS.style.display = p ? "" : "none";
            if (p) ensureS2Map();
            if (gLayout) gLayout.style();
            setTimeout(() => {
              try { if (map) map.resize(); if (mapS && p) { mapS.resize(); follow(map, mapS); } } catch (e) {}
              update(); sendView();
            }, 30);
          }
          let seq = 0, lastView = "";
          function sendView() {
            if (!map) return;
            const c = map.getCenter();
            const v = {longitude: c.lng, latitude: c.lat, zoom: map.getZoom(), w: mapEl.clientWidth, h: mapEl.clientHeight};
            const key = JSON.stringify(v);
            if (key === lastView) return;
            lastView = key;
            v.n = ++seq;
            model.set("view", JSON.stringify(v));
            model.save_changes();
            if (on.bld && v.zoom >= (cfg.bld_zoom || 13)) say("reading footprints…");
            else if (on.hex && v.zoom >= (cfg.hex_zoom || 9)) say("folding hexagons…");
          }
          const cellAt = (lngLat) => {
            if (res < 0) return null;
            try { const h = latLngToCell(lngLat.lat, lngLat.lng, res); return hexIndex.has(h) ? h : null; }
            catch (e) { return null; }
          };
          function boot() {
            const home = cfg.home || {longitude: -96, latitude: 38.5, zoom: 4};
            map = new maplibregl.Map({container: mapEl, style: STYLE, center: [home.longitude, home.latitude], zoom: home.zoom, attributionControl: {compact: false}});
            map.keyboard.disable();
            map.addControl(new maplibregl.FullscreenControl({container: root}), "top-right");
            map.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
            ov = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck: " + (e && e.message ? e.message : e))});
            map.addControl(ov);
            map.on("load", () => {
              // the basemap draws its own footprints; ours are the data (Stephen, 2026-09-24: "basemap footprints and our data footprints")
              (map.getStyle().layers || []).forEach((l) => { if (l["source-layer"] === "building" || /^building/.test(l.id)) map.setLayoutProperty(l.id, "visibility", "none"); });
              labels(labelsOn); adminSync(map); applyLayout();
            });
            map.on("moveend", sendView);
            map.on("zoom", () => update());
            map.on("move", () => { if (on.s2) update(); });
            map.on("mousemove", (e) => { if (!on.hex) return; const h = cellAt(e.lngLat); if (h !== hover) { hover = h; update(); } });
            map.on("mouseout", () => { if (hover) { hover = null; update(); } });
            map.on("click", (e) => {
              const h = on.hex ? cellAt(e.lngLat) : null;
              model.set("pick", JSON.stringify({cell: h, lon: e.lngLat.lng, lat: e.lngLat.lat, admin: adminAt(map, e.point), n: ++seq}));
              model.save_changes();
            });
            map.on("error", (ev) => { if (ev && ev.error && ev.error.message) say("map: " + ev.error.message); });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} }).observe(mapEl);
            window.__spTiles = tstat;
            window.__spMaps = () => [map, mapS].filter(Boolean);
            window.__spLayers = () => ({right: layers().filter((l) => l.props.visible !== false).map((l) => l.id), N, res, on: Object.assign({}, on), polys: polys ? polys.numRows : 0});
          }
          let pendingLoad = null, needCells = false;
          const flush = () => {
            pendingLoad = null;
            try { if (needCells) loadCells(); needCells = false; loadAttrs(); update(); }
            catch (e) { say("load: " + e.message); console.error(e); }
          };
          const reload = () => { needCells = true; if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          const reattr = () => { if (!pendingLoad) pendingLoad = setTimeout(flush, 0); };
          model.on("change:cells", () => { grab("cells"); reload(); });
          model.on("change:colors", () => { grab("colors"); reattr(); });
          model.on("change:polys", loadPolys);
          model.on("change:bcolors", loadBColors);
          model.on("change:config", () => {
            const was = cfg;
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (Number(cfg.s2_scale) && Number(cfg.s2_scale) !== s2scale) { s2scale = Number(cfg.s2_scale); scSent = s2scale; styleSc(); }
            if (cfg.fill && cfg.fill !== fill) { fill = cfg.fill; gFill.style(); }
            if (cfg.bfill && cfg.bfill !== bfill) { bfill = cfg.bfill; gBFill.style(); }
            if (cfg.layers) { let ch = false; for (const k in cfg.layers) if (!!cfg.layers[k] !== !!on[k]) { on[k] = !!cfg.layers[k]; ch = true; } if (ch) rows(); }
            if (cfg.labels !== was.labels) { labelsOn = cfg.labels !== false; labels(labelsOn); }
            update();
          });
          try { grab("cells"); grab("colors"); loadCells(); loadAttrs(); renderLegend(); say(model.get("status")); boot(); loadPolys(); loadBColors(); }
          catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { document.documentElement.classList.remove(FIT_CLS); try { map && map.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (OneMap,)


@app.cell
def _(
    AEF_FROM0,
    AEF_TO0,
    AEF_YEARS_ALL,
    BFILL0,
    BFILLS,
    BFILL_NAMES,
    BFILL_SHORT,
    BLD_STROKE,
    BLD_ZOOM,
    FILLS,
    FILL_NAMES,
    FILL_SHORT,
    HEX_ZOOM,
    HOME,
    LABELS_SLOT,
    LAYERS0,
    LAYER_DEFS,
    OV_DIV_PM,
    OneMap,
    RASTER_TILE,
    S2_SCALE0,
    S2_YEAR0,
    S2_YEARS,
    STRIP_MINIMAL,
    VIEW_H,
    json,
    wsf_bounds,
):
    # ---- the map: built ONCE, empty; never re-runs for a parameter ---------------
    pair = OneMap(config=json.dumps({
        "height": VIEW_H, "home": dict(HOME), "labels": True, "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
        "s2_year": S2_YEAR0, "s2_scale": S2_SCALE0, "s2_gen": 0, "fill": FILLS[0],
        "s2_years": list(S2_YEARS),
        "aef_from": AEF_FROM0, "aef_to": AEF_TO0, "aef_years": list(AEF_YEARS_ALL),
        "fills": [[f, FILL_SHORT[f], FILL_NAMES[f]] for f in FILLS],
        "hex_zoom": HEX_ZOOM, "extent": list(wsf_bounds),
        "minimal": STRIP_MINIMAL, "div_pm": OV_DIV_PM, "bld_zoom": BLD_ZOOM, "bld_stroke": list(BLD_STROKE),
        "bfill": BFILL0, "bfills": [[f, BFILL_SHORT[f], BFILL_NAMES[f]] for f in BFILLS],
        "layers": dict(LAYERS0), "layer_defs": [list(d) for d in LAYER_DEFS],
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "pending_force": False, "task": None, "loop": None,
        "s2y": S2_YEAR0, "s2scale": S2_SCALE0, "s2gen": 0, "fill": FILLS[0], "labels": True,
        "y0": AEF_FROM0, "y1": AEF_TO0,
        "hit": None, "pick_n": None, "panel_body": "", "memo": {}, "aef": {}, "wsf": {}, "h_cam": None, "h_ctl": None, "h_pick": None,
        "runs": 0,
        # the footprints under the box (`bld`), the box they were read for
        # (`obox`), the fill, and which layers are on
        "bld": None, "obox": None, "bfill": BFILL0, "on": dict(LAYERS0),
        "hex_status": "", "bld_status": "",
    }
    pair
    return HOLD, pair


@app.cell
def _(
    AEF_NODATA,
    AEF_YEARS_ALL,
    BFILLS,
    BFILL_NAMES,
    BLD_MAX,
    BLD_ZOOM,
    BLUE_RAMP,
    CELL_KM2,
    FA,
    FILLS,
    FILL_NAMES,
    HEX_ZOOM,
    HOLD,
    HOME,
    MIN_STABLE_CELLS,
    NEVER_RGB,
    NEW_MIN,
    OV_RELEASE,
    OV_RELEASE_HOW,
    S2_YEARS,
    SETTLE,
    SOURCE_RGB,
    STRIP_MINIMAL,
    WSF_DATE,
    WSF_RES,
    YEAR_RGB,
    aef_fold,
    aef_window,
    asyncio,
    bld_cover,
    bld_geoarrow,
    bld_rows,
    build_frame,
    con,
    contains,
    cpu,
    division_at,
    json,
    np,
    pad_box,
    pair,
    res_for_view,
    s2_raster_stats,
    s2_set_scale,
    s2_tile_png,
    time,
    traceback,
    view_to_bbox,
    wsf_fold,
    wsf_tile_png,
    wsf_window,
):
    # ---- wiring: the camera loop and the controls. Re-runs freely. ---------------
    # Two independent servers behind one camera: the hexagons (the pair's
    # fold, when that layer is on and the zoom allows) and the footprints
    # (when that layer is on and the zoom allows). Each holds its own box.
    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass
    HOLD["runs"] += 1

    async def _tile_fn(src, z, x, y, year):
        if src == "wsf":
            return await wsf_tile_png(z, x, y)
        return await s2_tile_png(z, x, y, year)

    pair.tile_fn = _tile_fn

    def _say(msg):
        # the browser writes its own "reading footprints…" on every settle and
        # only the kernel's next CHANGE replaces it; an unchanged status (a pan
        # inside the box already read) must still register, so a zero-width
        # space toggles on repeats (Stephen, 2026-09-24: "steady still reading
        # footprints")
        try:
            if pair.status == msg:
                msg = msg + "\u200b" if not msg.endswith("\u200b") else msg[:-1]
            pair.status = msg
        except Exception:
            pass

    def _cfg(**kw):
        c = json.loads(pair.config or "{}")
        c.update(kw)
        pair.config = json.dumps(c)

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

    _HEX = lambda c: "#%02x%02x%02x" % tuple(c[:3])

    # ---- the legend: whichever layer is on top and has something to say -----
    def _legend():
        b, fr, on = HOLD["bld"], HOLD["frame"], HOLD["on"]
        # the legend names its layer and fill first (Stephen, 2026-09-24: "wsf is
        # not selected still seeing legend for it": the buildings' WSF year fill
        # shares the raster's palette and labels)
        if on.get("bld") and b is not None and b.get("legend") is not None:
            pair.legend = json.dumps([{"name": f"buildings, {BFILL_SHORT[HOLD['bfill']]}"}] + b["legend"])
        elif on.get("hex") and fr is not None:
            pair.legend = json.dumps([{"name": f"AEF, {FILL_SHORT[HOLD['fill']]}"}] + fr["legend"](HOLD["fill"]))
        elif on.get("wsf"):
            pair.legend = json.dumps([{"name": "WSF, first built-up"}, {"name": "already built-up in 2016", "hex": _HEX(YEAR_RGB[2016])}]
                                     + [{"name": str(y), "hex": _HEX(YEAR_RGB[y])} for y in range(2017, 2026)])
        else:
            pair.legend = "[]"

    def _status():
        bits = [x for x in (HOLD.get("bld_status"), HOLD.get("hex_status")) if x]
        _say(" · ".join(bits) if bits else "")

    # ---- the hexagons (the pair's fold) ---------------------------------------
    def _hexes_off(msg=""):
        if HOLD["sent"] is not None:
            with pair.hold_sync():
                pair.cells, pair.colors = b"", b""
            HOLD["sent"] = None
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = None, None, None, None
        _cfg(hit=None)
        HOLD["hex_status"] = msg

    def _paint():
        fr = HOLD["frame"]
        if fr is None:
            return False
        rgba = fr["fill"](HOLD["fill"], HOLD["hit"])
        _cfg(hit=format(HOLD["hit"], "x") if HOLD["hit"] else None)
        with pair.hold_sync():
            if HOLD["sent"] is not fr:
                pair.cells = fr["cellid"].astype("<u8").tobytes()
                HOLD["sent"] = fr
            pair.colors = rgba.tobytes()
        _legend()
        return True

    async def _serve_hex(vsd, force=False):
        view = view_to_bbox(vsd)
        box = pad_box(view)
        inside = HOLD["box"] is not None and contains(HOLD["box"], view)
        if inside and not force and res_for_view(vsd, box) <= HOLD["res"]:
            return
        res = res_for_view(vsd, box)
        y0, y1 = HOLD["y0"], HOLD["y1"]
        rbox = tuple(round(v, 3) for v in box)
        key = (y0, y1, res, rbox)
        t0 = time.time()
        years = list(range(y0, y1 + 1))
        HOLD["hex_status"] = f"folding WSF and AlphaEarth {y0} to {y1} ({len(years)} years)…"
        _status()
        if key in HOLD["memo"]:
            fr, stats = HOLD["memo"][key]
        else:
            bkey = (res, rbox)
            need = [y for y in years if (y, bkey) not in HOLD["aef"]]
            wneed = bkey not in HOLD["wsf"]
            got = await asyncio.gather(
                wsf_fold(box, res) if wneed else asyncio.sleep(0, result=HOLD["wsf"].get(bkey)),
                *(aef_fold(box, res, y) for y in need),
            )
            nw, s1 = got[0]
            if wneed:
                HOLD["wsf"][bkey] = (nw, s1)
            for y, (tab, st) in zip(need, got[1:1 + len(need)]):
                HOLD["aef"][(y, bkey)] = (tab, st)
            for k_ in ("aef", "wsf"):
                if len(HOLD[k_]) > 40:
                    HOLD[k_].pop(next(iter(HOLD[k_])))
            if nw is None or nw.num_rows == 0:
                HOLD["hex_status"] = f"hexagons: res {res} · {s1}"
                return
            aef_by_year = {y: HOLD["aef"][(y, bkey)][0] for y in years if (y, bkey) in HOLD["aef"]}
            t1 = time.time()
            fr = await cpu(build_frame, nw, aef_by_year, y0, y1)
            stats = f"res {res} · {s1} · frame {time.time() - t1:.1f} s"
            HOLD["memo"][key] = (fr, stats)
            if len(HOLD["memo"]) > 12:
                HOLD["memo"].pop(next(iter(HOLD["memo"])))
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = fr, box, res, None
        _paint()
        HOLD["hex_status"] = f"hexagons: {stats} · {fr['score']} · {time.time() - t0:.1f} s"

    # ---- the footprints ----------------------------------------------------------
    _bstops = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in BLUE_RAMP], np.float64)
    _BLUES = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_bstops)), _bstops[:, k]) for k in range(3)], 1).round().astype(np.uint8)
    _BLUES_HEX = ["#%02x%02x%02x" % tuple(int(v) for v in _BLUES[i]) for i in range(0, 256, 17)]
    _NOJUMP_RGB, _NOEMB_RGB = (222, 222, 222), (170, 170, 170)

    def _cover_off():
        if HOLD["bld"] is None and HOLD["obox"] is None:
            return
        HOLD["bld"], HOLD["obox"] = None, None
        with pair.hold_sync():
            pair.polys, pair.bcolors = b"", b""

    async def _aef_for_bld(b):
        """AlphaEarth per footprint, on demand for the box: the mosaic window
        for every year, the footprints rasterized onto ITS grid, the mean
        vector per footprint per year (bincount per band), the steps
        (1 - cos between consecutive years), the quiet level from the
        footprints WSF says stood the whole time (index 1), and `ayear`:
        the first year a footprint's step cleared it (-1 none, -2 no
        embedding). Then the fill is repainted."""
        if b.get("aef_busy"):
            return
        b["aef_busy"] = True
        t0 = time.time()
        try:
            HOLD["bld_status"] = (HOLD.get("bld_status") or "") + f" · AlphaEarth: reading {len(AEF_YEARS_ALL)} years under the box…"
            _status()
            wins = await asyncio.gather(*(aef_window(b["box"], y) for y in AEF_YEARS_ALL))
            if HOLD["bld"] is not b:
                return
            n = b["rows"].num_rows
            years = [y for y, w in zip(AEF_YEARS_ALL, wins) if w is not None]
            if len(years) < 2 or n == 0:
                b["ayear"] = np.full(n, -2, np.int64)
                b["asteps"] = np.zeros((0, n), np.float32)
                b["ayears"] = years
                return
            w0 = next(w for w in wins if w is not None)
            _, lon0, lat0, px = w0
            shape = w0[0].shape[1:]
            coverA = await cpu(bld_cover, b["rows"], shape, lon0, lat0, px)
            flat = coverA.ravel()

            def _means(emb):
                e = emb.reshape(64, -1)
                ok = e[0] != AEF_NODATA
                cnt = np.bincount(flat[ok], minlength=n + 1)[1:].astype(np.float64)
                V = np.empty((n, 64), np.float64)
                x = e[:, ok].astype(np.float64) / 127.5
                x = np.sign(x) * x * x
                idx = flat[ok]
                for i in range(64):
                    V[:, i] = np.bincount(idx, weights=x[i], minlength=n + 1)[1:]
                V /= np.maximum(cnt, 1)[:, None]
                nrm = np.linalg.norm(V, axis=1)
                V /= np.maximum(nrm, 1e-9)[:, None]
                V[(cnt == 0) | (nrm == 0)] = np.nan
                return V

            Vs = await cpu(lambda: [_means(w[0]) for w in wins if w is not None])
            steps = np.full((len(years) - 1, n), np.nan, np.float32)
            for k in range(len(years) - 1):
                steps[k] = (1.0 - np.einsum("ij,ij->i", Vs[k], Vs[k + 1])).astype(np.float32)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                smax = np.nanmax(steps, axis=0)
            stepped = ~np.isnan(smax)
            quiet = stepped & (b["first"] == 1)
            ay = np.full(n, -2, np.int64)
            if quiet.sum() >= MIN_STABLE_CELLS:
                D0 = float(np.quantile(smax[quiet].astype(np.float64), 1 - FA))
                above = steps > D0
                firstk = above.argmax(axis=0)
                yrs = np.array(years[1:], np.int64)
                ay = np.where(above.any(axis=0), yrs[firstk], np.where(stepped, -1, -2)).astype(np.int64)
            else:
                D0 = float("nan")
                ay = np.where(stepped, -1, -2).astype(np.int64)
            b["ayear"], b["asteps"], b["ayears"], b["aD0"] = ay, steps, years, D0
            n_j = int((ay > 0).sum())
            HOLD["bld_status"] = (HOLD.get("bld_status") or "").split(" · AlphaEarth")[0] + (
                f" · AlphaEarth: {n_j:,} footprints jumped (quiet level {D0:.3f} from {int(quiet.sum()):,} pre-2016 footprints), {time.time() - t0:.1f} s"
                if not np.isnan(D0) else f" · AlphaEarth: too few pre-2016 footprints to set a quiet level ({int(quiet.sum())})")
            if HOLD["bfill"] == "ayear":
                _bld_paint()
            _status()
        except Exception as e:
            HOLD["bld_status"] = (HOLD.get("bld_status") or "").split(" · AlphaEarth")[0] + f" · AlphaEarth failed: {type(e).__name__}: {e}"
            _status()
        finally:
            b["aef_busy"] = False

    def _bld_paint():
        """rgba per footprint for HOLD["bfill"], and the legend. The polygons
        were sent with the box; this resends only the colours."""
        b = HOLD["bld"]
        if b is None:
            return False
        kind = HOLD["bfill"]
        n = b["rows"].num_rows
        first, px, built = b["first"], b["npx"], b["built"]
        a = np.full(n, 210, np.uint8)
        pct = lambda m: round(100 * int(m.sum()) / max(1, n), 1)
        if kind == "wyear":
            yrs = np.where(first > 0, 2016 + (first - 1) // 2, -9)
            rgb = np.array([YEAR_RGB[int(y)] if y > 0 else NEVER_RGB for y in yrs], np.uint8).reshape(n, 3)
            items = []
            for y in [2016] + list(range(2017, 2026)):
                m = yrs == y
                if m.any():
                    items.append({"name": ("already built-up in 2016" if y == 2016 else str(y)), "hex": _HEX(YEAR_RGB[y]), "pct": pct(m)})
            m = yrs < 0
            if m.any():
                items.append({"name": "never read as built-up by WSF", "hex": _HEX(NEVER_RGB), "pct": pct(m)})
        elif kind == "ayear":
            ay = b.get("ayear")
            if ay is None:
                # not computed yet for this box: ask for it, paint grey meanwhile
                rgb = np.tile(np.array(_NOEMB_RGB, np.uint8), (n, 1))
                items = [{"name": "AlphaEarth: reading the years…", "hex": _HEX(_NOEMB_RGB), "pct": 100.0}]
                _spawn(_aef_for_bld(b))
            else:
                rgb = np.array([YEAR_RGB[int(y)] if y > 0 else (_NOJUMP_RGB if y == -1 else _NOEMB_RGB) for y in ay], np.uint8).reshape(n, 3)
                items = []
                for y in range(2018, 2026):
                    m = ay == y
                    if m.any():
                        items.append({"name": f"fingerprint jumped in {y}", "hex": _HEX(YEAR_RGB[y]), "pct": pct(m)})
                m = ay == -1
                if m.any():
                    items.append({"name": "no jump 2017 to 2025 (under the quiet level)", "hex": _HEX(_NOJUMP_RGB), "pct": pct(m)})
                m = ay == -2
                if m.any():
                    items.append({"name": "no embedding", "hex": _HEX(_NOEMB_RGB), "pct": pct(m)})
        elif kind == "wshare":
            t = np.clip(built / np.maximum(px, 1), 0, 1)
            rgb = _BLUES[(t * 255).round().astype(np.int64)]
            items = [{"ramp": _BLUES_HEX, "lo": "0% of its pixels built-up", "hi": "100%", "title": BFILL_NAMES["wshare"]}]
        elif kind == "source":
            from collections import Counter
            ds = b["rows"]["dataset"].to_pylist()
            cnt = Counter(ds)
            order = [k for k, _ in cnt.most_common()]
            col = {k: SOURCE_RGB[min(i, len(SOURCE_RGB) - 1)] for i, k in enumerate(order)}
            rgb = np.array([col[d] for d in ds], np.uint8).reshape(n, 3)
            items = [{"name": (k or "unnamed source"), "hex": _HEX(col[k]), "pct": round(100 * cnt[k] / max(1, n), 1)} for k in order]
        else:
            up = b["rows"]["updated"].to_pylist()
            yrs = np.array([int(u[:4]) if u and u[:4].isdigit() else -9 for u in up], np.int64)

            def _c(y):
                if y >= 2016 and y in YEAR_RGB:
                    return YEAR_RGB[y]
                if y < 0:
                    return NEVER_RGB
                return (96, 96, 96) if y < 2008 else (128 + 8 * (y - 2016),) * 3
            rgb = np.array([_c(int(y)) for y in yrs], np.uint8).reshape(n, 3)
            items = [{"name": (f"touched {y}" if y > 0 else "no date"), "hex": _HEX(_c(y)), "pct": pct(yrs == y)} for y in sorted(set(int(v) for v in yrs))][-12:]
        rgba = np.ascontiguousarray(np.concatenate([rgb, a[:, None]], axis=1)).astype(np.uint8)
        b["legend"] = items
        pair.bcolors = rgba.tobytes()
        _legend()
        return True

    async def _serve_bld(vsd, force=False):
        """The Overture footprints under the padded box as polygons; the WSF
        window at native 10 m under the same box; the footprints rasterized
        onto it so each carries its pixel count, its built-up count and the
        earliest WSF date under it. The polygons go to the browser once per
        box; a fill is colours only. AlphaEarth is computed when asked for."""
        view = view_to_bbox(vsd)
        if HOLD["obox"] is not None and contains(HOLD["obox"], view) and not force:
            return
        box = pad_box(view)
        HOLD["bld_status"] = f"reading footprints (Overture {OV_RELEASE}) and WSF under the view…"
        _status()
        t0 = time.time()
        got = await asyncio.gather(wsf_window(0, box, 1), asyncio.to_thread(bld_rows, box))
        if got[0] is None:
            _cover_off()
            HOLD["bld_status"] = "WSF: nothing under the view (off the record, 60 S to 78 N)"
            return
        arr, lon, lat = got[0]
        rows = got[1]
        t1 = time.time()
        if rows.num_rows > BLD_MAX:
            _cover_off()
            HOLD["bld_status"] = f"Overture {OV_RELEASE}: more than {BLD_MAX:,} footprints under the view · zoom in"
            return
        psz = WSF_RES
        lon0, lat0 = float(lon[0]) - psz / 2, float(lat[0]) + psz / 2
        n_fp = rows.num_rows
        # the rasterizing (for the counts) and the GeoArrow encoding (for the
        # browser) need only the rows: side by side on the CPU pool
        cover, gj = await asyncio.gather(cpu(bld_cover, rows, arr.shape, lon0, lat0, psz), cpu(bld_geoarrow, rows))
        flat, aflat = cover.ravel(), arr.ravel()
        built_m = aflat > 0
        npx = np.bincount(flat, minlength=n_fp + 1)[1:]
        built = np.bincount(flat[built_m], minlength=n_fp + 1)[1:]
        first = np.zeros(n_fp, np.int64)
        if n_fp and built_m.any():
            idx, val = flat[built_m], aflat[built_m].astype(np.int64)
            o = np.lexsort((val, idx))
            idx, val = idx[o], val[o]
            keep = (idx > 0) & np.r_[True, idx[1:] != idx[:-1]]
            first[idx[keep] - 1] = val[keep]
        n_built_px = int(built_m.sum())
        n_gap_px = int((built_m & (flat == 0)).sum())
        n_never = int((built == 0).sum())
        by_ds = ""
        if n_fp:
            from collections import Counter
            vc = Counter(rows["dataset"].to_pylist()).most_common(4)
            by_ds = ", ".join(f"{k or 'unknown'} {v:,}" for k, v in vc)
        HOLD["bld"] = {"rows": rows, "cover": cover, "arr": arr, "lon0": lon0, "lat0": lat0, "psz": psz,
                       "first": first, "npx": npx, "built": built, "box": box, "legend": None}
        HOLD["obox"] = box
        with pair.hold_sync():
            pair.polys = gj
        HOLD["bld_status"] = (
            f"Overture {OV_RELEASE} ({OV_RELEASE_HOW}): {n_fp:,} footprints"
            + (f" ({by_ds})" if by_ds else "")
            + f" · {n_never:,} of them WSF has never read as built-up"
            + f" · WSF built-up pixels {n_built_px:,}, {100 * n_gap_px / max(1, n_built_px):.0f}% under no footprint"
            + f" · read {t1 - t0:.1f} s, {len(gj) / 1e6:.1f} MB of GeoArrow, {time.time() - t1:.1f} s"
        )
        _bld_paint()

    # ---- one serve per camera move: each layer that is on, in turn -----------------
    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        z = vsd["zoom"]
        on = HOLD["on"]
        want_hex = on.get("hex") and z >= HEX_ZOOM
        want_bld = on.get("bld") and z >= BLD_ZOOM
        if not want_hex:
            _hexes_off(f"hexagons from zoom {HEX_ZOOM:g}" if on.get("hex") else "")
        if not want_bld:
            _cover_off()
            HOLD["bld_status"] = f"footprints from zoom {BLD_ZOOM:g} (now {z:.1f})" if on.get("bld") else ""
        # the footprints and the hexagons are independent: served side by side,
        # so the hexagons no longer wait on the Overture read
        await asyncio.gather(
            _serve_bld(vsd, force) if want_bld else asyncio.sleep(0),
            _serve_hex(vsd, force) if want_hex else asyncio.sleep(0),
        )
        _legend()
        _status()

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

    def _spawn(coro):
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

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
            pair.unobserve(HOLD["h_cam"], names="view")
        except ValueError:
            pass
    pair.observe(_on_camera, names="view")
    HOLD["h_cam"] = _on_camera

    def _f(v, d=3):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"

    # ---- the click ------------------------------------------------------------------
    def _place_from_tiles(adm):
        out = []
        if adm.get("county"):
            out.append({"subtype": "county", "name": adm["county"]})
        if adm.get("region"):
            out.append({"subtype": "region", "name": adm["region"]})
        return out

    def _place_line(levels, pending):
        bits = []
        for lv in levels:
            nm = lv.get("name") or "?"
            if lv.get("name_en") and lv["name_en"] != nm:
                nm = f"{nm} {lv['name_en']}"
            lt = lv.get("local_type")
            tag = lv["subtype"] + (f", {lt}" if lt and lt != lv["subtype"] else "")
            bits.append(f"<b>{nm}</b> <span style='color:#777'>({tag})</span>")
        if pending:
            bits.append("<span style='opacity:.5'>the rest from Overture…</span>")
        if not bits:
            return ""
        return "<div style='font-size:13px;line-height:1.5'>" + " · ".join(bits) + "</div>"

    def _place_async(n, lon, lat, adm):
        async def _later():
            try:
                d = await asyncio.to_thread(division_at, lon, lat)
            except Exception as e:
                d = []
                _say(f"place (Overture divisions): {e}")
            if HOLD.get("pick_n") != n:
                return
            pair.panel = _place_line(d or _place_from_tiles(adm), pending=False) + HOLD["panel_body"]

        _spawn(_later())

    def _bld_lines(p):
        """The footprint on the clicked pixel, or the ground where there is
        none: a list of sentences, or None when the footprints are not up."""
        b = HOLD["bld"]
        if b is None or not HOLD["on"].get("bld"):
            return None
        lon, lat = p.get("lon"), p.get("lat")
        if lon is None or lat is None:
            return None
        px = b["psz"]
        c, r = int((lon - b["lon0"]) / px), int((b["lat0"] - lat) / px)
        arr, cover, rows = b["arr"], b["cover"], b["rows"]
        if not (0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]):
            return ["<span style='opacity:.7'>outside the footprints read for this view; pan and they follow</span>"]
        k, w = int(cover[r, c]), int(arr[r, c])
        if k > 0:
            row = rows.slice(k - 1, 1).to_pylist()[0]
            n_px, n_b, first = int(b["npx"][k - 1]), int(b["built"][k - 1]), int(b["first"][k - 1])
            what = ", ".join(x for x in (row.get("subtype"), row.get("class")) if x)
            extra = [x for x in (what, f"{row['height']:.0f} m" if row.get("height") is not None else None,
                                 f"{row['num_floors']} floors" if row.get("num_floors") is not None else None) if x]
            upd = (row.get("updated") or "")[:10]
            l1 = (f"Overture footprint <b>{row['id'][:8]}</b> from <b>{row.get('dataset') or 'an unnamed source'}</b>"
                  + (f", last touched {upd}" if upd else "") + (f" ({', '.join(extra)})" if extra else "") + ".")
            if n_b == 0:
                l2 = f"WSF has <b>never</b> read any of its {n_px} pixels as built-up."
            else:
                l2 = (f"WSF: <b>{n_b} of {n_px}</b> pixels under it are built-up, the first by <b>{WSF_DATE[first]}</b>"
                      + (f"; this pixel {WSF_DATE[w]}." if w else "; this pixel never."))
            out = [l1, l2]
            ay = b.get("ayear")
            if ay is not None:
                a = int(ay[k - 1])
                st = b["asteps"][:, k - 1]
                yrs = b["ayears"]
                stepsTxt = " · ".join(f"{ya}→{yb} {_f(float(v))}" for ya, yb, v in zip(yrs[:-1], yrs[1:], st))
                if a > 0:
                    l3 = f"AlphaEarth: the ground's fingerprint jumped in <b>{a}</b> (quiet level {_f(b.get('aD0'))})."
                elif a == -1:
                    l3 = "AlphaEarth: no single year stood out 2017 to 2025 (every step under the quiet level)."
                else:
                    l3 = "AlphaEarth: no embedding under this footprint."
                out.append(l3 + ("" if STRIP_MINIMAL else f"<br><span style='font-size:12px;color:#777'>{stepsTxt}</span>"))
            return out
        l1 = "<b>No Overture footprint here.</b>"
        l2 = (f"WSF: first read as built-up <b>{WSF_DATE[w]}</b>, a structure Overture has no building for." if w
              else "WSF has never read this pixel as built-up either.")
        return [l1, l2]

    def _hex_lines(p):
        fr = HOLD["frame"]
        cellh = p.get("cell")
        if fr is None or not cellh or not HOLD["on"].get("hex"):
            return None
        cell = int(cellh, 16)
        con.register("cur_cells", fr["cells"])
        r = con.execute("SELECT p_built, p_new, byear, byear_name, first_date, disp, disp_max, when_name FROM cur_cells WHERE cell = ?", [cell]).fetchone()
        if r is None:
            HOLD["hit"] = None
            return [f"<span style='opacity:.7'>{cellh}: not in the current frame</span>"]
        HOLD["hit"] = cell  # a repeat click keeps the pick (Stephen, 2026-09-24: "selecting the same hex again deselects")
        pb, pn, by, byn, fd, dsp, dmx, wn = r
        y0, y1 = fr["y0"], fr["y1"]
        if pb <= 0:
            l1 = "Hexagon, WSF: no built-up ground."
        elif by >= 0:
            l1 = f"Hexagon, WSF: <b>{100 * pb:.0f}%</b> built-up by the end of {y1}; <b>{100 * pn:.1f}%</b> of it built {y0 + 1} to {y1}, most of that in <b>{by}</b>."
        elif pn > 0:
            l1 = f"Hexagon, WSF: <b>{100 * pb:.0f}%</b> built-up by the end of {y1}; {100 * pn:.2f}% of it built {y0 + 1} to {y1} (under the {100 * NEW_MIN:g}% that counts as growth)."
        else:
            l1 = f"Hexagon, WSF: <b>{100 * pb:.0f}%</b> built-up, all of it before {y0 + 1} (first seen {fd})."
        if wn.startswith("no AlphaEarth") or dsp is None or np.isnan(dsp):
            l2 = "Hexagon, AlphaEarth: no embedding."
        else:
            s_lo, s_hi = fr.get("shift_lo", 0.0), fr.get("shift_hi", 1.0)
            t = (float(dsp) - s_lo) / max(s_hi - s_lo, 1e-9)
            how = "moved <b>significantly</b>" if t >= 0.75 else "moved <b>a fair amount</b>" if t >= 0.4 else "moved <b>a little</b>" if t >= 0.15 else "<b>barely moved</b>"
            l2 = f"Hexagon, AlphaEarth: the fingerprint {how} from {y0} to {y1} (shift {_f(dsp)})"
            l2 += ("; no single year stood out." if wn.startswith("no single year") else f", most sharply in <b>{wn.split('changed in ')[1].split(' ')[0]}</b>.")
        return [l1, l2 + f" <span style='color:#777'>({CELL_KM2.get(HOLD['res'], 0):.3f} km²)</span>"]

    def _on_pick(change):
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        try:
            lon, lat = p.get("lon"), p.get("lat")
            n, adm = p.get("n"), p.get("admin") or {}
            HOLD["pick_n"] = n
            lines = []
            for part in (_bld_lines(p), _hex_lines(p)):
                if part:
                    lines += part
            if not lines:
                HOLD["hit"] = None
                pair.panel = ""
                _paint()
                return
            where = f" at {lat:.5f}, {lon:.5f}" if lat is not None and lon is not None else ""
            l0 = _place_line(_place_from_tiles(adm), pending=True) if lon is not None else ""
            body = "<div style='font-size:14px;line-height:1.5'>" + "<br>".join(lines) + "</div>" + (
                "" if STRIP_MINIMAL else f"<div style='font-size:12px;color:#777'>{where}</div>")
            HOLD["panel_body"] = body
            pair.panel = l0 + body
            if l0:
                _place_async(n, lon, lat, adm)
        except Exception as e:
            pair.panel = f"<span style='opacity:.7'>click: {e}</span>"
        _paint()

    if HOLD.get("h_pick") is not None:
        try:
            pair.unobserve(HOLD["h_pick"], names="pick")
        except ValueError:
            pass
    pair.observe(_on_pick, names="pick")
    HOLD["h_pick"] = _on_pick

    # ---- the controls -----------------------------------------------------------------
    def _on_ctl_body(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        act = c.get("act")
        if act == "layers":
            new = {k: bool(v) for k, v in (c.get("on") or {}).items()}
            if new != HOLD["on"]:
                HOLD["on"] = new
                _cfg(layers=new)
                _request()
            return
        if act == "s2":
            y = int(c.get("s2y", HOLD["s2y"]))
            if y in S2_YEARS and y != HOLD["s2y"]:
                HOLD["s2y"] = y
                _cfg(s2_year=y)
            return
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
        if act == "fill":
            f = c.get("fill")
            if f in FILLS and f != HOLD["fill"]:
                HOLD["fill"] = f
                _cfg(fill=f)
                _paint()
            return
        if act == "bfill":
            f = c.get("bfill")
            if f in BFILLS and f != HOLD["bfill"]:
                HOLD["bfill"] = f
                _cfg(bfill=f)
                _bld_paint()
            return
        if act == "labels":
            HOLD["labels"] = bool(c.get("labels", True))
            _cfg(labels=HOLD["labels"])
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
            pair.unobserve(HOLD["h_ctl"], names="ctl")
        except ValueError:
            pass
    pair.observe(_on_ctl, names="ctl")
    HOLD["h_ctl"] = _on_ctl

    if HOLD["frame"] is None and HOLD["bld"] is None and not HOLD["busy"]:
        _request()
    else:
        _paint()
        _bld_paint()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Under the map

    Press the button once the map has settled to query the current view's
    hexagons with DuckDB. Columns, one row per hexagon:

    - `npx`: WSF pixels sampled in the hexagon.
    - `p_built`: share built-up by the end of the window; `p_new`: share that
      became built-up inside it; `grew`: whether `p_new` clears the growth
      threshold.
    - `byear` and `byear_name`: the year most of the new ground arrived
      (-1 built before the window, -3 nothing built).
    - `first_date`: the record's first built-up half-year in the hexagon.
    - `disp`: the AlphaEarth displacement between the ends of the window;
      `disp_max`: its largest single step, with one `step_YYYY` column per
      step; `moved`: whether a jump year was found inside the window.
    - `when` and `when_name`: the first year the embedding jumped
      (-1 never, -2 no embedding).

    The second table crosses the two year fills and counts how many hexagons
    WSF and AlphaEarth place in the same year.

    With the buildings on and the zoom past 13, the button also lists the
    Overture footprints under the view, one row each: `id`, `dataset` and
    `updated` (the source and the date it last touched the footprint),
    `subtype`, `class`, `height`, `num_floors`, `px` (WSF pixels the
    footprint claims), `built` (how many of them WSF reads as built-up) and
    `first` (the earliest WSF date among them, `never` when none).
    """)
    return


@app.cell
def _(mo):
    tables_btn = mo.ui.run_button(label="tables for the current view")
    tables_btn
    return (tables_btn,)


@app.cell
def _(HOLD, WSF_DATE, con, mo, np, pa, tables_btn):
    mo.stop(not tables_btn.value or HOLD["bld"] is None, None)
    _b = HOLD["bld"]
    _rows = _b["rows"]
    footprints = pa.table({
        "id": _rows["id"], "dataset": _rows["dataset"], "updated": _rows["updated"],
        "subtype": _rows["subtype"], "class": _rows["class"], "height": _rows["height"], "num_floors": _rows["num_floors"],
        "px": pa.array(np.asarray(_b["npx"]).astype(np.int64)), "built": pa.array(np.asarray(_b["built"]).astype(np.int64)),
        "first": pa.array([WSF_DATE[int(k)] for k in _b["first"]]),
    })
    con.register("view_footprints", footprints)
    footprints_table = mo.sql(
        """
        SELECT * FROM view_footprints ORDER BY built DESC, px DESC
        """,
        engine=con,
    )
    return


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None, mo.md("*no hexagons folded (switch the layer on past zoom 9)*") if HOLD["bld"] is None else None)
    con.register("view_cells", HOLD["frame"]["cells"])
    year_by_year = mo.sql(
        """
        PIVOT (SELECT byear_name AS wsf, when_name AS aef FROM view_cells WHERE grew OR moved)
        ON aef USING count(*) GROUP BY wsf ORDER BY wsf
        """,
        engine=con,
    )
    return


if __name__ == "__main__":
    app.run()
