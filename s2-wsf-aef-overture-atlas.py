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


"""Buildings on the map, checked against the ground: the slider, made easier.

Stephen, 2026-09-24: "go back to what was working ... and help the user
find what they need", "we're looking for a sleeker design", "include
overture and include S2". The slider notebook's data and reads, unchanged;
a new face on them:

- Sentinel-2 yearly imagery is the picture, full bleed, under a dark
  label-only basemap (its own buildings hidden).
- The Overture footprints sit on it, lit by the year they were first seen
  built (WSF, or AlphaEarth if you switch the witness): the timeline's
  year bright amber, the years before it dimmer, what stood in 2016 a white
  outline, what came later not drawn yet. Zoomed out, WSF's own built-up
  ground in the same colours. The timeline scrubs and plays in the
  browser (the footprints carry their years; the WSF tiles carry the index
  itself), so a year change never waits on the kernel; the imagery follows
  to the nearest S2 year.
- Three lenses: growth; map check (footprints WSF never read as built-up,
  and built-up ground 20 m or more from any footprint); sources.
- A card for the click, in sentences, with "show that year"; a before and
  after compare (swipe or side by side); the AlphaEarth hexagons, place
  names, imagery brightness and an about page behind the "..." menu.

Run: uv run marimo run s2-wsf-aef-overture-atlas.py (it fills the window;
X or Esc gives the notebook back)

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
    # Buildings on the map, checked against the ground

    Sentinel-2 imagery, the Overture building footprints over it, and two
    witnesses for when each was built: the WSF tracker and AlphaEarth. Press
    play on the timeline to watch a place fill in; click anything for its
    account. `X` fills the window.

    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/s2-wsf-aef-overture-pair/blob/main/s2-wsf-aef-overture-atlas.py)
    <small>molab runs in the same region as the data. Locally: `uv run marimo run s2-wsf-aef-overture-atlas.py`</small>
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
    AEF_FROM0, AEF_TO0 = 2017, 2025
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

    VIEW_W, VIEW_H = 700, 780  # one pane
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
    HOME = {"longitude": 3.62, "latitude": 6.46, "zoom": 11.2}  # Lagos, the Lekki corridor

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
    # the raw tiles for the growth lens: the index k itself in red, opaque
    # where built-up, so the browser colours "built by year Y" for any Y
    # without asking again (the timeline scrubs and plays client-side)
    _rawmap = np.zeros((256, 4), np.uint8)
    for _k in range(1, WSF_NIDX + 1):
        _rawmap[_k] = (_k, 0, 0, 255)

    def _wsf_png(got, k, n, y, lon0, lon1, raw=False):
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
        rgba = (_rawmap if raw else _cmap)[pxv.astype(np.uint8)]
        if not rgba[..., 3].any():
            return None
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    async def wsf_tile_png(z, x, y, raw=False):
        """RGBA PNG bytes for Web Mercator tile (z, x, y) of the pyramid (the
        earliest built-up date under each pixel, the year's color), or None
        where nothing is built. The level is the one whose pixel is nearest the
        tile's own (in metres at the tile's latitude)."""
        key = (z, x, y, raw)
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
        _png_cache[key] = await cpu(_wsf_png, got, k, n, y, lon0, lon1, raw)
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
    class Atlas(anywidget.AnyWidget):
        """The map: a plain basemap, the Overture footprints lit by the year
        they were first seen built, the AlphaEarth hexagons by the year the
        ground changed, a timeline, the Sentinel-2 imagery as a layer you add,
        and a card for whatever is clicked. When the clicked thing changed
        inside the imagery's years, the imagery flips between the year before
        and the year after on its own.

        Kernel -> browser: `polys` (the footprints, Arrow IPC, native GeoArrow
        polygons) with `battrs` (4 bytes per footprint: WSF index 0..20,
        AlphaEarth year code, source index, last-touched year) and `bmeta`
        (JSON: source names, box, counts); `gap` (PNG: built-up ground 20 m
        or more from any footprint, over `bmeta.box`); `cells` with `hattrs`
        (4 bytes per hexagon: AlphaEarth change year code, how much it
        changed 1..255, WSF new share, WSF built share) and `hmeta`; `card`
        (JSON); `status`; `config`. Browser -> kernel: `view`, `pick`, `ctl`.
        Tiles are custom messages: `s2` (a year's mosaic) and `wsfidx` (the
        WSF index itself, coloured in the browser for the timeline's year)."""

        cells = traitlets.Bytes(b"").tag(sync=True)
        colors = traitlets.Bytes(b"").tag(sync=True)
        hattrs = traitlets.Bytes(b"").tag(sync=True)
        hmeta = traitlets.Unicode("{}").tag(sync=True)
        polys = traitlets.Bytes(b"").tag(sync=True)
        battrs = traitlets.Bytes(b"").tag(sync=True)
        bmeta = traitlets.Unicode("{}").tag(sync=True)
        gap = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        card = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("[]").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)
        ctl = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None  # async (src, z, x, y, year) -> PNG bytes or None
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

        _css = r"""
        .at{--glass:rgba(255,255,255,.9);--glass-hi:#fff;--line:rgba(24,32,40,.13);--text:#1b2127;--muted:#5d6873;--amber:#e69f00;--deep:#b85c00;--cool:#0072b2;--sel:rgba(24,32,40,.08);
          position:relative;width:100%;background:#eef0f1;color:var(--text);font:14px/1.45 "Instrument Sans",ui-sans-serif,system-ui,sans-serif;font-variant-numeric:tabular-nums;overflow:hidden;border-radius:10px;-webkit-font-smoothing:antialiased}
        .at.fit{position:fixed;inset:0;z-index:9999;border-radius:0}
        .at *{box-sizing:border-box}
        .at-pane{position:relative;width:100%}
        .at-map{position:absolute;inset:0}
        .at-glass{background:var(--glass);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:12px;box-shadow:0 6px 22px rgba(20,30,40,.14)}
        .at button{font:inherit;color:inherit}
        .at button:focus-visible,.at input:focus-visible{outline:2px solid var(--cool);outline-offset:2px}
        .at-top{position:absolute;left:12px;top:12px;z-index:6;display:flex;flex-direction:column;gap:8px;align-items:flex-start;max-width:calc(100% - 150px)}
        .at-search{position:relative;display:flex;align-items:center;gap:8px;padding:0 12px;height:40px;width:270px}
        .at-search svg{flex:0 0 auto;opacity:.6}
        .at-search input{flex:1;min-width:0;background:none;border:0;color:var(--text);font:inherit;outline:none}
        .at-search input::placeholder{color:var(--muted)}
        .at-hits{position:absolute;left:-1px;right:-1px;top:46px;display:none;padding:4px;background:var(--glass-hi)}
        .at-hit{padding:7px 10px;border-radius:8px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .at-hit small{display:block;color:var(--muted);font-size:12px}
        .at-hit.sel{background:var(--sel)}
        .at-panel{display:flex;flex-direction:column;gap:6px;padding:7px 9px}
        .at-row{display:flex;align-items:center;gap:8px}
        .at-lab{font-size:12.5px;color:var(--muted);min-width:74px}
        .at-seg{display:flex;padding:3px;height:40px;gap:2px}
        .at-seg button{border:0;background:none;color:var(--muted);padding:0 12px;border-radius:9px;cursor:pointer;white-space:nowrap}
        .at-seg button:hover{color:var(--text)}
        .at-seg button.on{background:var(--text);color:#fff}
        .at-tools{position:absolute;right:12px;top:12px;z-index:7;display:flex;gap:8px}
        .at-btn{height:40px;min-width:40px;padding:0 13px;display:inline-flex;align-items:center;justify-content:center;gap:7px;cursor:pointer;white-space:nowrap}
        .at-btn:hover{border-color:rgba(24,32,40,.3)}
        .at-btn.on{background:var(--text);color:#fff;border-color:var(--text)}
        .at-bar{position:absolute;left:0;right:0;top:0;height:3px;z-index:9;overflow:hidden;pointer-events:none;opacity:0;transition:opacity .3s}
        .at-bar.busy{opacity:1}
        .at-bar i{position:absolute;top:0;height:3px;width:28%;background:linear-gradient(90deg,transparent,var(--amber),transparent);animation:at-run 1.2s ease-in-out infinite}
        @keyframes at-run{0%{left:-28%}100%{left:100%}}
        .at-msg{position:absolute;left:50%;transform:translateX(-50%);top:12px;z-index:5;font-size:13px;color:var(--muted);padding:6px 11px;display:none;max-width:min(520px,calc(100% - 24px))}
        .at-msg.err{color:#8a4b00}
        .at-card{position:absolute;right:12px;top:60px;z-index:6;width:340px;max-width:calc(100% - 24px);max-height:calc(100% - 250px);overflow:auto;padding:14px 16px 12px;display:none}
        .at-card .place{color:var(--muted);font-size:12.5px;margin-bottom:4px;padding-right:24px}
        .at-card h3{margin:0 0 10px;font-size:17px;font-weight:600;letter-spacing:-.005em}
        .at-card .row{display:grid;grid-template-columns:18px 1fr;gap:8px;margin:0 0 8px}
        .at-card .row i{width:10px;height:10px;border-radius:3px;margin-top:5px;display:block}
        .at-card .muted{color:var(--muted)}
        .at-card .sub{color:var(--muted);font-size:12.5px;margin:-4px 0 8px 26px}
        .at-card .x{position:absolute;right:8px;top:8px;border:0;background:none;color:var(--muted);cursor:pointer;width:28px;height:28px;border-radius:8px;font-size:17px;line-height:1}
        .at-card .x:hover{background:var(--sel);color:var(--text)}
        .at-card .flip{border-top:1px solid var(--line);margin-top:10px;padding-top:10px}
        .at-card .flip .yrs{display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap}
        .at-chip{border:1px solid var(--line);background:#fff;border-radius:999px;padding:4px 12px;cursor:pointer}
        .at-chip:hover{border-color:rgba(24,32,40,.35)}
        .at-chip.on{background:var(--text);color:#fff;border-color:var(--text)}
        .at-spark{margin:2px 0 8px 26px}
        .at-spark text{font-size:10px;fill:var(--muted)}
        .at-dock{position:absolute;left:50%;transform:translateX(-50%);bottom:16px;z-index:6;width:min(820px,calc(100% - 24px));padding:12px 16px 12px}
        .at-grow{display:grid;grid-template-columns:40px 1fr 200px;gap:14px;align-items:end}
        .at-play{width:40px;height:40px;border-radius:50%;border:1px solid var(--line);background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;align-self:center}
        .at-play:hover{border-color:rgba(24,32,40,.35)}
        .at-cols{display:flex;gap:4px;height:74px;align-items:stretch;cursor:pointer;user-select:none;touch-action:none}
        .at-col{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:stretch;gap:5px;min-width:0;position:relative}
        .at-col .b{border-radius:4px 4px 0 0;min-height:2px;background:rgba(24,32,40,.13)}
        .at-col .y{font-size:11.5px;color:var(--muted);text-align:center;line-height:1}
        .at-col.past .b{background:rgba(230,159,0,.55)}
        .at-col.cur .b{background:var(--deep)}
        .at-col.cur .y{color:var(--text);font-weight:600}
        .at-col.off{opacity:.35;cursor:default}
        .at-mid{display:flex;flex-direction:column;gap:2px;min-width:0}
        .at-slider{width:calc(100% - 100% / 10);margin:0 calc(100% / 20);accent-color:var(--deep);cursor:pointer;height:18px}
        .at-yr{display:flex;flex-direction:column;align-items:flex-start;justify-content:flex-end;min-width:0}
        .at-yr b{font-size:32px;line-height:1;font-weight:600;letter-spacing:-.02em}
        .at-yr span{font-size:12.5px;color:var(--muted);margin-top:5px;line-height:1.35}
        .at-foot{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-top:10px;flex-wrap:wrap;font-size:12.5px;color:var(--muted)}
        .seg-s{display:flex;gap:2px;padding:2px;border:1px solid var(--line);border-radius:9px;width:max-content}
        .seg-s button{border:0;background:none;color:var(--muted);padding:3px 10px;border-radius:7px;cursor:pointer}
        .seg-s button.on{background:var(--text);color:#fff}
        .at-keys{display:flex;gap:14px;flex-wrap:wrap}
        .at-key{display:inline-flex;align-items:center;gap:6px}
        .at-key i{display:inline-block;width:12px;height:12px;border-radius:3px}
        .at-tip{position:absolute;z-index:10;pointer-events:none;background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12.5px;white-space:nowrap;display:none;transform:translate(-50%,-100%);box-shadow:0 4px 14px rgba(20,30,40,.12)}
        .at-sum{display:grid;gap:8px}
        .at-sum h4{margin:0;font-size:14px;font-weight:600}
        .at-sum .line{display:grid;grid-template-columns:14px 1fr;gap:9px;align-items:start}
        .at-sum .line i{width:12px;height:12px;border-radius:3px;margin-top:4px}
        .at-sum b{font-weight:600}
        .at-stack{display:flex;gap:2px;height:12px;border-radius:4px;overflow:hidden}
        .at-stack span{display:block;height:100%}
        .at-ramp{height:12px;border-radius:4px;width:220px}
        .at-more{position:absolute;right:12px;top:60px;z-index:8;width:300px;padding:8px;display:none}
        .at-more .item{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:9px 10px;border-radius:9px}
        .at-more .item:hover{background:var(--sel)}
        .at-more .item small{display:block;color:var(--muted);font-size:12px}
        .at-more hr{border:0;border-top:1px solid var(--line);margin:4px 6px}
        .at-sw{position:relative;width:34px;height:20px;flex:0 0 auto;border-radius:999px;background:rgba(24,32,40,.2);border:0;cursor:pointer;transition:background .2s}
        .at-sw::after{content:"";position:absolute;left:3px;top:3px;width:14px;height:14px;border-radius:50%;background:#fff;transition:left .2s}
        .at-sw.on{background:var(--cool)}
        .at-sw.on::after{left:17px}
        .at-more input[type=range]{width:120px;accent-color:var(--cool)}
        .at-imgtag{position:absolute;left:50%;transform:translateX(-50%);top:56px;z-index:6;padding:6px 14px;font-weight:600;display:none;white-space:nowrap}
        .at-about{position:absolute;inset:0;z-index:20;display:none;align-items:center;justify-content:center;background:rgba(20,30,40,.35)}
        .at-about .box{width:min(620px,calc(100% - 32px));max-height:calc(100% - 64px);overflow:auto;padding:22px 26px;line-height:1.55;background:#fff}
        .at-about h2{margin:0 0 10px;font-size:22px;font-weight:600;letter-spacing:-.01em}
        .at-about p{margin:0 0 10px;max-width:66ch}
        .at-about small{color:var(--muted)}
        .at .maplibregl-ctrl-group{border:1px solid var(--line);box-shadow:0 6px 22px rgba(20,30,40,.14);border-radius:10px}
        .at .maplibregl-ctrl-bottom-right{bottom:0}
        @media (max-width:760px){.at-top{max-width:calc(100% - 24px)}.at-search{width:calc(100vw - 48px)}.at-grow{grid-template-columns:36px 1fr}.at-yr{grid-column:1/-1;flex-direction:row;align-items:baseline;gap:10px}.at-yr span{min-width:0}.at-col .y{font-size:9.5px}.at-card{top:auto;bottom:270px;max-height:38%}.at-tools{top:108px;right:12px}.at-btn span{display:none}}
        @media (prefers-reduced-motion:reduce){.at-bar i{animation:none;left:0;width:100%}}
        """

        _esm = r"""
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {TileLayer, H3HexagonLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import * as arrow from "https://esm.sh/apache-arrow@18.1.0";
        import {GeoArrowPolygonLayer} from "https://esm.sh/@geoarrow/deck.gl-layers@0.3.2?deps=@deck.gl/core@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/geo-layers@9.3.10,@deck.gl/aggregation-layers@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0,@luma.gl/core@9.3.6,@luma.gl/engine@9.3.6,@luma.gl/webgl@9.3.6,@luma.gl/shadertools@9.3.6,@luma.gl/gltf@9.3.6";
        import {latLngToCell, getResolution, cellToBoundary} from "https://esm.sh/h3-js@4.5.0";
        import {Protocol as PMProtocol} from "https://esm.sh/pmtiles@4.5.0";
        maplibregl.addProtocol("pmtiles", new PMProtocol().tile);

        // a regular basemap; the imagery is a layer you add (Stephen,
        // 2026-09-24: "we can have a base map, like a regular base map, and
        // then add the imagery if we want it")
        const STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";
        const FONTS = "https://fonts.googleapis.com/css2?family=Instrument+Sans:wdth,wght@75..100,400..700&display=swap";

        // the data colours: amber for built or changed (the timeline's year
        // deep amber, the years before it lighter: one hue, lightness carries
        // the year), grey outlines for what already stood in 2016, blue for
        // "the map and WSF disagree". Nothing hangs on red vs green.
        const DEEP = [184, 92, 0], AMBER = [230, 159, 0], GREY = [96, 106, 116], WHITE = [255, 255, 255], COOL = [0, 114, 178];
        const SRC = [[0, 114, 178], [230, 159, 0], [204, 121, 167], [86, 180, 233]], OTHER = [150, 156, 162];
        const MONTH = ["", "Jan", "Jul"];
        const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
        const fmt = (n) => Number(n).toLocaleString("en-US");

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
          play: '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l13-7.5z"/></svg>',
          pause: '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4.5" width="4" height="15" rx="1"/><rect x="14" y="4.5" width="4" height="15" rx="1"/></svg>',
          image: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="m3 16 5-5 4 4 3-3 6 6"/></svg>',
          more: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg>',
          expand: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
          shrink: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>',
        };

        // WSF index k (1..20): the half-year it books. 1 = built by July
        // 2016; even k = July of its year to January of the next; odd k = January to July
        const wsfYear = (k) => 2016 + Math.floor((k - 1) / 2);
        function wsfWindow(k) {
          if (k <= 0) return null;
          if (k === 1) return "by July 2016";
          const y = wsfYear(k);
          return k % 2 === 0 ? `between July ${y} and January ${y + 1}` : `between January ${y} and July ${y}`;
        }
        const wsfYearWindow = (y) => (y === 2016 ? "July 2016 to January 2017" : `January ${y} to January ${y + 1}`);

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
          // AlphaEarth in viridis, as the slider had it (Stephen, 2026-09-24:
          // "go back to Viridis"): opaque fills, quiet cells faint grey
          const VIR = (cfg.viridis || "440154fde725").match(/.{6}/g).map((h) => [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)));
          const vir = (t) => { t = Math.max(0, Math.min(1, t)) * (VIR.length - 1); const i = Math.min(VIR.length - 2, Math.floor(t)), f = t - i; return VIR[i].map((v, j) => Math.round(v + (VIR[i + 1][j] - v) * f)); };
          const virYear = (y) => vir((y - 2018) / 7);
          const A_FILL = cfg.alpha_fill || 235, A_QUIET = cfg.alpha_quiet || 70, QUIET = [200, 200, 200];
          const virCss = (n) => Array.from({length: n}, (_, i) => `rgb(${vir(i / (n - 1)).join(",")})`).join(",");
          const YMIN = 2016, YMAX = 2025, BLDZ = cfg.bld_zoom || 13, HEXZ = cfg.hex_zoom || 9;
          // the layers, on or off, as in the slider (B W A S); the buildings'
          // colour (Q steps through it); the AlphaEarth fill; the timeline's year
          const st = {
            on: Object.assign({bld: true, wsf: false, hex: false, img: false}, cfg.layers0 || {}),
            bfill: "wyear", Y: YMAX, playing: false, gmode: "when", y0: 2017, y1: 2025,
            imgYear: S2Y[S2Y.length - 1], labels: true, s2scale: Number(cfg.s2_scale) || 1,
            fit: !!cfg.fit, picked: -1, flip: null,
          };

          // ---- the frame ----------------------------------------------------
          const root = el_("div", "at");
          const pane = el_("div", "at-pane");
          const mapEl = el_("div", "at-map");
          const bar = el_("div", "at-bar", "<i></i>");
          const msg = el_("div", "at-msg at-glass");
          const imgTag = el_("div", "at-imgtag at-glass");
          pane.append(mapEl, bar, msg, imgTag);
          root.appendChild(pane);
          el.appendChild(root);

          const top = el_("div", "at-top");
          const search = el_("div", "at-search at-glass", ICON.search);
          const gc = el_("input"); gc.type = "search"; gc.placeholder = "Search a place"; gc.autocomplete = "off"; gc.spellcheck = false;
          const hits = el_("div", "at-hits at-glass");
          search.append(gc, hits);
          const BFILLS = [["wyear", "WSF year", "lit by when WSF first saw them built"], ["ayear", "AlphaEarth year", "lit by the first year their ground's AlphaEarth numbers jumped (reads nine years for the area, the first time takes a moment)"],
                          ["check", "Map check", "blue: on the map but never built-up to WSF; amber: built-up ground 20 m or more from any building on the map"], ["source", "Source", "the dataset each building came from"]];
          const LAYERS = [["bld", "Buildings", "Overture building footprints, from zoom " + BLDZ + " (B)"], ["wsf", "WSF", "WSF built-up ground, lit by when it was first seen built (W)"],
                          ["hex", "AlphaEarth", "AlphaEarth hexagons: the year the ground changed, or how much, from zoom " + HEXZ + " (A)"], ["img", "Imagery", "Sentinel-2 yearly imagery (S or I)"]];
          const panel = el_("div", "at-panel at-glass");
          const rowOf = (label) => { const r = el_("div", "at-row"); r.appendChild(el_("span", "at-lab", label)); panel.appendChild(r); return r; };
          const segOf = (row, items, isOn, onClick) => {
            const seg = el_("div", "seg-s");
            const bs = items.map(([k, label, title]) => { const b = el_("button", "", label); b.title = title || ""; b.onclick = () => onClick(k); seg.appendChild(b); return b; });
            row.appendChild(seg);
            return () => items.forEach(([k], i) => bs[i].classList.toggle("on", isOn(k)));
          };
          const rLayers = rowOf("Layers");
          const styleLayers = segOf(rLayers, LAYERS, (k) => !!st.on[k], (k) => toggleLayer(k));
          const rBld = rowOf("Buildings");
          const styleBfill = segOf(rBld, BFILLS, (k) => k === st.bfill, (k) => setBfill(k));
          const rHex = rowOf("AlphaEarth");
          const styleHex = segOf(rHex, [["when", "Year it changed", "the year the ground first changed clearly"], ["much", "How much", "how much the ground changed over the years"]], (k) => k === st.gmode, (k) => { st.gmode = k; recolorHex(); styleRows(); renderDock(); update(); });
          const rImg = rowOf("Imagery");
          const styleImgRow = segOf(rImg, S2Y.map((y) => [y, String(y), "the Sentinel-2 mosaic of " + y + " ([ ] step, F first or last)"]), (k) => k === st.imgYear, (k) => { stopFlip(true); st.imgYear = k; styleRows(); update(); });
          function styleRows() {
            styleLayers(); styleBfill(); styleHex(); styleImgRow();
            rBld.style.display = st.on.bld ? "" : "none";
            rHex.style.display = st.on.hex ? "" : "none";
            rImg.style.display = st.on.img ? "" : "none";
          }
          top.append(search, panel);
          pane.appendChild(top);

          const tools = el_("div", "at-tools");
          const bMore = el_("button", "at-btn at-glass", ICON.more); bMore.title = "settings and about";
          const bFit = el_("button", "at-btn at-glass", ICON.expand); bFit.title = "fill the window (X)";
          tools.append(bMore, bFit);
          pane.appendChild(tools);

          // the more menu
          const more = el_("div", "at-more at-glass");
          const item = (title, sub, ctl) => { const r = el_("div", "item"); r.append(el_("div", "", `${title}${sub ? `<small>${sub}</small>` : ""}`), ctl); more.appendChild(r); return r; };
          const sw = (get, set) => { const b = el_("button", "at-sw"); b.setAttribute("role", "switch"); const sty = () => { b.classList.toggle("on", !!get()); b.setAttribute("aria-checked", String(!!get())); }; b.onclick = () => { set(!get()); sty(); }; sty(); b.sty = sty; return b; };
          const gam = el_("input"); gam.type = "range"; gam.min = 0.3; gam.max = 2.5; gam.step = 0.1; gam.value = st.s2scale;
          gam.title = "imagery brightness (gamma); double-click for 1.0";
          let gamT = null;
          gam.oninput = () => { st.s2scale = Number(gam.value); clearTimeout(gamT); gamT = setTimeout(() => send("s2scale"), 250); };
          gam.ondblclick = () => { gam.value = 1; gam.oninput(); };
          item("Imagery brightness", "; and ' step it", gam);
          const swLab = sw(() => st.labels, (v) => { st.labels = v; labels(v); send("labels"); });
          item("Place names", "", swLab);
          more.appendChild(el_("hr"));
          const bAbout = el_("button", "at-chip", "About this map"); bAbout.style.margin = "4px 10px 6px";
          more.appendChild(bAbout);
          pane.appendChild(more);

          const card = el_("div", "at-card at-glass");
          pane.appendChild(card);
          const dock = el_("div", "at-dock at-glass");
          pane.appendChild(dock);
          const tip = el_("div", "at-tip");
          pane.appendChild(tip);

          const about = el_("div", "at-about");
          about.innerHTML = `<div class="box at-glass">
            <h2>Buildings on the map, checked against the ground</h2>
            <p>Turn the layers on and off in the panel at the top left. <b>Buildings</b>: the footprints from Overture Maps, each lit by when the World Settlement Footprint tracker (WSF) first read the ground under it as built-up. WSF looks twice a year, July 2016 to January 2026, at 10 m. Deep amber is the year on the timeline, lighter amber the years before it, a grey outline was already standing in 2016. Zoomed out, the same colours paint WSF's built-up ground. You can switch the date to AlphaEarth's instead.</p>
            <p><b>AlphaEarth</b>: describes every 10 m of ground with 64 numbers a year, 2017 to 2025. When those numbers jump from one year to the next by more than they do on ground WSF says stayed the same (the quiet level), the ground changed that year: building, clearing, water, fields. Hexagons show the year it changed, or how much.</p>
            <p>The buildings can also be coloured for a <b>map check</b>: buildings on the map WSF never read as built-up (blue), and built-up ground 20 m or more from any building on the map (amber); or by <b>source</b>, the dataset each came from.</p>
            <p><b>Imagery</b> adds the Sentinel-2 yearly mosaic (2022 to 2025). Click something that changed in 2023 to 2025 and the imagery flips between the year before and the year after on its own.</p>
            <p><small>Keys: B W A S the layers; Q the buildings' colour; arrows the timeline year, space play; [ ] and F the imagery year, ; and ' its brightness; 1 to 9 the choices in the last row shown; - = and _ + the AlphaEarth years; L place names; X fill the window; / search; Esc close.</small></p>
            <p><small>WSF Tracker (c) DLR and MindEarth. AlphaEarth Foundations by Google and Google DeepMind (CC BY 4.0). Sentinel-2 mosaics by Earth Genome (CC BY 4.0). Overture Maps buildings and divisions (ODbL). Photon over OpenStreetMap (ODbL). Basemap by Carto. Overture release ${cfg.ov_release || ""}.</small></p>
            <div style="margin-top:12px"><button class="at-chip">Close</button></div></div>`;
          pane.appendChild(about);
          about.querySelector("button").onclick = () => { about.style.display = "none"; };
          about.onclick = (e) => { if (e.target === about) about.style.display = "none"; };
          bAbout.onclick = () => { more.style.display = "none"; about.style.display = "flex"; };

          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({act, on: {bld: st.on.bld, wsf: st.on.wsf, hex: st.on.hex, s2: st.on.img}, s2scale: st.s2scale,
              labels: st.labels, witness: st.bfill === "ayear" ? "aef" : "wsf", y0: st.y0, y1: st.y1, n: Date.now()}, extra || {})));
            model.save_changes();
          };

          // ---- status ------------------------------------------------------------
          const ERR = /failed|error|no match|search:|timed? ?out|^(deck|map|footprints|load|boot|\w+ tile):/i;
          let msgT = null;
          const note = (t, ms) => { msg.textContent = t; msg.style.display = t ? "block" : "none"; msg.classList.toggle("err", ERR.test(t)); clearTimeout(msgT); if (ms) msgT = setTimeout(() => { msg.style.display = "none"; }, ms); };
          const say = (t) => {
            t = (t || "").replace(/​/g, "");
            if (ERR.test(t)) { note(t); bar.classList.remove("busy"); return; }
            const many = /more than [\d,]+ footprints/.test(t);
            if (many) note("Too many buildings here to draw at once: zoom in a little.");
            const busy = t.split(" · ").filter((p) => p.includes("…"));
            const names = [];
            for (const p of busy) {
              if (/footprint|Overture/i.test(p) && !names.includes("buildings")) names.push("buildings");
              if (/AlphaEarth|AEF|hexagon/i.test(p) && !names.includes("AlphaEarth")) names.push("AlphaEarth");
            }
            bar.classList.toggle("busy", names.length > 0);
            if (names.length) note("Loading " + names.join(" and ") + "…");
            else if (!many) note("");
          };

          // ---- tiles ----------------------------------------------------------------
          const pending = new Map();
          let tseq = 0;
          const tstat = {asked: 0, got: 0, empty: 0, err: 0, abort: 0};
          model.on("msg:custom", (m, buffers) => {
            if (!m || m.kind !== "tile") return;
            const p = pending.get(m.id);
            if (!p) return;
            pending.delete(m.id);
            if (m.err) { tstat.err++; p.reject(new Error(m.err)); return; }
            if (m.empty || !buffers || !buffers.length) { tstat.empty++; p.resolve(null); return; }
            tstat.got++;
            p.resolve(bytesOf(buffers[0]));
          });
          const ask = (src, year, index, signal) => new Promise((resolve, reject) => {
            const id = ++tseq; tstat.asked++;
            pending.set(id, {resolve, reject});
            model.send({kind: "tile", id, src, year, x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => { pending.delete(id); tstat.abort++; const e = new Error("aborted"); e.name = "AbortError"; reject(e); });
          });
          const pngBitmap = (u8) => createImageBitmap(new Blob([u8], {type: "image/png"}), {premultiplyAlpha: "none", colorSpaceConversion: "none"});

          const rawTiles = new Map();
          async function rawTile({index, signal, bbox}) {
            const u8 = await ask("wsfidx", 0, index, signal);
            if (!u8) return null;
            const bm = await pngBitmap(u8);
            const c = new OffscreenCanvas(bm.width, bm.height);
            const g = c.getContext("2d", {willReadFrequently: true});
            g.drawImage(bm, 0, 0);
            const d = g.getImageData(0, 0, bm.width, bm.height).data;
            const k = new Uint8Array(bm.width * bm.height);
            for (let i = 0, j = 0; i < k.length; i++, j += 4) k[i] = d[j + 3] ? d[j] : 0;
            const t = {k, w: bm.width, h: bm.height, z: index.z, bbox, painted: null, key: null};
            rawTiles.set(`${index.z}/${index.x}/${index.y}`, t);
            if (rawTiles.size > 600) rawTiles.delete(rawTiles.keys().next().value);
            countsSoon();
            return t;
          }
          // the year ramp shared by every layer: deep amber for the year
          // shown, lighter for the years before, nothing for later years
          const rampA = (y, Y) => { const t = (y - 2016) / Math.max(1, Y - 1 - 2016); return 70 + 100 * t; };
          function growthLUT(Y) {
            const lut = new Uint8ClampedArray(256 * 4);
            for (let k = 1; k <= 20; k++) {
              const yr = wsfYear(k), o = 4 * k;
              let c = null, a = 0;
              if (k === 1) { c = GREY; a = 55; }
              else if (yr > Y) a = 0;
              else if (yr === Y) { c = DEEP; a = 235; }
              else { c = AMBER; a = rampA(yr, Y); }
              if (c) { lut[o] = c[0]; lut[o + 1] = c[1]; lut[o + 2] = c[2]; lut[o + 3] = a; }
            }
            return lut;
          }
          function paintRaw(t, Y) {
            if (t.key === Y && t.painted) return t.painted;
            const lut = growthLUT(Y);
            const img = new ImageData(t.w, t.h), o = img.data;
            for (let i = 0; i < t.k.length; i++) { const k = t.k[i]; if (!k) continue; const b = 4 * k, j = 4 * i; o[j] = lut[b]; o[j + 1] = lut[b + 1]; o[j + 2] = lut[b + 2]; o[j + 3] = lut[b + 3]; }
            const c = document.createElement("canvas"); c.width = t.w; c.height = t.h;
            c.getContext("2d").putImageData(img, 0, 0);
            t.painted = c; t.key = Y;
            return c;
          }

          // ---- the footprints ---------------------------------------------------------
          let polys = null, polysSeq = 0, attrs = null, meta = {}, fillVec = null, lineVec = null, colSeq = 0;
          const RGBA = new arrow.FixedSizeList(4, new arrow.Field("rgba", new arrow.Uint8(), false));
          const vecOf = (u8, n) => arrow.makeVector(arrow.makeData({type: RGBA, length: n, nullCount: 0,
            child: arrow.makeData({type: new arrow.Uint8(), length: 4 * n, nullCount: 0, data: u8})}));
          // per footprint: [0] WSF index 0..20, [1] AlphaEarth (0 not read, 1 no
          // jump, 2 no data, else year - 2000), [2] source index, [3] last touched
          function yearOf(i) {
            if (st.bfill === "ayear") { const c = attrs[4 * i + 1]; return c >= 16 ? 2000 + c : (c === 1 ? -1 : 0); }
            const k = attrs[4 * i];
            return k === 0 ? 0 : (k === 1 ? -1 : wsfYear(k));
          }
          function recolor() {
            const n = polys ? polys.numRows : 0;
            if (!n || !attrs || attrs.length !== 4 * n) { fillVec = lineVec = null; return; }
            const F = new Uint8Array(4 * n), L = new Uint8Array(4 * n);
            const put = (A, i, c, a) => { A[4 * i] = c[0]; A[4 * i + 1] = c[1]; A[4 * i + 2] = c[2]; A[4 * i + 3] = a; };
            const Y = st.Y, img = st.on.img, base = img ? WHITE : GREY;
            // over imagery the footprints are outlines only: colour on colour reads as neither
            const fillK = img ? 0 : 1;
            for (let i = 0; i < n; i++) {
              if (st.bfill === "wyear" || st.bfill === "ayear") {
                const y = yearOf(i);
                if (y === -1) put(L, i, base, img ? 120 : 110);
                else if (y === 0) put(L, i, base, img ? 70 : 60);
                else if (y > Y) { /* later: not drawn */ }
                else if (y === Y) { put(F, i, DEEP, 225 * fillK); put(L, i, img ? [255, 190, 90] : DEEP, 255); }
                else { const a = rampA(y, Y); put(F, i, AMBER, a * fillK); put(L, i, AMBER, img ? 230 : Math.min(255, a + 70)); }
              } else if (st.bfill === "check") {
                if (attrs[4 * i] === 0) { put(F, i, COOL, 110 * fillK); put(L, i, img ? [120, 200, 255] : COOL, 255); }
                else put(L, i, base, img ? 90 : 80);
              } else {
                const s = attrs[4 * i + 2];
                const c = s < SRC.length ? SRC[s] : OTHER;
                put(F, i, c, 120 * fillK); put(L, i, c, 255);
              }
            }
            if (st.picked >= 0 && st.picked < n) { put(F, st.picked, COOL, 70 * fillK); put(L, st.picked, img ? [120, 200, 255] : COOL, 255); }
            fillVec = vecOf(F, n); lineVec = vecOf(L, n); colSeq++;
          }

          // ---- the hexagons (AlphaEarth) ---------------------------------------------------
          let hexes = [], N = 0, res = -1, hexIndex = new Map(), hattrs = null, hmeta = {}, hcol = null, hexSeq = 0, hover = null;
          // per hexagon: [0] AlphaEarth change year (0 no data, 1 no single
          // year, else year - 2000), [1] how much it changed 1..255, [2] WSF new share, [3] WSF built share
          function recolorHex() {
            if (!N || !hattrs || hattrs.length !== 4 * N) { hcol = null; return; }
            hcol = new Uint8Array(4 * N);
            const Y = st.Y;
            for (let i = 0; i < N; i++) {
              const o = 4 * i, c = hattrs[o];
              let col = null, a = 0;
              if (st.gmode === "when") {
                // changed by the timeline's year: its year in viridis; no single
                // year, or changed later than the timeline's year: faint grey
                if (c >= 16 && 2000 + c <= Y) { col = virYear(2000 + c); a = A_FILL; }
                else if (c >= 1) { col = QUIET; a = A_QUIET; }
              } else {
                const s = hattrs[o + 1];
                if (s) { col = vir((s - 1) / 254); a = A_FILL; }
              }
              if (col) { hcol[o] = col[0]; hcol[o + 1] = col[1]; hcol[o + 2] = col[2]; hcol[o + 3] = a; }
            }
            hexSeq++;
          }

          // ---- counts for the timeline ------------------------------------------------------
          let counts = null;
          function countFootprints() {
            const n = polys ? polys.numRows : 0;
            if (!n || !attrs) return null;
            const c = {kind: "bld", years: {}, pre: 0, never: 0, total: n, src: {}};
            for (let y = YMIN; y <= YMAX; y++) c.years[y] = 0;
            for (let i = 0; i < n; i++) {
              const y = yearOf(i);
              if (y >= YMIN) c.years[y]++;
              else if (y === -1) c.pre++;
              if (attrs[4 * i] === 0) c.never++;
              const s = attrs[4 * i + 2];
              c.src[s] = (c.src[s] || 0) + 1;
            }
            return c;
          }
          const inView = (bb, b) => !(bb.east < b.getWest() || bb.west > b.getEast() || bb.north < b.getSouth() || bb.south > b.getNorth());
          function countRaster() {
            if (!map) return null;
            const b = map.getBounds();
            let zmax = -1;
            for (const t of rawTiles.values()) if (t.bbox && inView(t.bbox, b) && t.z > zmax) zmax = t.z;
            if (zmax < 0) return null;
            const h = new Float64Array(21);
            for (const t of rawTiles.values()) { if (!t.bbox || t.z !== zmax || !inView(t.bbox, b)) continue; for (let i = 0; i < t.k.length; i++) h[t.k[i]]++; }
            const c = {kind: "px", years: {}, pre: h[1], total: 0};
            for (let y = YMIN; y <= YMAX; y++) c.years[y] = 0;
            for (let k = 2; k <= 20; k++) c.years[wsfYear(k)] += h[k];
            for (let k = 1; k <= 20; k++) c.total += h[k];
            return c;
          }
          function countHex() {
            if (!N || !hattrs) return null;
            const c = {kind: "hex", years: {}, total: 0, quiet: 0, km2: hmeta.km2 || 0};
            for (let y = YMIN; y <= YMAX; y++) c.years[y] = 0;
            for (let i = 0; i < N; i++) { const v = hattrs[4 * i]; if (v >= 16) { c.years[2000 + v]++; c.total++; } else if (v === 1) c.quiet++; }
            return c;
          }
          let countT = null;
          function countsSoon() { clearTimeout(countT); countT = setTimeout(recount, 180); }
          const bldUp = () => !!(polys && polys.numRows && map && map.getZoom() >= BLDZ);
          // the timeline counts what is on: the buildings (when drawn and lit
          // by a year), else the AlphaEarth hexagons, else WSF's ground
          const tlKind = () => (st.on.bld && bldUp() && (st.bfill === "wyear" || st.bfill === "ayear") ? "bld" : st.on.hex ? "hex" : st.on.wsf ? "px" : (st.on.bld && (st.bfill === "wyear" || st.bfill === "ayear") ? "wait" : null));
          function recount() {
            const k = tlKind();
            counts = k === "bld" ? countFootprints() : k === "hex" ? countHex() : k === "px" ? countRaster() : null;
            renderDock();
          }

          // ---- the dock ---------------------------------------------------------------------
          let dockKind = null, colsEl = null, yrEl = null, footEl = null, playBtn = null, sliderEl = null;
          const minY = () => { const k = tlKind(); return k === "hex" || (k === "bld" && st.bfill === "ayear") ? 2018 : YMIN; };
          function buildTimeline() {
            dock.innerHTML = "";
            const g = el_("div", "at-grow");
            playBtn = el_("button", "at-play", ICON.play); playBtn.title = "play the years (space)";
            playBtn.onclick = () => togglePlay();
            colsEl = el_("div", "at-cols"); colsEl.setAttribute("role", "slider"); colsEl.setAttribute("aria-label", "year"); colsEl.tabIndex = 0;
            for (let y = YMIN; y <= YMAX; y++) { const c = el_("div", "at-col"); c.dataset.y = y; c.append(el_("div", "b"), el_("div", "y", String(y))); colsEl.appendChild(c); }
            const pick = (e) => { const r = colsEl.getBoundingClientRect(); const f = Math.max(0, Math.min(0.999, (e.clientX - r.left) / r.width)); setY(YMIN + Math.floor(f * (YMAX - YMIN + 1))); };
            colsEl.addEventListener("pointerdown", (e) => { stopPlay(); colsEl.setPointerCapture(e.pointerId); pick(e); });
            colsEl.addEventListener("pointermove", (e) => {
              if (colsEl.hasPointerCapture(e.pointerId)) pick(e);
              const col = e.target.closest && e.target.closest(".at-col");
              if (col && counts) showTip(col, Number(col.dataset.y)); else tip.style.display = "none";
            });
            colsEl.addEventListener("pointerleave", () => { tip.style.display = "none"; });
            colsEl.addEventListener("pointerup", (e) => { try { colsEl.releasePointerCapture(e.pointerId); } catch (err) {} });
            yrEl = el_("div", "at-yr");
            const mid = el_("div", "at-mid");
            sliderEl = el_("input", "at-slider"); sliderEl.type = "range"; sliderEl.min = YMIN; sliderEl.max = YMAX; sliderEl.step = 1; sliderEl.value = st.Y;
            sliderEl.setAttribute("aria-label", "timeline year");
            sliderEl.addEventListener("input", () => { stopPlay(); setY(Number(sliderEl.value)); if (Number(sliderEl.value) !== st.Y) sliderEl.value = st.Y; });
            mid.append(colsEl, sliderEl);
            g.append(playBtn, mid, yrEl);
            footEl = el_("div", "at-foot");
            dock.append(g, footEl);
          }
          const tipText = (y) => {
            const c = counts, v = c.years[y] || 0;
            if (c.kind === "hex") return y < 2018 ? "AlphaEarth starts in 2017: the first change it can date is 2018" : `${fmt(v)} hexagon${v === 1 ? "" : "s"} changed between the ${y - 1} and ${y} AlphaEarth pictures`;
            if (c.kind === "bld") return st.bfill === "ayear" ? `${y}: ${fmt(v)} building${v === 1 ? "" : "s"} whose ground changed that year (AlphaEarth)` : `${fmt(v)} building${v === 1 ? "" : "s"} first seen built ${wsfYearWindow(y)}`;
            return `${(100 * v / Math.max(1, c.total)).toFixed(1)}% of the built-up ground first seen ${wsfYearWindow(y)}`;
          };
          function showTip(col, y) {
            tip.textContent = tipText(y);
            const r = col.getBoundingClientRect(), p = pane.getBoundingClientRect();
            tip.style.left = Math.min(p.width - 170, Math.max(170, r.left + r.width / 2 - p.left)) + "px"; tip.style.top = (r.top - p.top - 6) + "px"; tip.style.display = "block";
          }
          function renderTimeline() {
            const kind = "tl";
            if (dockKind !== kind) { buildTimeline(); dockKind = kind; }
            const c = counts;
            let max = 1;
            if (c) for (let y = YMIN; y <= YMAX; y++) max = Math.max(max, c.years[y] || 0);
            colsEl.querySelectorAll(".at-col").forEach((col) => {
              const y = Number(col.dataset.y), v = c ? (c.years[y] || 0) : 0;
              const bEl = col.querySelector(".b");
              bEl.style.height = c ? (v ? Math.max(3, 52 * v / max) : 2) + "px" : "2px";
              bEl.style.background = c && c.kind === "hex" && y >= 2018 ? `rgba(${virYear(y).join(",")},${y <= st.Y ? 1 : 0.25})` : "";
              col.classList.toggle("cur", y === st.Y);
              col.classList.toggle("past", y < st.Y);
              col.classList.toggle("off", y < minY());
            });
            colsEl.setAttribute("aria-valuenow", String(st.Y));
            if (sliderEl && Number(sliderEl.value) !== st.Y) sliderEl.value = st.Y;
            const v = c ? (c.years[st.Y] || 0) : null;
            let head = "", sub = "";
            if (tlKind() === "hex") {
              head = st.Y < 2018 ? "AlphaEarth starts in 2017" : `between the ${st.Y - 1} and ${st.Y} pictures`;
              sub = !c ? (map && map.getZoom() < HEXZ ? `Zoom in to zoom ${HEXZ} or closer` : "Reading AlphaEarth…") : `${fmt(v)} of ${fmt(c.total + c.quiet)} hexagons changed clearly`;
            } else if (!c) { head = wsfYearWindow(st.Y); sub = tlKind() === "wait" ? `Buildings from zoom ${BLDZ}: zoom in, or turn on WSF` : "Reading…"; }
            else if (c.kind === "bld") {
              if (st.bfill === "ayear") { head = st.Y < 2018 ? "AlphaEarth starts in 2017" : `between the ${st.Y - 1} and ${st.Y} AlphaEarth pictures`; sub = meta.aef ? `${fmt(v)} buildings whose ground changed` : "Reading AlphaEarth for this area…"; }
              else { head = wsfYearWindow(st.Y); sub = `${fmt(v)} building${v === 1 ? "" : "s"} first seen built`; }
            } else { head = wsfYearWindow(st.Y); sub = `${(100 * v / Math.max(1, c.total)).toFixed(1)}% of the built-up ground here first seen`; }
            yrEl.innerHTML = `<b>${st.Y}</b><span>${head}<br>${sub}</span>`;
            const key = (col, a, lab, outline) => `<span class="at-key"><i style="background:${outline ? "none" : rgba(col, a)};${outline ? `border:1.5px solid ${rgba(col, a)}` : ""}"></i>${lab}</span>`;
            if (tlKind() === "hex") {
              footEl.innerHTML = "";
              const keys = el_("div", "at-keys");
              keys.innerHTML = st.gmode === "when"
                ? `<span class="at-key">2018 <i class="at-ramp" style="width:140px;background:linear-gradient(90deg,${virCss(8)})"></i> 2025</span>` + key(QUIET, .8, `no clear change${st.Y < 2025 ? `, or after ${st.Y}` : ""}`)
                : `<span class="at-key">moved little <i class="at-ramp" style="width:140px;background:linear-gradient(90deg,${virCss(8)})"></i> a lot, ${hmeta.y0 || 2017} to ${hmeta.y1 || 2025}</span>`;
              footEl.append(keys);
            } else {
              footEl.innerHTML = "";
              const keys = el_("div", "at-keys");
              const aef = st.bfill === "ayear" && tlKind() === "bld";
              keys.innerHTML = key(DEEP, .9, `${aef ? "changed" : "built"} in ${st.Y}`) + key(AMBER, .5, "earlier") + key(GREY, .8, aef ? "no clear change" : "standing by 2016", true)
                + (st.on.bld && !bldUp() ? `<span class="at-key">buildings from zoom ${BLDZ}</span>` : "");
              footEl.append(keys);
            }
          }
          function renderCheck() {
            dockKind = "check";
            const gp = meta.gap_share, c = counts;
            if (!bldUp() || !c || c.kind !== "bld") { dock.innerHTML = `<div class="at-sum"><h4>Where the map and WSF disagree</h4><div class="line"><i style="background:${rgba(COOL, .9)}"></i><span>Zoom in to zoom ${BLDZ} or closer to compare the buildings on the map with WSF's built-up ground.</span></div></div>`; return; }
            dock.innerHTML = `<div class="at-sum"><h4>Where the map and WSF disagree, in this area</h4>
              <div class="line"><i style="background:${rgba(COOL, .9)}"></i><span><b>${fmt(c.never)}</b> of ${fmt(c.total)} buildings on the map (${(100 * c.never / Math.max(1, c.total)).toFixed(1)}%) sit on ground WSF has never read as built-up. New, small, or not there?</span></div>
              <div class="line"><i style="background:${rgba(AMBER, .9)}"></i><span>${gp != null ? `<b>${(100 * gp).toFixed(0)}%</b> of WSF's built-up ground is 20 m or more from any building on the map.` : "Built-up ground 20 m or more from any building on the map."} Structures the map may be missing.</span></div></div>`;
          }
          function renderSources() {
            dockKind = "sources";
            const c = counts;
            if (!bldUp() || !c || c.kind !== "bld") { dock.innerHTML = `<div class="at-sum"><h4>Where the buildings come from</h4><div class="line"><i style="background:${rgba(SRC[0], .9)}"></i><span>Zoom in to zoom ${BLDZ} or closer to see the buildings on the map.</span></div></div>`; return; }
            const names = meta.sources || [];
            const order = Object.keys(c.src).map(Number).sort((a, b) => a - b);
            let other = 0;
            const rows = [];
            for (const s of order) { if (s < SRC.length) rows.push([names[s] || "unnamed", c.src[s], SRC[s]]); else other += c.src[s]; }
            if (other) rows.push(["other sources", other, OTHER]);
            const stack = rows.map(([, v, col]) => `<span style="flex:${v};background:${rgba(col, 1)}"></span>`).join("");
            const lines = rows.map(([nm, v, col]) => `<div class="line"><i style="background:${rgba(col, 1)}"></i><span><b>${nm}</b> ${fmt(v)} (${(100 * v / c.total).toFixed(1)}%)</span></div>`).join("");
            dock.innerHTML = `<div class="at-sum"><h4>Where the ${fmt(c.total)} buildings on the map come from</h4><div class="at-stack">${stack}</div>${lines}</div>`;
          }
          function renderDock() {
            const bldFill = st.on.bld && (st.bfill === "check" || st.bfill === "source");
            if (bldFill && (bldUp() || !(st.on.hex || st.on.wsf))) { if (st.bfill === "check") renderCheck(); else renderSources(); }
            else if (tlKind()) renderTimeline();
            else { dockKind = null; dock.innerHTML = `<div class="at-sum"><h4>Nothing on to count</h4><div class="line"><i style="background:${rgba(AMBER, .8)}"></i><span>Turn on Buildings, WSF or AlphaEarth under Layers (B, W, A).</span></div></div>`; }
          }

          // ---- state changes ---------------------------------------------------------------
          function toggleLayer(k) {
            st.on[k] = !st.on[k];
            if (k === "img" && !st.on.img) stopFlip(true);
            send(k === "hex" ? "hex" : "layers");
            if (st.Y < minY()) st.Y = YMAX;
            styleRows(); recolor(); recolorHex(); recount(); update();
          }
          function setBfill(k) {
            st.bfill = k;
            if (!st.on.bld) { st.on.bld = true; send("layers"); }
            if (k === "ayear" && !meta.aef) send("witness");
            if (st.Y < minY()) st.Y = YMAX;
            styleRows(); recolor(); recount(); update();
          }
          function setY(y) {
            y = Math.max(minY(), Math.min(YMAX, y));
            if (y === st.Y) return;
            st.Y = y;
            recolor(); recolorHex(); renderDock(); update();
          }
          let playT = null;
          function stopPlay() { st.playing = false; clearTimeout(playT); if (playBtn) { playBtn.innerHTML = ICON.play; playBtn.title = "play the years (space)"; } }
          function togglePlay() {
            if (st.playing) { stopPlay(); return; }
            if (!tlKind()) return;
            st.playing = true;
            playBtn.innerHTML = ICON.pause; playBtn.title = "pause (space)";
            if (st.Y >= YMAX) setY(minY());
            const step = () => { if (!st.playing) return; if (st.Y >= YMAX) { stopPlay(); return; } setY(st.Y + 1); playT = setTimeout(step, 900); };
            playT = setTimeout(step, 700);
          }

          // ---- the before and after: automatic, from the clicked thing's year ----------------
          // the imagery covers 2022 to 2025; a change dated inside it gets the
          // year before and the year after, flipped; anything earlier has no
          // "before" picture and says so
          function flipPair(y) {
            if (!y || y < S2Y[0] + 1 || y > S2Y[S2Y.length - 1]) return null;
            return [y - 1, Math.min(y + 1, S2Y[S2Y.length - 1])];
          }
          let flipT = null;
          function startFlip(pair, why) {
            stopFlip(true);
            st.flip = {a: pair[0], b: pair[1], cur: pair[0], why, wasImagery: st.on.img, paused: false};
            if (!st.on.img) { st.on.img = true; styleRows(); recolor(); }
            st.imgYear = pair[0];
            const tick = () => {
              const f = st.flip; if (!f) return;
              if (!f.paused) { f.cur = f.cur === f.a ? f.b : f.a; st.imgYear = f.cur; styleFlip(); styleImgRow(); update(); }
              flipT = setTimeout(tick, 1100);
            };
            styleFlip(); update();
            flipT = setTimeout(tick, 1400);
          }
          function stopFlip(keep) {
            clearTimeout(flipT); flipT = null;
            const f = st.flip; st.flip = null;
            if (f && !keep && !f.wasImagery) { st.on.img = false; recolor(); }
            if (f) st.imgYear = S2Y[S2Y.length - 1];
            imgTag.style.display = "none";
            styleRows(); update();
          }
          function styleFlip() {
            const f = st.flip;
            if (!f) { imgTag.style.display = "none"; return; }
            imgTag.style.display = "block";
            imgTag.textContent = f.cur === f.a ? `Imagery ${f.a}, before` : `Imagery ${f.b}, after`;
            card.querySelectorAll("[data-flip]").forEach((b) => b.classList.toggle("on", Number(b.dataset.flip) === f.cur));
            const pb = card.querySelector("[data-act=pause]");
            if (pb) pb.textContent = f.paused ? "Flip" : "Hold";
          }

          // ---- the layers --------------------------------------------------------------------
          let map = null, ov = null;
          let gapImg = null, gapSeq = 0, s2Shown = [];
          const slot = () => { const want = cfg.labels_slot || "watername_ocean"; const s = map && map.getStyle && map.getStyle(); if (!s || !s.layers || s.layers.some((x) => x.id === want)) return want; const l = s.layers.find((x) => x.type === "symbol"); return (l && l.id) || want; };
          const s2Layer = (year) => new TileLayer({
            id: "s2-" + year + "-g" + (cfg.s2_gen || 0),
            getTileData: async ({index, signal}) => { const u8 = await ask("s2", year, index, signal); return u8 ? pngBitmap(u8) : null; },
            onTileError: (e) => { if (!e || e.name !== "AbortError") say("s2 tile: " + ((e && e.message) || e)); },
            tileSize: cfg.tile || 256, minZoom: cfg.s2_min_z || 7, maxZoom: 14, refinementStrategy: "best-available", beforeId: slot(),
            visible: year === st.imgYear,
            renderSubLayers: (p) => { if (!p.data) return null; const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]}); },
          });
          const growthLayer = () => new TileLayer({
            id: "wsf-growth",
            getTileData: ({index, signal, bbox}) => rawTile({index, signal, bbox}),
            onTileError: (e) => { if (!e || e.name !== "AbortError") say("wsf tile: " + ((e && e.message) || e)); },
            tileSize: cfg.tile || 256, minZoom: 0, maxZoom: 14, extent: cfg.extent || null, refinementStrategy: "best-available",
            // over the imagery, never: two colour pictures on top of each other read as neither
            visible: !!st.on.wsf, beforeId: slot(),
            updateTriggers: {renderSubLayers: [st.Y]},
            renderSubLayers: (p) => { if (!p.data) return null; const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {id: p.id + "-" + st.Y, data: null, image: paintRaw(p.data, st.Y), bounds: [west, south, east, north]}); },
          });
          const ring = (h) => { try { return cellToBoundary(h, true); } catch (e) { return null; } };
          const outline = (id, h, color, width) => { const r = h ? ring(h) : null; return r ? new PathLayer({id, data: [r], getPath: (d) => d, getColor: color, widthUnits: "pixels", getWidth: width, beforeId: slot()}) : null; };
          function layers() {
            const out = [];
            if (st.on.img) {
              // every imagery year asked for stays mounted (hidden when not
              // shown), so a flip is instant once both years have loaded
              if (!s2Shown.includes(st.imgYear)) s2Shown = s2Shown.concat([st.imgYear]).slice(-4);
              for (const y of s2Shown) out.push(s2Layer(y));
            }
            out.push(growthLayer());
            if (st.on.hex && hcol && map && map.getZoom() >= HEXZ) out.push(new H3HexagonLayer({
              id: "hexes", data: {length: N}, getHexagon: (_, {index}) => hexes[index],
              getFillColor: (_, {index}) => [hcol[4 * index], hcol[4 * index + 1], hcol[4 * index + 2], hcol[4 * index + 3]],
              updateTriggers: {getFillColor: [hexSeq], getHexagon: [hexSeq]},
              filled: true, stroked: false, extruded: false, highPrecision: true, pickable: false, beforeId: slot(),
              // faint over the imagery, fainter while a before and after flips: the picture is the point then
              opacity: st.flip ? 0.2 : (st.on.img ? 0.55 : 1),
            }));
            if (st.on.bld && st.bfill === "check" && gapImg && meta.box && bldUp()) out.push(new BitmapLayer({id: "gap-" + gapSeq, image: gapImg, bounds: meta.box, opacity: st.on.img ? 0.7 : 1, beforeId: slot()}));
            if (st.on.bld && bldUp() && fillVec) out.push(new GeoArrowPolygonLayer({
              id: "fp-" + polysSeq, data: polys, filled: true, stroked: true,
              getFillColor: fillVec, getLineColor: lineVec,
              updateTriggers: {getFillColor: [colSeq], getLineColor: [colSeq]},
              lineWidthUnits: "pixels", getLineWidth: st.on.img ? 1.3 : 1, lineWidthMinPixels: 0.8, pickable: false, beforeId: slot(),
            }));
            if (st.on.hex) { const h = outline("hover", hover, [24, 32, 40, 200], 1.5); if (h) out.push(h); }
            if (cfg.hit) { const p = outline("picked", cfg.hit, [...COOL, 255], 3); if (p) out.push(p); }
            return out;
          }
          function update() { if (ov) ov.setProps({layers: layers()}); }
          function labels(on) {
            if (!map || !map.isStyleLoaded()) return;
            (map.getStyle().layers || []).forEach((l) => { if (l.layout && l.layout["text-field"] !== undefined) map.setLayoutProperty(l.id, "visibility", on ? "visible" : "none"); });
          }
          const hideBuildings = (m) => (m.getStyle().layers || []).forEach((l) => { if (l["source-layer"] === "building" || /^building/.test(l.id)) m.setLayoutProperty(l.id, "visibility", "none"); });

          // ---- the card ----------------------------------------------------------------------
          // a small chart of AlphaEarth's year-to-year steps against the quiet
          // level: what "the ground changed in 2021" is made of
          function spark(steps, years, D0) {
            if (!steps || !steps.length || D0 == null) return "";
            const W = 280, H = 64, pad = 16, n = steps.length, bw = (W - 8) / n;
            const vmax = Math.max(D0 * 1.6, ...steps.filter((v) => v != null));
            const yOf = (v) => H - pad - (H - pad - 4) * Math.min(1, v / vmax);
            let s = `<svg class="at-spark" width="${W}" height="${H}" role="img" aria-label="AlphaEarth year-to-year change">`;
            steps.forEach((v, i) => {
              if (v == null) return;
              const x = 4 + i * bw, y = yOf(v), up = v > D0;
              s += `<rect x="${x + 2}" y="${y}" width="${bw - 4}" height="${H - pad - y}" rx="2" fill="${up ? rgba(DEEP, .9) : "rgba(24,32,40,.22)"}"><title>${years[i] - 1} to ${years[i]}: ${v.toFixed(3)}${up ? " (above the quiet level)" : ""}</title></rect>`;
              s += `<text x="${x + bw / 2}" y="${H - 3}" text-anchor="middle">’${String(years[i]).slice(-2)}</text>`;
            });
            const qy = yOf(D0);
            s += `<line x1="2" x2="${W - 2}" y1="${qy}" y2="${qy}" stroke="rgba(24,32,40,.55)" stroke-dasharray="3 3"/><text x="${W - 4}" y="${qy - 3}" text-anchor="end">quiet level</text></svg>`;
            return s;
          }
          function renderCard() {
            let c = null;
            try { c = JSON.parse(model.get("card") || "null"); } catch (e) { c = null; }
            if (!c || !c.kind) { card.style.display = "none"; if (st.flip) stopFlip(); if (st.picked !== -1) { st.picked = -1; recolor(); update(); } return; }
            const row = (col, html) => `<div class="row"><i style="background:${col}"></i><div>${html}</div></div>`;
            let h = `<button class="x" title="close (Esc)">×</button>`;
            if (c.place && c.place.length) h += `<div class="place">${c.place.join(", ")}</div>`;
            let changeY = null, changeWhy = "";
            if (c.kind === "footprint") {
              h += `<h3>${c.what ? c.what[0].toUpperCase() + c.what.slice(1) : "A building on the map"}</h3>`;
              const k = c.wk || 0;
              if (k === 1) h += row(rgba(GREY, .9), `WSF: already standing when its record opens, July 2016.`);
              else if (k >= 2) { h += row(rgba(DEEP, 1), `WSF first saw it built <b>${wsfWindow(k)}</b>.`); changeY = wsfYear(k); changeWhy = "WSF"; }
              else h += row(rgba(COOL, 1), `WSF has <b>never</b> read it as built-up. It may be new, small, or not there.`);
              if (c.npx) h += `<div class="sub">${c.nbuilt} of the ${c.npx} WSF pixels under it are built-up</div>`;
              const a = c.acode;
              if (a === 0) h += row("rgba(24,32,40,.25)", `AlphaEarth not read for this area yet. <button class="at-chip" data-act="aef" style="margin-top:6px;display:block">Check with AlphaEarth</button>`);
              else if (a === 1) h += row(rgba(GREY, .9), `AlphaEarth: no year-to-year jump above the quiet level, 2017 to 2025. The ground under it looks the same throughout.`);
              else if (a === 2) h += row("rgba(24,32,40,.25)", `AlphaEarth: no data here.`);
              else { h += row(rgba(AMBER, 1), `AlphaEarth: the ground first changed clearly between its <b>${2000 + a - 1} and ${2000 + a}</b> pictures.`); if (!changeY) { changeY = 2000 + a; changeWhy = "AlphaEarth"; } }
              if (c.asteps) h += spark(c.asteps, c.ayears, c.aD0);
              if (k >= 2 && a >= 16) { const d = Math.abs(wsfYear(k) - (2000 + a)); h += `<div class="sub">${d <= 1 ? "WSF and AlphaEarth agree on the year." : `WSF and AlphaEarth differ by ${d} years.`}</div>`; }
              const bits = [c.dataset ? `On the map from <b>${c.dataset}</b>` : "On the map from an unnamed source", c.updated ? `last edited ${c.updated}` : null].filter(Boolean).join(", ");
              const dims = [c.height != null ? `${Math.round(c.height)} m tall` : null, c.floors != null ? `${c.floors} floors` : null].filter(Boolean).join(", ");
              h += row(rgba(SRC[Math.min(c.src ?? 99, SRC.length)] || OTHER, 1), bits + "." + (dims ? ` ${dims[0].toUpperCase() + dims.slice(1)}.` : ""));
            } else if (c.kind === "ground") {
              h += `<h3>No building on the map here</h3>`;
              const k = c.wk || 0;
              if (k >= 1) {
                const why = c.near ? "It is within 20 m of a building on the map, so likely a yard, a lane or that building's edge." : "No building on the map within 20 m: something may be here that the map doesn't have.";
                h += row(rgba(AMBER, c.near ? .5 : 1), `WSF read this ground as built-up <b>${wsfWindow(k)}</b>. ${why}`);
                if (k >= 2) { changeY = wsfYear(k); changeWhy = "WSF"; }
              } else h += row("rgba(24,32,40,.25)", `WSF has never read this ground as built-up either.`);
            } else if (c.kind === "hex") {
              h += `<h3>This hexagon of ground</h3>`;
              const lvl = c.level == null ? null : (c.level >= 0.75 ? "a lot" : c.level >= 0.4 ? "a fair amount" : c.level >= 0.15 ? "a little" : "barely");
              if (c.when >= 2018) { h += row(rgba(DEEP, 1), `AlphaEarth: the ground first changed clearly between its <b>${c.when - 1} and ${c.when}</b> pictures${lvl ? `, and ${lvl} over ${c.y0} to ${c.y1}` : ""}.`); changeY = c.when; changeWhy = "AlphaEarth"; }
              else if (c.when === -1) h += row(rgba(GREY, .9), `AlphaEarth: no year stood out${lvl ? `; it moved ${lvl} over ${c.y0} to ${c.y1}` : ""}. Every year-to-year step is under the quiet level.`);
              else h += row("rgba(24,32,40,.25)", `AlphaEarth: no data here.`);
              h += spark(c.steps, c.step_years, c.D0);
              if (c.p_built > 0) h += row(rgba(AMBER, .7), `WSF: <b>${Math.round(100 * c.p_built)}%</b> of it built-up by ${c.y1}` + (c.p_new > 0 ? `, ${(100 * c.p_new).toFixed(1)}% new since ${c.y0 + 1}${c.byear > 0 ? `, most of that in ${c.byear}` : ""}.` : ", all of it before " + (c.y0 + 1) + "."));
              else h += row("rgba(24,32,40,.25)", `WSF: no built-up ground in it.`);
              h += `<div class="sub">${c.km2 ? c.km2.toFixed(3) + " km² hexagon. " : ""}The quiet level is the step size that 95% of the ground WSF says stayed the same stays under, in this view.</div>`;
            } else if (c.kind === "note") h += `<h3 style="font-size:15px">${c.title || ""}</h3>`;
            // the before and after
            const pair = flipPair(changeY);
            if (changeY) {
              if (pair) h += `<div class="flip"><div>Imagery, before and after (${changeWhy} ${changeY}):</div><div class="yrs"><button class="at-chip" data-flip="${pair[0]}">${pair[0]} before</button><button class="at-chip" data-flip="${pair[1]}">${pair[1]} after</button><button class="at-chip" data-act="pause">Hold</button></div></div>`;
              else h += `<div class="flip muted">${changeWhy} dates this to ${changeY}, before the imagery starts in ${S2Y[0]}, so there's no picture from before it to compare.</div>`;
            }
            card.innerHTML = h;
            card.style.display = "block";
            card.querySelector(".x").onclick = () => closeCard();
            const ab = card.querySelector("[data-act=aef]");
            if (ab) ab.onclick = () => { ab.disabled = true; ab.textContent = "Reading AlphaEarth…"; send("witness"); };
            card.querySelectorAll("[data-flip]").forEach((b) => { b.onclick = () => { if (!st.flip) return; st.flip.paused = true; st.flip.cur = Number(b.dataset.flip); st.imgYear = st.flip.cur; styleFlip(); update(); }; });
            const pb = card.querySelector("[data-act=pause]");
            if (pb) pb.onclick = () => { if (!st.flip) return; st.flip.paused = !st.flip.paused; styleFlip(); };
            const same = st.flip && pair && st.flip.a === pair[0] && st.flip.b === pair[1] && st.flip.n === c.n;
            if (pair && !same) { startFlip(pair, changeWhy); st.flip.n = c.n; }
            else if (!pair && st.flip) stopFlip();
            else styleFlip();
            const idx = c.kind === "footprint" && c.idx != null ? c.idx : -1;
            if (idx !== st.picked) { st.picked = idx; recolor(); update(); }
          }
          function closeCard() { model.set("pick", JSON.stringify({close: true, n: ++seq})); model.save_changes(); card.style.display = "none"; if (st.flip) stopFlip(); st.picked = -1; recolor(); update(); }

          // ---- search ------------------------------------------------------------------------
          const PHOTON = "https://photon.komoot.io/api/";
          let gcHits = [], gcSel = -1, gcTimer = null, gcSeq = 0;
          const hitName = (f) => { const p = f.properties || {}; return [p.name, p.street && !p.name ? p.street : null, p.city && p.city !== p.name ? p.city : null, p.state, p.country].filter(Boolean).join(", "); };
          const hitKind = (f) => { const p = f.properties || {}; return [p.osm_value, p.type].filter((x) => x && x !== "yes").join(", "); };
          const gcHide = () => { hits.style.display = "none"; hits.replaceChildren(); gcSel = -1; };
          const gcShow = () => {
            hits.replaceChildren();
            if (!gcHits.length) { gcHide(); return; }
            gcHits.forEach((f, i) => { const r = el_("div", "at-hit" + (i === gcSel ? " sel" : "")); r.textContent = hitName(f); const k = el_("small"); k.textContent = hitKind(f); r.appendChild(k); r.onmousedown = (e) => { e.preventDefault(); gcFly(f); }; r.onmouseenter = () => { gcSel = i; gcShow(); }; hits.appendChild(r); });
            hits.style.display = "block";
          };
          const gcAsk = async () => {
            const q = gc.value.trim();
            if (q.length < 2) { gcHits = []; gcHide(); return; }
            const s = ++gcSeq;
            const params = new URLSearchParams({q, limit: "6", lang: "en"});
            if (map) { const c = map.getCenter(); params.set("lon", c.lng.toFixed(4)); params.set("lat", c.lat.toFixed(4)); }
            try { const r = await fetch(PHOTON + "?" + params.toString()); const d = await r.json(); if (s !== gcSeq) return; gcHits = (d.features || []).filter((f) => f.geometry && f.geometry.coordinates); gcSel = gcHits.length ? 0 : -1; gcShow(); }
            catch (e) { if (s === gcSeq) note("search: " + e.message, 4000); }
          };
          const gcFly = (f) => {
            const [lon, lat] = f.geometry.coordinates;
            const ext = (f.properties || {}).extent;
            let zoom = 13.5;
            if (ext && ext.length === 4) { const span = Math.max(Math.abs(ext[2] - ext[0]), Math.abs(ext[1] - ext[3]) * 2, 0.01); zoom = Math.log2(360 * ((mapEl.clientWidth || 1200) / 512) / span) - 0.3; }
            zoom = Math.max(4, Math.min(16, zoom));
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
            s.textContent = ["notebook-actions-dropdown", "cell-actions-button", "drag-button", "expand-output-button", "fullscreen-output-button"].map((t) => "html." + FIT_CLS + " [data-testid='" + t + "']").join(",") + ",html." + FIT_CLS + " div[class*='top-[25vh]']{display:none!important}html." + FIT_CLS + "{overflow:hidden}";
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
          bMore.onclick = (e) => { e.stopPropagation(); more.style.display = more.style.display === "block" ? "none" : "block"; bMore.classList.toggle("on", more.style.display === "block"); };
          more.addEventListener("click", (e) => e.stopPropagation());
          root.addEventListener("click", () => { if (more.style.display === "block") { more.style.display = "none"; bMore.classList.remove("on"); } });
          window.addEventListener("resize", () => { if (st.fit) sizes(); });

          // ---- keys ---------------------------------------------------------------------------
          root.tabIndex = 0;
          const onKey = (e) => {
            const path = e.composedPath ? e.composedPath() : [];
            if (!st.fit && !path.includes(root)) return;
            const tgt = path[0] || e.target;
            if (tgt && /^(INPUT|SELECT|TEXTAREA)$/.test(tgt.tagName)) return;
            const k = e.key;
            // the slider's keys (B W A S layers, Q building colour, [ ] F
            // imagery year, ; ' brightness, 1-9 the choices of the last row
            // shown, - = _ + AlphaEarth years, L labels, X window), plus the
            // arrows and space for the timeline, / search
            const stepImg = (d) => { const i = S2Y.indexOf(st.imgYear); st.imgYear = S2Y[Math.max(0, Math.min(S2Y.length - 1, i + d))]; stopFlip(true); if (!st.on.img) toggleLayer("img"); styleRows(); update(); };
            const lo = st.y0, hi = st.y1;
            if (k === "ArrowLeft" || k === "ArrowRight") { stopPlay(); setY(st.Y + (k === "ArrowRight" ? 1 : -1)); }
            else if (k === " ") { if (tgt && tgt.tagName === "BUTTON") return; togglePlay(); }
            else if (k === "b" || k === "B") toggleLayer("bld");
            else if (k === "w" || k === "W") toggleLayer("wsf");
            else if (k === "a" || k === "A" || k === "e" || k === "E") toggleLayer("hex");
            else if (k === "s" || k === "S" || k === "i" || k === "I") toggleLayer("img");
            else if (k === "q" || k === "Q") { const i = BFILLS.findIndex((f) => f[0] === st.bfill); setBfill(BFILLS[(i + 1) % BFILLS.length][0]); }
            else if (k === "[" || k === "]") stepImg(k === "]" ? 1 : -1);
            else if (k === "f" || k === "F") { st.imgYear = st.imgYear === S2Y[S2Y.length - 1] ? S2Y[0] : S2Y[S2Y.length - 1]; stopFlip(true); if (!st.on.img) toggleLayer("img"); styleRows(); update(); }
            else if (k === ";" || k === "'") { st.s2scale = Math.round(10 * Math.max(0.3, Math.min(2.5, st.s2scale + (k === "'" ? 0.1 : -0.1)))) / 10; gam.value = st.s2scale; clearTimeout(gamT); gamT = setTimeout(() => send("s2scale"), 250); }
            else if (k >= "1" && k <= "9") {
              // the last row shown: imagery, else AlphaEarth, else buildings
              const n = Number(k) - 1;
              if (st.on.img && S2Y[n]) { st.imgYear = S2Y[n]; stopFlip(true); styleRows(); update(); }
              else if (st.on.hex && n < 2) { st.gmode = n ? "much" : "when"; recolorHex(); styleRows(); renderDock(); update(); }
              else if (st.on.bld && BFILLS[n]) setBfill(BFILLS[n][0]);
            }
            else if (k === "-" || k === "=") { const v = Math.max(2017, Math.min(hi - 1, lo + (k === "=" ? 1 : -1))); if (v !== lo) { st.y0 = v; send("aef"); note(`AlphaEarth years ${st.y0} to ${st.y1}`, 1800); } }
            else if (k === "_" || k === "+") { const v = Math.max(lo + 1, Math.min(2025, hi + (k === "+" ? 1 : -1))); if (v !== hi) { st.y1 = v; send("aef"); note(`AlphaEarth years ${st.y0} to ${st.y1}`, 1800); } }
            else if (k === "l" || k === "L") { st.labels = !st.labels; labels(st.labels); swLab.sty(); send("labels"); }
            else if (k === "x" || k === "X") { st.fit = !st.fit; sizes(); }
            else if (k === "/") gc.focus();
            else if (k === "Escape") { if (about.style.display === "flex") about.style.display = "none"; else if (more.style.display === "block") more.style.display = "none"; else if (card.style.display === "block") closeCard(); else if (st.fit) { st.fit = false; sizes(); } }
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
            if (v.zoom >= BLDZ) say("reading footprints…");
          }
          const cellAt = (ll) => { if (res < 0) return null; try { const h = latLngToCell(ll.lat, ll.lng, res); return hexIndex.has(h) ? h : null; } catch (e) { return null; } };
          const adminAt = (pt) => {
            const out = {};
            const one = (k) => { const id = "ov-div-" + k; if (!map.getLayer(id)) return null; const fs = map.queryRenderedFeatures(pt, {layers: [id]}); return fs && fs.length ? fs[0].properties : null; };
            try { const r = one("region"); if (r) out.region = r["@name"] || r.names || null; const c = one("county"); if (c) out.county = c["@name"] || c.names || null; } catch (e) {}
            return out;
          };
          function boot() {
            const home = cfg.home || {longitude: 3.6, latitude: 6.46, zoom: 11};
            map = new maplibregl.Map({container: mapEl, style: STYLE, center: [home.longitude, home.latitude], zoom: home.zoom, attributionControl: {compact: true}});
            map.keyboard.disable();
            map.addControl(new maplibregl.NavigationControl({showCompass: false}), "bottom-right");
            ov = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck: " + (e && e.message ? e.message : e))});
            map.addControl(ov);
            map.on("load", () => {
              hideBuildings(map);
              labels(st.labels);
              if (cfg.div_pm && !map.getSource("ov-div")) {
                try {
                  map.addSource("ov-div", {type: "vector", url: "pmtiles://" + cfg.div_pm});
                  for (const k of ["region", "county"]) map.addLayer({id: "ov-div-" + k, type: "fill", source: "ov-div", "source-layer": "division_area", filter: ["all", ["==", ["get", "subtype"], k], ["==", ["get", "class"], "land"]], paint: {"fill-opacity": 0}}, slot());
                } catch (e) { console.error("divisions", e); }
              }
              update(); sendView();
            });
            map.on("moveend", () => { sendView(); countsSoon(); });
            map.on("zoomend", () => { recount(); update(); });
            map.on("mousemove", (e) => { if (!st.on.hex) return; const h = cellAt(e.lngLat); if (h !== hover) { hover = h; update(); } });
            map.on("click", (e) => {
              const h = st.on.hex ? cellAt(e.lngLat) : null;
              model.set("pick", JSON.stringify({cell: h, lon: e.lngLat.lng, lat: e.lngLat.lat, admin: adminAt(e.point), n: ++seq}));
              model.save_changes();
            });
            map.on("error", (ev) => { if (ev && ev.error && ev.error.message && !/tile|404/i.test(ev.error.message)) say("map: " + ev.error.message); });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} }).observe(mapEl);
            window.__atMaps = () => [map];
            // for tests: the centre of the first hexagon whose ground changed in year y
            window.__atHexAt = (y) => { for (let i = 0; i < N; i++) if (hattrs && hattrs[4 * i] === y - 2000) { const r = cellToBoundary(hexes[i], true); const c = r.slice(0, -1).reduce((a, p) => [a[0] + p[0] / (r.length - 1), a[1] + p[1] / (r.length - 1)], [0, 0]); return c; } return null; };
            window.__atState = () => ({st: Object.assign({}, st, {flip: st.flip && {a: st.flip.a, b: st.flip.b, cur: st.flip.cur}}), polys: polys ? polys.numRows : 0, counts, tiles: tstat, raw: rawTiles.size, meta, hex: N, hmeta});
          }

          // ---- the kernel's data -----------------------------------------------------------------
          const loadPolys = () => {
            const u8 = bytesOf(model.get("polys"));
            polysSeq++; polys = null;
            if (u8 && u8.length) { try { polys = arrow.tableFromIPC(new Uint8Array(copyOf(u8))); } catch (e) { say("footprints: " + e.message); } }
            recolor(); recount(); update();
          };
          const loadAttrs = () => {
            const u8 = bytesOf(model.get("battrs"));
            attrs = u8 && u8.length ? new Uint8Array(copyOf(u8)) : null;
            try { meta = JSON.parse(model.get("bmeta") || "{}"); } catch (e) { meta = {}; }
            recolor(); recount(); update();
          };
          const loadGap = async () => {
            const u8 = bytesOf(model.get("gap"));
            gapImg = null;
            if (u8 && u8.length) { try { gapImg = await createImageBitmap(new Blob([new Uint8Array(copyOf(u8))], {type: "image/png"})); } catch (e) { gapImg = null; } }
            gapSeq++; update();
          };
          const loadHex = () => {
            const cb = bytesOf(model.get("cells")), ab = bytesOf(model.get("hattrs"));
            try { hmeta = JSON.parse(model.get("hmeta") || "{}"); } catch (e) { hmeta = {}; }
            if (!cb || !cb.length) { hexes = []; N = 0; hexIndex = new Map(); res = -1; hattrs = null; hcol = null; recount(); update(); return; }
            const ids = new BigUint64Array(copyOf(cb));
            N = ids.length; hexes = new Array(N); hexIndex = new Map();
            for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
            try { res = getResolution(hexes[0]); } catch (e) { res = -1; }
            hattrs = ab && ab.length === 4 * N ? new Uint8Array(copyOf(ab)) : null;
            recolorHex(); recount(); update();
          };
          let pendHex = null;
          const hexSoon = () => { clearTimeout(pendHex); pendHex = setTimeout(loadHex, 0); };
          model.on("change:cells", hexSoon);
          model.on("change:hattrs", hexSoon);
          model.on("change:hmeta", hexSoon);
          model.on("change:polys", loadPolys);
          model.on("change:battrs", loadAttrs);
          model.on("change:bmeta", loadAttrs);
          model.on("change:gap", loadGap);
          model.on("change:card", renderCard);
          model.on("change:status", () => say(model.get("status")));
          model.on("change:config", () => { try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; } update(); });
          try {
            sizes(); boot();
            styleRows();
            loadPolys(); loadAttrs(); loadGap(); loadHex(); renderDock(); say(model.get("status"));
          } catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { window.removeEventListener("keydown", onKey); stopFlip(true); document.documentElement.classList.remove(FIT_CLS); try { map && map.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (Atlas,)


@app.cell
def _(
    AEF_FROM0,
    AEF_TO0,
    AEF_YEARS_ALL,
    Atlas,
    BFILL0,
    BLD_ZOOM,
    FILLS,
    FILL_NAMES,
    FILL_SHORT,
    HEX_ZOOM,
    HOME,
    LABELS_SLOT,
    LAYERS0,
    OV_DIV_PM,
    OV_RELEASE,
    RASTER_TILE,
    S2_SCALE0,
    S2_TILE_MIN_Z,
    S2_YEAR0,
    S2_YEARS,
    VIEW_H,
    VIRIDIS,
    ALPHA_FILL,
    ALPHA_QUIET,
    json,
    mo,
    wsf_bounds,
):
    # ---- the map: built ONCE, empty; never re-runs for a parameter ---------------
    # as an app (marimo run) it fills the window from the start; in the editor
    # it sits in the page (X fills the window, Esc brings it back)
    try:
        _fit = mo.app_meta().mode == "run"
    except Exception:
        _fit = False
    pair = Atlas(config=json.dumps({
        "height": VIEW_H, "home": dict(HOME), "labels": True, "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
        "s2_year": S2_YEAR0, "s2_scale": S2_SCALE0, "s2_gen": 0, "fill": FILLS[0],
        "s2_years": list(S2_YEARS), "s2_min_z": S2_TILE_MIN_Z,
        "aef_from": AEF_FROM0, "aef_to": AEF_TO0, "aef_years": list(AEF_YEARS_ALL),
        "fills": [[f, FILL_SHORT[f], FILL_NAMES[f]] for f in FILLS],
        "hex_zoom": HEX_ZOOM, "extent": list(wsf_bounds),
        "div_pm": OV_DIV_PM, "bld_zoom": BLD_ZOOM, "ov_release": OV_RELEASE, "fit": _fit,
        "viridis": VIRIDIS, "alpha_fill": ALPHA_FILL, "alpha_quiet": ALPHA_QUIET,
        "layers0": {"bld": LAYERS0["bld"], "wsf": LAYERS0["wsf"], "hex": LAYERS0["hex"], "img": LAYERS0["s2"]},
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "pending_force": False, "task": None, "loop": None,
        "s2y": S2_YEAR0, "s2scale": S2_SCALE0, "s2gen": 0, "fill": FILLS[0], "labels": True,
        "y0": AEF_FROM0, "y1": AEF_TO0,
        "hit": None, "pick_n": None, "card": None, "memo": {}, "aef": {}, "wsf": {}, "h_cam": None, "h_ctl": None, "h_pick": None,
        "runs": 0,
        # the footprints under the box (`bld`), the box they were read for
        # (`obox`), and which layers are on
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
    Image,
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
    io,
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
        if src == "wsfidx":
            return await wsf_tile_png(z, x, y, raw=True)
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

    # ---- the hexagons' legend (the only kernel legend left: the browser keys
    # the footprints and the growth raster itself) -----------------------------
    def _legend():
        fr, on = HOLD["frame"], HOLD["on"]
        if on.get("hex") and fr is not None:
            pair.legend = json.dumps(fr["legend"](HOLD["fill"]))
        else:
            pair.legend = "[]"

    def _status():
        bits = [x for x in (HOLD.get("bld_status"), HOLD.get("hex_status")) if x]
        _say(" · ".join(bits) if bits else "")

    # ---- the hexagons (the pair's fold) ---------------------------------------
    def _hexes_off(msg=""):
        if HOLD["sent"] is not None:
            with pair.hold_sync():
                pair.cells, pair.hattrs = b"", b""
            HOLD["sent"] = None
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = None, None, None, None
        _cfg(hit=None)
        HOLD["hex_status"] = msg

    def _paint():
        fr = HOLD["frame"]
        if fr is None:
            return False
        _cfg(hit=format(HOLD["hit"], "x") if HOLD["hit"] else None)
        if HOLD["sent"] is not fr:
            # 4 bytes per hexagon, coloured in the browser for the timeline's
            # year: the AlphaEarth change year (0 no data, 1 no single year,
            # else year - 2000), how much it moved over the window (1..255 over
            # this view's p2..p98, 0 no data), the WSF new and built shares
            when, d = fr["when"], fr["disp"]
            wc = np.where(when > 0, when - 2000, np.where(when == -1, 1, 0)).astype(np.uint8)
            lo, hi = fr["shift_lo"], fr["shift_hi"]
            sh = np.where(np.isnan(d), 0, 1 + 254 * np.clip((np.nan_to_num(d) - lo) / max(hi - lo, 1e-9), 0, 1)).astype(np.uint8)
            pn = (255 * np.clip(fr["p_new"], 0, 1)).astype(np.uint8)
            pb = (255 * np.clip(fr["p_built"], 0, 1)).astype(np.uint8)
            with pair.hold_sync():
                pair.cells = fr["cellid"].astype("<u8").tobytes()
                pair.hattrs = np.ascontiguousarray(np.stack([wc, sh, pn, pb], 1)).tobytes()
                pair.hmeta = json.dumps({"y0": int(fr["y0"]), "y1": int(fr["y1"]), "km2": float(CELL_KM2.get(HOLD["res"], 0)),
                                         "D0": None if np.isnan(fr["D0"]) else float(fr["D0"])})
            HOLD["sent"] = fr
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

    # ---- the click: a card of plain facts, as JSON the browser lays out ----------
    def _acode(a):
        """AlphaEarth year -> code: 0 not read, 1 no jump, 2 no data, else year - 2000."""
        if a is None:
            return 0
        a = int(a)
        return a - 2000 if a > 0 else (1 if a == -1 else 2)

    def _place_names(levels):
        out = []
        for lv in levels:
            nm = lv.get("name_en") or lv.get("name")
            if nm and nm not in out:
                out.append(nm)
        return out[:4]

    def _bld_card(p):
        """What is under the click among the footprints read for the view: a
        footprint, bare ground, or None when the footprints are not up."""
        b = HOLD["bld"]
        if b is None:
            return None
        lon, lat = p.get("lon"), p.get("lat")
        if lon is None or lat is None:
            return None
        px = b["psz"]
        c, r = int((lon - b["lon0"]) / px), int((b["lat0"] - lat) / px)
        arr, cover, rows = b["arr"], b["cover"], b["rows"]
        if not (0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]):
            return {"kind": "note", "title": "Outside the buildings read for this view. Pan a little and they follow."}
        k, w = int(cover[r, c]), int(arr[r, c])
        if k > 0:
            row = rows.slice(k - 1, 1).to_pylist()[0]
            ay = b.get("ayear")
            srcs = b.get("sources") or []
            ds = row.get("dataset")
            return {
                "kind": "footprint", "idx": k - 1, "id": row["id"], "dataset": ds,
                "src": srcs.index(ds) if ds in srcs else 99,
                "updated": (lambda u: u if u[:4].isdigit() and int(u[:4]) >= 2005 else None)(str(row.get("updated") or "")[:10]),
                "what": ", ".join(x for x in (row.get("subtype"), row.get("class")) if x) or None,
                "height": float(row["height"]) if row.get("height") is not None else None,
                "floors": int(row["num_floors"]) if row.get("num_floors") is not None else None,
                "npx": int(b["npx"][k - 1]), "nbuilt": int(b["built"][k - 1]),
                "wk": int(b["first"][k - 1]), "acode": _acode(ay[k - 1] if ay is not None else None),
                "asteps": [None if np.isnan(v) else float(v) for v in b["asteps"][:, k - 1]] if ay is not None and b.get("asteps") is not None and b["asteps"].shape[0] else None,
                "ayears": [int(y) for y in b.get("ayears", [])[1:]] if ay is not None else None,
                "aD0": None if ay is None or np.isnan(b.get("aD0", float("nan"))) else float(b["aD0"]),
            }
        return {"kind": "ground", "wk": w, "near": bool(b["near"][r, c]) if b.get("near") is not None else False}


    def _hex_card(p):
        """The clicked hexagon, as facts for the card: AlphaEarth's change year
        and year-to-year steps against the quiet level, how much it moved,
        and WSF's shares."""
        fr = HOLD["frame"]
        cellh = p.get("cell")
        if fr is None or not cellh or not HOLD["on"].get("hex"):
            return None
        cell = np.uint64(int(cellh, 16))
        ids = fr["cellid"]
        i = int(np.searchsorted(ids, cell))
        if i >= len(ids) or ids[i] != cell:
            HOLD["hit"] = None
            return {"kind": "note", "title": "That hexagon is not in the current view's frame."}
        HOLD["hit"] = int(cell)
        d = float(fr["disp"][i])
        lo, hi = fr.get("shift_lo", 0.0), fr.get("shift_hi", 1.0)
        steps = [None if np.isnan(v) else float(v) for v in fr["steps"][:, i]] if len(fr["step_years"]) else []
        return {
            "kind": "hex", "when": int(fr["when"][i]), "level": None if np.isnan(d) else float(np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)),
            "y0": int(fr["y0"]), "y1": int(fr["y1"]), "p_built": float(fr["p_built"][i]), "p_new": float(fr["p_new"][i]), "byear": int(fr["byear"][i]),
            "steps": steps, "step_years": [int(yb) for _, yb in fr["step_years"]], "D0": None if np.isnan(fr["D0"]) else float(fr["D0"]),
            "km2": float(CELL_KM2.get(HOLD["res"], 0)),
        }

    def _card_send(p, place=None):
        """Build and send the card for the pick p (kept, so it can be rebuilt
        when AlphaEarth lands or the place arrives)."""
        card = _hex_card(p) or _bld_card(p) or {}
        if not card:
            HOLD["card"] = None
            pair.card = ""
            return
        prev = HOLD.get("card") or {}
        card["place"] = place if place is not None else (prev.get("place") if prev.get("n") == p.get("n") else [])
        card["n"] = p.get("n")
        HOLD["card"], HOLD["card_pick"] = card, p
        pair.card = json.dumps(card)

    def _place_async(p, adm):
        n = p.get("n")

        async def _later():
            try:
                d = await asyncio.to_thread(division_at, p["lon"], p["lat"])
            except Exception:
                d = []
            if HOLD.get("pick_n") != n:
                return
            names = _place_names(d) or [x for x in (adm.get("county"), adm.get("region")) if x]
            _card_send(p, place=names)

        _spawn(_later())


    # ---- the footprints ----------------------------------------------------------
    def _cover_off():
        if HOLD["bld"] is None and HOLD["obox"] is None:
            return
        HOLD["bld"], HOLD["obox"] = None, None
        with pair.hold_sync():
            pair.polys, pair.battrs, pair.gap = b"", b"", b""
            pair.bmeta = "{}"

    def _bld_send():
        """The footprints' attributes for the browser, 4 bytes each: the WSF
        index (0 never, 1 standing by July 2016, k the half-year), the
        AlphaEarth code (0 not read, 1 no jump, 2 no data, else year - 2000),
        the source's index (by how common it is in the box) and the year its
        source last touched it (year - 2000). The browser colours them for the
        lens and the timeline year, so a scrub never comes back here."""
        b = HOLD["bld"]
        if b is None:
            return False
        from collections import Counter
        n = b["rows"].num_rows
        first = b["first"]
        wy = np.clip(first, 0, 20).astype(np.uint8)  # the WSF index itself: the browser knows its half-years
        ay = b.get("ayear")
        if ay is None:
            ac = np.zeros(n, np.uint8)
        else:
            ac = np.where(ay > 0, ay - 2000, np.where(ay == -1, 1, 2)).astype(np.uint8)
        ds = b["rows"]["dataset"].to_pylist()
        order = [k for k, _ in Counter(ds).most_common()]
        ix = {k: i for i, k in enumerate(order)}
        si = np.array([min(ix[d], 255) for d in ds], np.uint8).reshape(n)
        up = [str(u or "")[:4] for u in b["rows"]["updated"].to_pylist()]
        uy = np.array([int(u) - 2000 if u.isdigit() and 2000 < int(u) < 2100 else 0 for u in up], np.uint8).reshape(n)
        b["sources"] = order
        meta = {"sources": [o or "unnamed source" for o in order], "box": b["gap_box"], "gap_share": b["gap_share"],
                "aef": ay is not None, "n": int(n)}
        with pair.hold_sync():
            pair.battrs = np.ascontiguousarray(np.stack([wy, ac, si, uy], 1)).tobytes()
            pair.bmeta = json.dumps(meta)
        return True


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
                _bld_send()
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
            _bld_send()
            cp = HOLD.get("card_pick")
            if cp is not None and (HOLD.get("card") or {}).get("kind") == "footprint":
                _card_send(cp)
            _status()
        except Exception as e:
            HOLD["bld_status"] = (HOLD.get("bld_status") or "").split(" · AlphaEarth")[0] + f" · AlphaEarth failed: {type(e).__name__}: {e}"
            _status()
        finally:
            b["aef_busy"] = False

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
        # built-up ground with no footprint within 20 m (two pixels): the map
        # check's amber. The yards, lanes and roads between mapped buildings
        # are built-up to WSF too, so plain "no footprint on this pixel" paints
        # the whole town; two pixels off any footprint is ground where a
        # building may be missing from the map
        near = cover > 0
        grown = near.copy()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy or dx:
                    sh = np.zeros_like(near)
                    ys, yd = (slice(dy, None), slice(0, -dy)) if dy > 0 else ((slice(0, dy), slice(-dy, None)) if dy < 0 else (slice(None), slice(None)))
                    xs, xd = (slice(dx, None), slice(0, -dx)) if dx > 0 else ((slice(0, dx), slice(-dx, None)) if dx < 0 else (slice(None), slice(None)))
                    sh[ys, xs] = near[yd, xd]
                    grown |= sh
        gapm = built_m.reshape(arr.shape) & ~grown
        n_far_px = int(gapm.sum())

        def _gap_png():
            rgba = np.zeros(arr.shape + (4,), np.uint8)
            rgba[gapm] = (246, 183, 60, 200)
            buf = io.BytesIO()
            Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
            return buf.getvalue()

        gap_png = await cpu(_gap_png) if gapm.any() else b""
        h_, w_ = arr.shape
        HOLD["bld"] = {"rows": rows, "cover": cover, "arr": arr, "lon0": lon0, "lat0": lat0, "psz": psz,
                       "first": first, "npx": npx, "built": built, "box": box, "legend": None,
                       "gap_box": [lon0, lat0 - h_ * psz, lon0 + w_ * psz, lat0], "near": grown,
                       "gap_share": n_far_px / max(1, n_built_px)}
        HOLD["obox"] = box
        with pair.hold_sync():
            pair.polys = gj
            pair.gap = gap_png
        HOLD["bld_status"] = (
            f"Overture {OV_RELEASE} ({OV_RELEASE_HOW}): {n_fp:,} footprints"
            + (f" ({by_ds})" if by_ds else "")
            + f" · {n_never:,} of them WSF has never read as built-up"
            + f" · WSF built-up pixels {n_built_px:,}, {100 * n_gap_px / max(1, n_built_px):.0f}% under no footprint"
            + f" · read {t1 - t0:.1f} s, {len(gj) / 1e6:.1f} MB of GeoArrow, {time.time() - t1:.1f} s"
        )
        _bld_send()
        if HOLD.get("witness") == "aef":
            _spawn(_aef_for_bld(HOLD["bld"]))

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

    def _on_pick(change):
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        try:
            HOLD["pick_n"] = p.get("n")
            if p.get("close"):
                HOLD["hit"], HOLD["card"], HOLD["card_pick"] = None, None, None
                pair.card = ""
                _paint()
                return
            adm = p.get("admin") or {}
            _card_send(p, place=[x for x in (adm.get("county"), adm.get("region")) if x])
            if HOLD.get("card") and p.get("lon") is not None:
                _place_async(p, adm)
        except Exception as e:
            pair.card = json.dumps({"kind": "note", "title": f"click: {type(e).__name__}: {e}"})
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
        if act in ("s2", "s2scale", "aef", "fill", "labels", "layers"):
            HOLD["witness"] = c.get("witness") or HOLD.get("witness")
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
        if act == "hex":
            new = {k: bool(v) for k, v in (c.get("on") or {}).items()}
            a, b = int(c.get("y0", HOLD["y0"])), int(c.get("y1", HOLD["y1"]))
            if a in AEF_YEARS_ALL and b in AEF_YEARS_ALL and a < b:
                HOLD["y0"], HOLD["y1"] = a, b
            HOLD["on"] = new
            _cfg(layers=new, aef_from=HOLD["y0"], aef_to=HOLD["y1"])
            _request(force=True)
            return
        if act == "witness":
            # "dated by AlphaEarth" (or the card's "check with AlphaEarth"):
            # the nine years for the box, once per box
            HOLD["witness"] = c.get("witness") or "aef"
            b = HOLD["bld"]
            if b is not None and b.get("ayear") is None:
                _spawn(_aef_for_bld(b))
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
        _bld_send()
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
