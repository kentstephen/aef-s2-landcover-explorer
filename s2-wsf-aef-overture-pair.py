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
# ]
# ///


"""S2 x WSF x AEF: the ground as a picture beside one H3 fill, one camera.

Two maps in one widget. LEFT: Earth Genome's Sentinel-2 yearly mosaic (true
color, 2022-2025) as tiles the kernel renders from the COGs; it is never
covered. RIGHT: one opaque H3 fill per hexagon over the same camera, from two
sources folded onto the same cells:

  WSF Tracker (DLR / MindEarth, GeoZarr on source.coop): one int8 per 10 m
  pixel, the half-year the pixel first read as built-up (1 = by July 2016 ..
  20 = by January 2026, 0 = never). Native pixels are sampled on a stride
  that fits a budget (the pyramid is a min and cannot be folded; it draws the
  right pane below the hexagon zoom). Per cell: the share built-up by the end
  of the window, the share that became built-up INSIDE the window, and the
  year most of that new ground arrived.

  AlphaEarth (Google DeepMind, COG overviews on source.coop): the cell's mean
  64-vector per year; the displacement between the two ends of the window
  (1 - cos), and the first year inside it the fingerprint jumped past the
  view's quiet level, set by the cells WSF says did not grow.

The window (from year, to year) is the one control both sides read. The
build-year fill and the AEF change-year fill share one palette, so a hexagon
that agrees is the same color on both.

The fold is the H3 UDF inside DataFusion (repo rule): every raster's pixels
cross as one Dataset and the cell is the GROUP BY. Nothing is tessellated in
the kernel; the browser gets cell ids and rgba.

Overture's division lines and the county join were here (Stephen,
2026-09-02, at the New Capital: "we're not joining this massive data at
country or even county level, they're not helpful, i say remove"). The fold
is per hexagon; an admin line told it nothing. Gone with them: the gold lines,
the hover name, the county tables, key B. The notebook moved to its own repo
on 2026-09-20 (from kentstephen/zarr-grid-h3) and kept its file name; overture
in the name now stands for the place under a click, below.

What came back (Stephen, 2026-09-10) is the PLACE under a click, not the
lines and not a join: the ladder of Overture divisions holding the point,
locality up to country, each level named the way Overture names it and the
country's own word for it beside (city, town, prefecture, governorate). Two
Source Cooperative sources, the split the New England Landsat pair uses:
cboettig/overturemaps PMTiles (regions and counties, worldwide) range-read
by the browser, which names the region and county the instant the click
lands; fused/overture GeoParquet, one DuckDB point query off the loop for
the whole ladder, local_type included, 1 to 3 s warm. Both cover the world.

Run: uv run marimo edit s2-wsf-aef-overture-pair.py

Attribution: WSF Tracker (c) DLR and MindEarth, via source.coop
(mindearth/wsf, DOI 10.5281/zenodo.20424537). "The AlphaEarth Foundations
Satellite Embedding dataset is produced by Google and Google DeepMind."
(CC-BY 4.0.) Sentinel-2 yearly mosaics and Sentinel-2 L2A temporal mosaics
(CC-BY 4.0) by Earth Genome (Copernicus Sentinel data). Place search by Photon
(komoot), OpenStreetMap data (ODbL). Overture Maps divisions (ODbL) via
Source Cooperative: cboettig/overturemaps (PMTiles), fused/overture (GeoParquet).

TODO (Stephen, 2026-09-02): Overture BUILDINGS from the fused partition on
source.coop as the next layer, drawn only zoomed in (the bias-bounty tutorial
notebook already does that for a lot of buildings). Not started.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import asyncio
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
        duckdb,
        io,
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
    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/s2-wsf-aef-overture-pair/blob/main/s2-wsf-aef-overture-pair.py)
    <small>molab runs in the same region as the data and is the faster place to open this notebook.
    Locally: `uv run marimo edit s2-wsf-aef-overture-pair.py --sandbox`</small>

    # Settlement growth: Sentinel-2, WSF Tracker and AlphaEarth on one H3 grid

    This notebook places a true-colour Sentinel-2 mosaic beside a hexagon map
    built from two independent records of change, so that what the satellite
    shows can be read against what the models say. All three sources are
    read live from Source Cooperative; nothing is downloaded ahead of time.

    **Left pane**: the Sentinel-2 yearly mosaic (Earth Genome, 2022 to 2025),
    rendered by the kernel from the cloud-optimised GeoTIFFs. It is never
    covered by data layers. **Right pane**: one opaque H3 fill per hexagon over
    the same camera. The two panes share one view: pan or zoom either and both
    follow, and a hexagon hovered on one side is outlined on both.

    ## Sources

    - **WSF Tracker** (DLR and MindEarth). One value per 10 m pixel: the
      half-year the pixel was first observed as built-up, from July 2016 to
      January 2026. Per hexagon the notebook derives the share built-up by the
      end of the window, the share that became built-up inside it, and the
      year most of that new ground arrived.
    - **AlphaEarth Foundations** (Google DeepMind). A 64-dimensional embedding
      per pixel per year, 2017 to 2025. Per hexagon the notebook takes the
      mean embedding for each year and measures how far it moved between the
      two ends of the window (1 minus cosine similarity), together with the
      first year inside the window in which it jumped past a quiet level set
      by the hexagons WSF says did not grow.
    - **Overture Maps divisions**. Used only to name the place under a click,
      from locality up to country. See the click panel below.

    ## Controls

    - **S2** (left header): a slider over the mosaic years, also driven by the
      arrow keys or `[` and `]` after a click on either map. The `scale` slider
      is a gain on the picture. Where a yearly mosaic has a gap (for example
      2022 over Nusantara, clouded all year), the same year's temporal median
      composite fills it and the status line reports the share of pixels that
      came from it.
    - **FIND**: a place search (Photon, OpenStreetMap data). Results drop down
      as you type; Enter or a click flies both maps there.
    - **FILL** (keys `1` to `5`) selects what the right pane shows:
      **WSF built**, the record itself as a raster, one colour per year of
      first detection; **WSF grew**, the share of each hexagon that became
      built-up inside the window; **WSF build year**, the year most of that
      new ground arrived; **AEF changed**, how far the AlphaEarth embedding
      moved between the ends of the window; **AEF change year**, the first
      year inside the window the embedding jumped past the quiet level.
    - **WINDOW**: a two-handle slider over whole years 2017 to 2025, read by
      both sources (`-` and `=` move the start, `_` and `+` the end). The
      frame is rebuilt when a handle is released. WSF is folded once per view;
      only the AlphaEarth years not already held are fetched.
    - `L` toggles basemap labels, `F` enters full screen.

    ## Reading the map

    The two year fills share one palette. A hexagon that has the same colour
    under **WSF build year** and **AEF change year** is one where the
    settlement record and the embedding agree on the year of change.

    Hexagons appear from zoom 9 and refine to H3 resolution 12 (about three
    pixels of either source) beyond zoom 14.6. Below zoom 9 the right pane
    shows the WSF pyramid, the earliest built-up date under each pixel, and
    the left pane keeps the mosaic down to zoom 7.

    ## The click panel

    Clicking a hexagon writes its account below the map: what WSF recorded
    there and when, how far and when AlphaEarth saw the ground change, and
    the place it belongs to. The place is the ladder of Overture divisions
    that contain the point, smallest first, each labelled with its level and
    with the local term for that level where Overture provides one (city,
    town, prefecture, governorate). Region and county are answered at once
    from vector tiles in the browser; the full ladder follows from a point
    query against the Overture GeoParquet on Source Cooperative, typically
    within a few seconds once the kernel has warmed up.

    ## Method

    Every raster is folded onto H3 cells by the H3 UDF inside DataFusion: the
    pixels cross as one dataset and the cell is the GROUP BY. Nothing is
    tessellated in the kernel; the browser receives cell ids and colours.

    **How a hexagon gets its WSF numbers.** Each 10 m pixel carries one value,
    the half-year it first read as built-up. The fold counts the pixels under
    each hexagon by that value, so a hexagon is a small histogram of first
    detection dates. From it: the share built-up by the end of the window,
    the share whose first date falls inside the window (**WSF grew**), and
    the year holding most of those new pixels (**WSF build year**). A
    hexagon counts as having grown when at least 1% of its pixels are new
    inside the window; below that it is treated as quiet.

    **How a hexagon gets its AlphaEarth numbers.** AlphaEarth describes each
    10 m pixel, for each year, as a 64-number vector: a fingerprint of what
    the ground looked like that year. The fold averages those vectors over
    the pixels under each hexagon, giving one fingerprint per hexagon per
    year. Two fingerprints are compared by the angle between them: 1 minus
    cosine similarity, 0 when they point the same way, larger the more they
    differ. **AEF changed** is that distance between the first and last year
    of the window. For **AEF change year** the notebook walks the window one
    year at a time and looks for the first step whose distance clears a
    quiet level. The quiet level is set by the view itself: among the
    hexagons WSF says did not grow, 95% of their largest yearly steps fall
    below it, so a step above it is bigger than almost anything the
    unchanged ground does. The change year is the first year that happens.
    If a view has fewer than 30 quiet hexagons the level is not set and the
    fill stays unscored.

    ## Attribution

    WSF Tracker (c) DLR and MindEarth, via Source Cooperative (mindearth/wsf,
    DOI 10.5281/zenodo.20424537). The AlphaEarth Foundations Satellite
    Embedding dataset is produced by Google and Google DeepMind (CC BY 4.0).
    Sentinel-2 yearly and temporal mosaics by Earth Genome from Copernicus
    Sentinel data (CC BY 4.0). Overture Maps divisions (ODbL) via Source
    Cooperative (cboettig/overturemaps, fused/overture). Place search by
    Photon (komoot) over OpenStreetMap data (ODbL). Basemap by Carto.
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
    HEX_ZOOM = 9.0
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
    FILLS = ("built", "grew", "byear", "shift", "when")
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

    # ---- place: Overture Maps divisions, live from Source Cooperative --------
    # Two repositories, each for what it holds. cboettig/overturemaps (release
    # 2026-02-18.0) has regions and counties as PMTiles, the world over: the
    # browser range-reads the tiles it needs and answers a click's region and
    # county from them. fused/overture (release 2026-05-20-0) has the whole
    # divisions theme as geo-partitioned GeoParquet, localities and
    # local_type included, which the PMTiles are not: the ladder under a click
    # is one DuckDB point query against it. The bucket is open: the s3 path
    # goes straight to the opendata bucket, no secret.
    ADMIN_PM = "https://data.source.coop/cboettig/overturemaps/2026-02-18.0"
    ADMIN_PQ = "s3://us-west-2.opendata.source.coop/fused/overture/2026-05-20-0/theme=divisions"

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
        ADMIN_PM,
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
    ctx,
    io,
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
    _store = S3Store(WSF_BUCKET, region="us-west-2", skip_signature=True, prefix=WSF_PREFIX)
    _root = zarr.open_group(ObjectStore(_store, read_only=True), mode="r")
    _arr = {k: _root[str(k)]["wsf_tracker"] for k in range(WSF_LEVELS)}
    _win = {}
    _sem = asyncio.Semaphore(6)
    _fold_lock = asyncio.Lock()
    _png_cache = {}

    def idx_year(k):
        """The calendar year index k books its built-up date in (1 -> 2016)."""
        return 2016 + (int(k) - 1) // 2

    def idx_h1(y):
        return 2 * (int(y) - 2016) + 1

    def idx_h2(y):
        return 2 * (int(y) - 2016) + 2

    WSF_DATE = ["never"] + [f"{2016 + (k - 1) // 2 + (k % 2 == 0)}-{'01' if k % 2 == 0 else '07'}-01" for k in range(1, WSF_NIDX + 1)]

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
            _png_cache[key] = None
            return None
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        _png_cache[key] = buf.getvalue()
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
        LON, LAT = np.meshgrid(lon, lat)
        async with _fold_lock:
            try:
                ctx.deregister_table("wsf")
            except Exception:
                pass
            ctx.from_dataset(
                "wsf",
                xr.Dataset(
                    {"idx": (("y", "x"), arr.astype(np.int16)), "lat": (("y", "x"), LAT), "lon": (("y", "x"), LON)},
                    coords={"y": np.arange(h), "x": np.arange(w)},
                ),
                chunks={"y": 512},
            )
            out = ctx.sql(f"""
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell,
                       count(*) AS npx,
                       sum(CASE WHEN idx > 0 THEN 1 ELSE 0 END) AS built,
                       {_CNT}
                FROM wsf
                WHERE lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                GROUP BY cell
            """).to_arrow_table()
        return out, (
            f"WSF {w:,}x{h:,} samples (stride {stride}, {10 * stride} m) read {tr - t0:.1f} s · fold {out.num_rows:,} {time.time() - tr:.1f} s"
        )

    return WSF_DATE, idx_h1, idx_h2, wsf_fold, wsf_tile_png


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
    Transformer,
    Window,
    asyncio,
    ctx,
    duckdb,
    np,
    os,
    pq,
    time,
    xr,
):
    # ---- AlphaEarth: the COG overviews (mosaic past res 10), one fold per year --
    # `aef_fold(box, res, year)` for any year in AEF_YEARS_ALL (2017..2025, the
    # whole run; the window control picks from them); each year has its own COG index
    # slice (cached as parquet under tmp) and its own mosaic time index.
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True)
    _mstore = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True, prefix=AEF_PREFIX)
    _ds = xr.open_zarr(ObjectStore(_mstore, read_only=True), chunks=None, consolidated=False)
    _ti = {y: int(np.where(_ds.time.values == y)[0][0]) for y in AEF_YEARS_ALL}

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
        a = np.asarray(np.ma.filled(ra.as_masked(), AEF_NODATA)).reshape(64, r1 - r0, c1 - c0)
        xs = t.c + (np.arange(c0, c1) + 0.5) * sx
        ys = t.f + (np.arange(r0, r1) + 0.5) * sy
        X, Y = np.meshgrid(xs, ys)
        lon, lat = inv.transform(X, Y)
        return a, lon, lat

    _DEQ = ", ".join(f"avg(signum(e{i:02d}) * power(e{i:02d} / 127.5, 2)) AS e{i:02d}" for i in range(64))
    _fold_lock = asyncio.Lock()

    async def _fold_rows(res, box, cols, lat, lon):
        W_, S_, E_, N_ = box
        ds1 = xr.Dataset(
            {f"e{i:02d}": (("i",), cols[i]) for i in range(64)} | {"lat": (("i",), lat), "lon": (("i",), lon)},
            coords={"i": np.arange(lat.size)},
        )
        async with _fold_lock:
            try:
                ctx.deregister_table("aef")
            except Exception:
                pass
            ctx.from_dataset("aef", ds1, chunks={"i": 262_144})
            return ctx.sql(f"""
                SELECT h3_latlng_to_cell(lat, lon, CAST({res} AS INT)) AS cell, count(*) AS naef, {_DEQ}
                FROM aef
                WHERE e00 != {AEF_NODATA}
                  AND lon >= {W_} AND lon < {E_} AND lat >= {S_} AND lat < {N_}
                GROUP BY cell
            """).to_arrow_table()

    async def aef_fold(box, res, year):
        """Mean AlphaEarth vector per res cell over the box for one year.
        Returns (arrow table or None, stats)."""
        t0 = time.time()
        W_, S_, E_, N_ = box
        if res >= MOSAIC_MIN_RES:
            x0, x1 = int((W_ - AEF_X0) / AEF_RES), int((E_ - AEF_X0) / AEF_RES)
            y0, y1 = int((AEF_Y0 - N_) / AEF_RES), int((AEF_Y0 - S_) / AEF_RES)
            loop = asyncio.get_running_loop()
            ti = _ti[year]
            emb = await loop.run_in_executor(
                None, lambda: _ds.embeddings.isel(time=ti, y=slice(y0, y1), x=slice(x0, x1)).values
            )
            lat = AEF_Y0 - (np.arange(y0, y1) + 0.5) * AEF_RES
            lon = AEF_X0 + (np.arange(x0, x1) + 0.5) * AEF_RES
            LON, LAT = np.meshgrid(lon, lat)
            t1 = time.time()
            out = await _fold_rows(res, box, emb.reshape(64, -1), LAT.ravel(), LON.ravel())
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
        cols = np.concatenate([p[0].reshape(64, -1) for p in parts], axis=1)
        lon = np.concatenate([p[1].ravel() for p in parts])
        lat = np.concatenate([p[2].ravel() for p in parts])
        t1 = time.time()
        out = await _fold_rows(res, box, cols, lat, lon)
        return out, (
            f"AEF {year} ov{li} ({10 * 2 ** (li + 1)} m) {len(parts)} files {cols.shape[1] / 1e6:.2f} Mpx "
            f"{t1 - t0:.1f} s · fold {out.num_rows:,} {time.time() - t1:.1f} s"
        )

    return (aef_fold,)


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
    Window,
    asyncio,
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
    _store = S3Store("us-west-2.opendata.source.coop", region="us-west-2", skip_signature=True)
    _R = 6378137.0
    _items = {}  # item id -> {tci: path, bbox}
    _boxes = {}  # (year, rounded box) -> item ids
    _open = {}
    _sem = asyncio.Semaphore(32)
    _png = {}  # (year, z, x, y, scale) -> PNG bytes or None
    _arr = {}  # (year, z, x, y) -> the composited RGBA tile before the gain, or None
    _gain = {"v": float(S2_SCALE0)}  # the strip's `scale`
    _tstat = {"served": 0, "blank": 0, "ms": 0.0}
    _fill = {}  # year -> [pixels painted by the fill collection, pixels painted]

    def _encode(key, out):
        """The gain (the header's `scale`) on the composited bytes, then PNG."""
        g = _gain["v"]
        rgba = out if g == 1.0 else np.concatenate(
            [np.clip(out[..., :3].astype(np.float32) * g, 0, 255).astype(np.uint8), out[..., 3:]], axis=2)
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgba), mode="RGBA").save(buf, format="PNG")
        _png[key] = buf.getvalue()
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
            return _encode(key, out) if out is not None else None
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
        out = np.zeros((T, T, 4), np.uint8)
        li = min(S2_TCI_MAX_Z - z, S2_TCI_MAX_Z - S2_PYRAMID_Z)  # L5 (306 m) below z9: decimated
        for iid in ids:
            g = await _get(_items[iid]["tci"])
            if g is None:
                continue
            lv = [g, *g.overviews][li]
            L, _B, R_, Tt = g.bounds
            H, W = lv.shape
            px = (R_ - L) / W
            c0, c1 = max(0, int(math.floor((tx0 - L) / px))), min(W, int(math.ceil((tx0 + T * tpx - L) / px)))
            r0, r1 = max(0, int(math.floor((Tt - ty1) / px))), min(H, int(math.ceil((Tt - (ty1 - T * tpx)) / px)))
            if c1 <= c0 or r1 <= r0:
                continue
            async with _sem:
                ra = await lv.read(window=Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0))
            a = np.asarray(np.ma.filled(ra.as_masked(), 0)).reshape(-1, r1 - r0, c1 - c0)[:3]
            cols = np.floor((xs - (L + c0 * px)) / px).astype(np.int64)
            rows = np.floor(((Tt - r0 * px) - ys) / px).astype(np.int64)
            okc, okr = (cols >= 0) & (cols < c1 - c0), (rows >= 0) & (rows < r1 - r0)
            rgb = a[:, np.clip(rows, 0, r1 - r0 - 1)[:, None], np.clip(cols, 0, c1 - c0 - 1)[None, :]].transpose(1, 2, 0)
            valid = okr[:, None] & okc[None, :] & (rgb.sum(2) > 0) & (out[..., 3] == 0)
            out[valid, :3] = rgb[valid]
            out[valid, 3] = 255
            n_new = int(valid.sum())
            fy = _fill.setdefault(year, [0, 0])
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
        png = _encode(key, out)
        _tstat["served"] += 1
        _tstat["ms"] += 1000 * (time.time() - t0)
        return png

    def s2_set_scale(v):
        """The header's `scale`: the gain the next S2 tiles are encoded with.
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
                items = [{"name": "built-up by July 2016", "hex": "#%02x%02x%02x" % YEAR_RGB[2016]}]
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
    class PairMap(anywidget.AnyWidget):
        """Two maplibre maps in a row, one camera. LEFT: the S2 mosaic as tiles the
        kernel renders (custom messages, PNG bytes back), keyed by year. RIGHT:
        an H3HexagonLayer (highPrecision) from cell ids + rgba, or the WSF
        pyramid as tiles below the hexagon zoom. Hover on either pane: h3-js
        cell at the frame's res, its ring drawn on BOTH panes.

        Kernel -> browser: `cells` (uint64 LE), `colors` (rgba u8), `config`
        (JSON), `status` / `panel` / `legend` (strings for the strip).
        Browser -> kernel: `view` (JSON lon/lat/zoom + the pane's w/h on every
        moveend), `pick` (JSON: the clicked cell as hex, or null), `ctl`
        (JSON: s2 year, s2 scale, the window, fill, labels). The Photon
        geocoder on the left card is browser-only: it asks photon.komoot.io
        as you type, lists the hits, and flies the left map; the camera
        sync and moveend do the rest."""

        cells = traitlets.Bytes(b"").tag(sync=True)
        colors = traitlets.Bytes(b"").tag(sync=True)
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
        import {latLngToCell, getResolution, cellToBoundary} from "https://esm.sh/h3-js@4.5.0";
        import {Protocol as PMProtocol} from "https://esm.sh/pmtiles@4.5.0";
        // the Overture division tiles come straight off the Source Cooperative
        // bucket by range request: one protocol handler, shared by both maps
        maplibregl.addProtocol("pmtiles", new PMProtocol().tile);

        const STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

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
          const font = "font:12px ui-sans-serif,system-ui,sans-serif";
          const mono = "font:11px ui-monospace,Menlo,monospace";
          const root = document.createElement("div");
          root.className = "sp-root";
          root.style.cssText = "width:100%;background:#fff;color:#222;" + font;
          const row = document.createElement("div");
          row.style.cssText = "display:flex;gap:4px;width:100%";
          const mkPane = (side) => {
            const pane = document.createElement("div");
            pane.className = "sp-pane sp-" + side;
            pane.style.cssText = "position:relative;flex:1 1 0;min-width:0;height:" + (cfg.height || 720) + "px;background:#f4f2ee";
            const mapEl = document.createElement("div");
            mapEl.className = "sp-map";
            mapEl.style.cssText = "position:absolute;inset:0";
            const head = document.createElement("div");
            head.className = "sp-head";
            // a small card, two rows at most, never into maplibre's control column
            head.style.cssText = "position:absolute;left:8px;top:8px;z-index:5;display:flex;flex-direction:column;gap:.3rem;align-items:flex-start;" +
              "max-width:calc(100% - 72px);background:rgba(255,255,255,.94);color:#1d1d1b;padding:4px 8px;border-radius:6px;" +
              "box-shadow:0 1px 3px rgba(0,0,0,.18);white-space:nowrap;font-variant-numeric:tabular-nums";
            pane.append(mapEl, head);
            return {pane, mapEl, head};
          };
          const L = mkPane("left"), R = mkPane("right");
          row.append(L.pane, R.pane);
          const strip = document.createElement("div");
          strip.style.cssText = "display:flex;flex-direction:column;gap:.25rem;padding:.35rem .4rem;background:#fff;color:#222";  // rewritten by paneHeight (stripCss) in full screen
          const status = document.createElement("div");
          status.className = "sp-status";
          status.style.cssText = "font:14px ui-sans-serif,system-ui,sans-serif;color:#444;white-space:pre-wrap";
          const legend = document.createElement("div");
          legend.className = "sp-legend";
          legend.style.cssText = "display:flex;flex-wrap:wrap;gap:.3rem .9rem;align-items:center;font-size:14px";
          const panel = document.createElement("div");
          panel.className = "sp-panel";
          panel.style.cssText = "font-size:14px";
          strip.append(legend, panel, status);
          status.hidden = !!cfg.minimal;  // STRIP_MINIMAL: see say()
          root.append(row, strip);
          el.append(css, root);

          // ---- the controls: the pane headers ------------------------------
          const ACCENT = "#2a5db0";
          const btnCss = font + ";padding:.15rem .55rem;border:0;background:transparent;color:#1d1d1b;cursor:pointer;line-height:1.4;font-variant-numeric:tabular-nums";
          const onCss = (b, on) => { b.style.background = on ? ACCENT : "transparent"; b.style.color = on ? "#fff" : "#1d1d1b"; };
          let s2y = cfg.s2_year, fill = cfg.fill || "built", labelsOn = cfg.labels !== false;
          let s2scale = Number(cfg.s2_scale) || 1;  // the S2 mosaic's gain
          let y0 = cfg.aef_from, y1 = cfg.aef_to;
          const send = (act) => {
            model.set("ctl", JSON.stringify({act, s2y, s2scale, fill, y0, y1, labels: labelsOn, n: Date.now()}));
            model.save_changes();
          };
          // an eyebrow label + a segmented control (joined buttons, one border)
          const mkGroup = (head, title, values, get, set, act, cls, isOn) => {
            const wrap = document.createElement("span");
            wrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
            const lab = document.createElement("span"); lab.textContent = title;
            lab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
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
            rowOf(head).appendChild(wrap);
            const style = () => btns.forEach((b) => onCss(b, isOn ? isOn(b) : b.dataset.value === String(get())));
            style();
            return style;
          };
          // the card is a column of shrink-wrapped rows; groups land in the last row
          const newRow = (head) => { const r = document.createElement("span"); r.style.cssText = "display:inline-flex;gap:.6rem;align-items:center"; head.appendChild(r); return r; };
          const rowOf = (head) => head.lastElementChild || newRow(head);
          const rowBreak = (head) => { newRow(head); };
          // S2 year: a stepped one-handle slider, built below once the CSS is in
          // (Stephen, 2026-09-03: "change it from buttons to a slider... as the
          // cursor moves, it's cheap enough to load the data"). `styleS2` is
          // assigned there; the [ ] keys and the cfg echo call it.
          let styleS2 = () => {};
          const fills = (cfg.fills || []).map((f) => ({value: f[0], label: f[1], title: f[2]}));
          // FILL has the first row, the window slider the second
          const styleFill = mkGroup(R.head, "fill", fills, () => fill, (v) => { fill = v; }, "fill", "sp-fill");
          // the AlphaEarth window: a stepped two-handle slider (Stephen,
          // 2026-09-01: "simple and intuitive... a stepped two way slider").
          // Two range inputs on one track; the handle you grab moves, the
          // other stays, and they never cross. The kernel hears the release
          // (change), not every notch (input), so a drag is one fold.
          rowBreak(R.head);
          const aefYears = cfg.aef_years || [];
          const sty = document.createElement("style");
          sty.textContent = [
            ".sp-range{position:relative;width:300px;height:30px}",
            ".sp-range input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}",
            ".sp-range input:focus{outline:none}",
            ".sp-range input::-webkit-slider-runnable-track{background:none;height:22px}",
            ".sp-range input::-moz-range-track{background:none;height:22px}",
            ".sp-range input::-webkit-slider-thumb{pointer-events:auto;-webkit-appearance:none;appearance:none;width:16px;height:16px;margin-top:3px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-range input::-moz-range-thumb{pointer-events:auto;width:12px;height:12px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-range .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:rgba(29,29,27,.22);border-radius:2px}",
            ".sp-range .spn{position:absolute;top:9px;height:4px;background:#2a5db0;border-radius:2px}",
            ".sp-range .tks{position:absolute;left:8px;right:8px;top:19px;display:flex;justify-content:space-between;font-size:9px;color:#6b6b68;line-height:1}",
            ".sp-range .tks span{width:0;display:flex;justify-content:center}",
            // the S2 scale: one handle on the same track
            ".sp-scale{position:relative;width:120px;height:22px}",
            ".sp-scale input{position:absolute;left:0;top:0;width:100%;height:22px;margin:0;background:none;-webkit-appearance:none;appearance:none}",
            ".sp-scale input:focus{outline:none}",
            ".sp-scale input::-webkit-slider-runnable-track{background:none;height:22px}",
            ".sp-scale input::-moz-range-track{background:none;height:22px}",
            ".sp-scale input::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;margin-top:3px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-scale input::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:#2a5db0;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.35);cursor:grab}",
            ".sp-scale .trk{position:absolute;left:8px;right:8px;top:9px;height:4px;background:rgba(29,29,27,.22);border-radius:2px}",
            ".sp-scale .spn{position:absolute;left:8px;top:9px;height:4px;background:#2a5db0;border-radius:2px}",
            // the S2 year: the scale slider's handle and track, the range's ticks
            ".sp-year{height:30px}",
            ".sp-year .tks{position:absolute;left:8px;right:8px;top:19px;display:flex;justify-content:space-between;font-size:9px;color:#6b6b68;line-height:1}",
            ".sp-year .tks span{width:0;display:flex;justify-content:center}",
          ].join("\n");
          el.appendChild(sty);
          const aefWrap = document.createElement("span");
          aefWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const aefLab = document.createElement("span"); aefLab.textContent = "window";
          aefLab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          const rng = document.createElement("span"); rng.className = "sp-range";
          const trk = document.createElement("span"); trk.className = "trk";
          const spn = document.createElement("span"); spn.className = "spn";
          const tks = document.createElement("span"); tks.className = "tks";
          for (const y of aefYears) { const t = document.createElement("span"); const i = document.createElement("i"); i.style.fontStyle = "normal"; i.textContent = String(y); t.appendChild(i); tks.appendChild(t); }
          const mkRange = () => {
            const r = document.createElement("input"); r.type = "range"; r.min = 0; r.max = Math.max(0, aefYears.length - 1); r.step = 1;
            r.title = "the window both WSF and AlphaEarth read: drag either end (whole years); release to rebuild"; return r;
          };
          const rFrom = mkRange(), rTo = mkRange();
          const aefTxt = document.createElement("span");
          aefTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:6.5em";
          rng.append(trk, spn, tks, rFrom, rTo);
          aefWrap.append(aefLab, rng, aefTxt);
          rowOf(R.head).appendChild(aefWrap);
          const styleAef = () => {
            const i0 = Math.max(0, aefYears.indexOf(y0)), i1 = Math.max(0, aefYears.indexOf(y1)), n = Math.max(1, aefYears.length - 1);
            rFrom.value = i0; rTo.value = i1;
            // the higher handle sits on top so it can always be grabbed at the far end
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
          const aefRelease = () => { if (y0 !== aefSent[0] || y1 !== aefSent[1]) { aefSent = [y0, y1]; send("aef"); } };
          let aefSent = [y0, y1];
          rFrom.addEventListener("change", aefRelease);
          rTo.addEventListener("change", aefRelease);
          setTimeout(styleAef, 0);
          try { new ResizeObserver(styleAef).observe(rng); } catch (e) {}  // the lit span is in px: redo it when the track gets its width
          // the S2 mosaic's `scale`, beside the S2 years (Stephen, 2026-09-02:
          // "a scale slider for s2 next to the years on the left map (brightness
          // for observation)"): a gain on the TCI bytes, applied in the kernel
          // where the tiles are composited, so a release drops the S2 tiles and
          // asks deck for them again. The kernel hears the release (change),
          // not every notch (input); the readout follows the drag.
          const s2Years = cfg.s2_years || [];
          const yrWrap = document.createElement("span");
          yrWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const yrLab = document.createElement("span"); yrLab.textContent = "S2";
          yrLab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          const yr = document.createElement("span"); yr.className = "sp-scale sp-year";
          const yrTrk = document.createElement("span"); yrTrk.className = "trk";
          const yrSpn = document.createElement("span"); yrSpn.className = "spn";
          const yrTks = document.createElement("span"); yrTks.className = "tks";
          for (const y of s2Years) { const t = document.createElement("span"); const i = document.createElement("i"); i.style.fontStyle = "normal"; i.textContent = String(y); t.appendChild(i); yrTks.appendChild(t); }
          const yri = document.createElement("input"); yri.type = "range"; yri.min = 0; yri.max = Math.max(0, s2Years.length - 1); yri.step = 1;
          yri.title = "which Sentinel-2 yearly mosaic is drawn; the tiles follow the drag (arrow keys, or [ and ])";
          const yrTxt = document.createElement("span");
          yrTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.6em";
          yr.append(yrTrk, yrSpn, yrTks, yri);
          yrWrap.append(yrLab, yr, yrTxt);
          rowOf(L.head).appendChild(yrWrap);
          styleS2 = () => {
            const i = Math.max(0, s2Years.indexOf(s2y)), n = Math.max(1, s2Years.length - 1);
            yri.value = i;
            yrSpn.style.width = Math.max(0, (yr.clientWidth - 16) * i / n) + "px";
            yrTxt.textContent = String(s2y);
          };
          // the tiles follow the drag: a year is one cfg echo and a layer swap,
          // and a year already seen is served from the kernel's PNG cache, so
          // every notch can go, debounced like the scale below
          let yrSent = s2y, yrTimer = null;
          const yrRelease = () => { if (yrTimer) { clearTimeout(yrTimer); yrTimer = null; } if (s2y !== yrSent) { yrSent = s2y; send("s2"); } };
          yri.addEventListener("input", () => { s2y = s2Years[Number(yri.value)]; styleS2(); if (yrTimer) clearTimeout(yrTimer); yrTimer = setTimeout(yrRelease, 150); });
          yri.addEventListener("change", yrRelease);
          setTimeout(styleS2, 0);
          try { new ResizeObserver(styleS2).observe(yr); } catch (e) {}
          const SC_MIN = 0.2, SC_MAX = 3;
          const scWrap = document.createElement("span");
          scWrap.style.cssText = "display:inline-flex;align-items:center;gap:.4rem";
          const scLab = document.createElement("span"); scLab.textContent = "scale";
          scLab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          const scr = document.createElement("span"); scr.className = "sp-scale";
          const scTrk = document.createElement("span"); scTrk.className = "trk";
          const scSpn = document.createElement("span"); scSpn.className = "spn";
          const sc = document.createElement("input"); sc.type = "range"; sc.min = SC_MIN; sc.max = SC_MAX; sc.step = 0.1;
          sc.title = "brightness of the Sentinel-2 mosaic (a gain on the TCI bytes, the tiles re-served as you drag; double-click for 1.0)";
          const scTxt = document.createElement("span");
          scTxt.style.cssText = "font-variant-numeric:tabular-nums;min-width:2.4em";
          scr.append(scTrk, scSpn, sc);
          scWrap.append(scLab, scr, scTxt);
          rowOf(L.head).appendChild(scWrap);
          const styleSc = () => {
            sc.value = s2scale;
            scSpn.style.width = Math.max(0, (scr.clientWidth - 16) * (s2scale - SC_MIN) / (SC_MAX - SC_MIN)) + "px";
            scTxt.textContent = s2scale.toFixed(1) + "\u00d7";
          };
          let scSent = s2scale, scTimer = null;
          const scRelease = () => { if (scTimer) { clearTimeout(scTimer); scTimer = null; } if (s2scale !== scSent) { scSent = s2scale; send("s2scale"); } };
          // LIVE while dragging: a re-serve is cheap (the composited tile is kept
          // in the kernel pre-gain; a scale change is a re-encode, no read), so the
          // map follows the drag, debounced so a fast sweep is a few re-serves
          // rather than one per notch (Stephen, 2026-09-02: "let the results show
          // live on the map as toggled and add debounce for rapid movement")
          const SC_DEBOUNCE_MS = 150;
          sc.addEventListener("input", () => { s2scale = Number(sc.value); styleSc(); if (scTimer) clearTimeout(scTimer); scTimer = setTimeout(scRelease, SC_DEBOUNCE_MS); });
          sc.addEventListener("change", scRelease);
          // double-click the slider: back to 1.0 (Stephen, 2026-09-02)
          scr.addEventListener("dblclick", (e) => { e.preventDefault(); s2scale = 1; styleSc(); scRelease(); });
          setTimeout(styleSc, 0);
          try { new ResizeObserver(styleSc).observe(scr); } catch (e) {}
          // the geocoder: Photon (the x-sql deck notebook's field), asked from the
          // browser as you type, debounced, biased on the camera; the hits in a
          // dropdown under the field so the place is seen before the flight
          // (Stephen, 2026-09-02: "the dropdown auto complete so i can see where
          // it's gonna take me"). Enter or a click flies the left map; the camera
          // sync carries the right, and moveend sends the view like any pan. No
          // kernel round trip: no ctl, no comm, nothing for the fold to see early.
          rowBreak(L.head);
          const PHOTON = "https://photon.komoot.io/api/";
          const gcWrap = document.createElement("span");
          gcWrap.style.cssText = "position:relative;display:inline-flex;align-items:center;gap:.4rem";
          const gcLab = document.createElement("span"); gcLab.textContent = "find";
          gcLab.style.cssText = "font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#6b6b68";
          const gc = document.createElement("input");
          gc.type = "search"; gc.placeholder = "a place\u2026"; gc.autocomplete = "off"; gc.spellcheck = false;
          gc.title = "Photon geocoder: type, pick a hit (arrows, Enter, or click) and both maps fly there";
          gc.style.cssText = "width:16rem;" + font + ";padding:.15rem .45rem;border:1px solid rgba(29,29,27,.28);border-radius:5px;background:#fff;color:#1d1d1b";
          const gcList = document.createElement("div");
          gcList.className = "sp-hits";
          gcList.style.cssText = "position:absolute;left:0;top:calc(100% + 4px);z-index:9;display:none;min-width:100%;max-width:26rem;" +
            "background:#fff;color:#1d1d1b;border:1px solid rgba(29,29,27,.28);border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.18);overflow:hidden";
          gcWrap.append(gcLab, gc, gcList);
          rowOf(L.head).appendChild(gcWrap);
          let gcHits = [], gcSel = -1, gcTimer = null, gcSeq = 0;
          const GC_DEBOUNCE_MS = 250, GC_LIMIT = 6;
          const hitName = (f) => {
            const p = f.properties || {};
            const parts = [p.name, p.street && !p.name ? p.street : null, p.city && p.city !== p.name ? p.city : null,
              p.county && p.county !== p.city && p.county !== p.name ? p.county : null, p.state, p.country];
            return parts.filter((x) => x).join(", ");
          };
          const hitKind = (f) => { const p = f.properties || {}; return [p.osm_value, p.type].filter((x) => x && x !== "yes").join(" \u00b7 "); };
          const gcHide = () => { gcList.style.display = "none"; gcList.replaceChildren(); gcSel = -1; };
          const gcShow = () => {
            gcList.replaceChildren();
            if (!gcHits.length) { gcHide(); return; }
            gcHits.forEach((f, i) => {
              const row = document.createElement("div");
              row.style.cssText = "padding:.3rem .55rem;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.3;" +
                (i === gcSel ? "background:" + ACCENT + ";color:#fff" : "");
              const nm = document.createElement("div"); nm.textContent = hitName(f);
              const kd = document.createElement("div"); kd.textContent = hitKind(f);
              kd.style.cssText = "font-size:11px;opacity:" + (i === gcSel ? ".85" : ".6");
              row.append(nm, kd);
              row.onmousedown = (e) => { e.preventDefault(); gcFly(f); };  // before the input's blur
              row.onmouseenter = () => { gcSel = i; gcShow(); };
              gcList.appendChild(row);
            });
            gcList.style.display = "block";
          };
          const gcAsk = async () => {
            const q = gc.value.trim();
            if (q.length < 2) { gcHits = []; gcHide(); return; }
            const seq = ++gcSeq;
            const params = new URLSearchParams({q, limit: String(GC_LIMIT), lang: "en"});
            if (mapL) { const c = mapL.getCenter(); params.set("lon", c.lng.toFixed(4)); params.set("lat", c.lat.toFixed(4)); }
            try {
              const r = await fetch(PHOTON + "?" + params.toString());
              const data = await r.json();
              if (seq !== gcSeq) return;  // a later keystroke answered first
              gcHits = (data.features || []).filter((f) => f.geometry && f.geometry.coordinates);
              gcSel = gcHits.length ? 0 : -1;
              gcShow();
            } catch (e) { if (seq === gcSeq) say("search: " + e.message); }
          };
          const gcFly = (f) => {
            const [lon, lat] = f.geometry.coordinates;
            const ext = (f.properties || {}).extent;  // [minLon, maxLat, maxLon, minLat]
            const w = (L.mapEl.clientWidth || 700);
            let zoom = 10;
            if (ext && ext.length === 4) {
              const span = Math.max(Math.abs(ext[2] - ext[0]), Math.abs(ext[1] - ext[3]) * 2, 0.01);
              zoom = Math.log2(360 * (w / 512) / span) - 0.3;
            }
            zoom = Math.max(3.5, Math.min(14, zoom));
            gc.value = hitName(f); gcHits = []; gcHide(); gc.blur();
            if (mapL) mapL.flyTo({center: [lon, lat], zoom, duration: 2000});
            say("\u2192 " + hitName(f) + " \u00b7 zoom " + zoom.toFixed(1));
          };
          gc.addEventListener("input", () => { if (gcTimer) clearTimeout(gcTimer); gcTimer = setTimeout(gcAsk, GC_DEBOUNCE_MS); });
          gc.addEventListener("focus", () => { if (gcHits.length) gcShow(); });
          gc.addEventListener("blur", () => { setTimeout(gcHide, 120); });
          gc.addEventListener("keydown", (e) => {
            e.stopPropagation();  // the map's own keys (1-5, [ ], ...) stay out of the field
            if (e.key === "ArrowDown" && gcHits.length) { gcSel = (gcSel + 1) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "ArrowUp" && gcHits.length) { gcSel = (gcSel - 1 + gcHits.length) % gcHits.length; gcShow(); e.preventDefault(); }
            else if (e.key === "Enter") {
              e.preventDefault();
              if (gcHits.length) gcFly(gcHits[Math.max(0, gcSel)]);
              else { if (gcTimer) clearTimeout(gcTimer); gcAsk().then(() => { if (gcHits.length) gcFly(gcHits[0]); else say("no match: " + gc.value.trim()); }); }
            }
            else if (e.key === "Escape") { gcHide(); gc.blur(); }
          });
          // full screen. The widget lives in a shadow root, so the document's
          // fullscreenElement is the shadow HOST, never our root: walk down the
          // shadow roots before comparing (the deck notebook's realFs), or the
          // full-screen layout never applies and the strip is clipped at the
          // viewport (Stephen, 2026-09-01: "we still have that problem with the
          // strip"). In full screen the panes take the whole viewport and the
          // strip floats over their foot, translucent, scrolling inside itself
          // past 45vh, the way the deck notebooks' HUD does.
          const isFull = () => {
            let fe = document.fullscreenElement;
            while (fe && fe.shadowRoot && fe.shadowRoot.fullscreenElement) fe = fe.shadowRoot.fullscreenElement;
            return fe === root;
          };
          const stripCss = "display:flex;flex-direction:column;gap:.25rem;padding:.35rem .4rem;background:#fff;color:#222";
          const paneHeight = () => {
            const full = isFull();
            root.style.position = full ? "relative" : "";
            root.style.height = full ? "100vh" : "";
            root.style.boxSizing = "border-box";
            for (const pn of [L.pane, R.pane]) pn.style.height = full ? "100vh" : (cfg.height || 720) + "px";
            strip.style.cssText = full
              ? stripCss + ";position:absolute;left:0;right:0;bottom:0;z-index:30;background:rgba(255,255,255,.94);max-height:45vh;overflow-y:auto;box-sizing:border-box;box-shadow:0 -1px 4px rgba(0,0,0,.18)"
              : stripCss;
            styleAef();
          };
          const toggleFull = () => {
            if (isFull()) document.exitFullscreen();
            else root.requestFullscreen().catch((e) => say("fullscreen: " + e.message));
          };
          document.addEventListener("fullscreenchange", () => { setTimeout(paneHeight, 30); });
          window.addEventListener("resize", () => { paneHeight(); });
          const hint = document.createElement("div");
          hint.style.cssText = mono + ";opacity:.55";
          hint.textContent = "keys: [ ] S2 year · ; ' S2 scale · 1-5 fill · - = window from · _ + window to · L labels · F full screen · click a hexagon for its row";
          hint.style.color = "#666";
          strip.appendChild(hint);
          hint.hidden = !!cfg.minimal;
          const step = (arr, cur, d) => { const i = arr.indexOf(cur); return arr[Math.max(0, Math.min(arr.length - 1, (i < 0 ? 0 : i) + d))]; };
          root.tabIndex = 0;
          // a click anywhere in the widget (a map, a pick, a button) leaves the
          // keyboard with the root: the maps' own keyboard is off, and a click on
          // a canvas otherwise parks focus on the canvas or the page body, where
          // the keys above never arrive (Stephen, 2026-09-03). Inputs keep theirs.
          root.addEventListener("pointerup", (e) => {
            if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
            setTimeout(() => { try { root.focus({preventScroll: true}); } catch (err) {} }, 0);
          });
          root.addEventListener("keydown", (e) => {
            if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
            const k = e.key;
            if (k === "[" || k === "]" || k === "ArrowLeft" || k === "ArrowRight") { s2y = step(cfg.s2_years || [], s2y, (k === "]" || k === "ArrowRight") ? 1 : -1); styleS2(); yrRelease(); }
            else if (k === ";" || k === "'") { s2scale = Math.round(10 * Math.max(SC_MIN, Math.min(SC_MAX, s2scale + (k === "'" ? 0.1 : -0.1)))) / 10; styleSc(); scRelease(); }
            else if (k >= "1" && k <= "9") { const f = fills[Number(k) - 1]; if (f) { fill = f.value; styleFill(); send("fill"); } }
            else if (k === "-" || k === "=") { const v = step(aefYears, y0, k === "=" ? 1 : -1); if (v < y1) { y0 = v; styleAef(); aefRelease(); } }
            else if (k === "_" || k === "+") { const v = step(aefYears, y1, k === "+" ? 1 : -1); if (v > y0) { y1 = v; styleAef(); aefRelease(); } }
            else if (k === "l" || k === "L") { labelsOn = !labelsOn; labels(labelsOn); send("labels"); }
            else if (k === "f" || k === "F") { toggleFull(); }
            else return;
            e.preventDefault();
          });

          // STRIP_MINIMAL: the status line shows only while a fold is running,
          // after a failure, or when the hexagons are off for zoom (Stephen,
          // 2026-09-01: "we need some status like folding if it is"); the
          // timings and tile counts of a finished fold stay hidden
          const say = (t) => {
            status.textContent = t || "";
            if (cfg.minimal) status.hidden = !/folding|failed|zoom in past|no match|search:/.test(t || "");
          };
          const renderLegend = () => {
            legend.replaceChildren();
            let items = [];
            try { items = JSON.parse(model.get("legend") || "[]"); } catch (e) { items = []; }
            // ROWS (Stephen, 2026-09-02, after the stacked bar: "this is not
            // helpful"): the year fills come as items with a share; one row
            // per year, the colour box first, then the year and its share
            // on the same line, each year on a new line, inside a card
            const rows = items.length > 0 && items.every((it) => !it.ramp && it.pct != null);
            if (rows) {
              const card = document.createElement("div");
              card.className = "sp-rows";
              card.style.cssText = "display:flex;flex-direction:column;gap:.25rem;padding:6px 10px 8px;font-size:13px;" +
                "background:#fff;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.18)";
              for (const it of items) {
                const row = document.createElement("div");
                row.style.cssText = "display:flex;align-items:center;gap:.45rem;white-space:nowrap";
                const chip = document.createElement("span");
                chip.style.cssText = "display:inline-block;flex:0 0 12px;width:12px;height:12px;border-radius:2px;background:" + it.hex;
                const t = document.createElement("span");
                t.textContent = it.name + " " + it.pct + "%";
                row.append(chip, t);
                card.appendChild(row);
              }
              legend.appendChild(card);
              return;
            }
            for (const it of items) {
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

          // ---- the data ----------------------------------------------------
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

          // ---- S2 tiles: ask the kernel ------------------------------------
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

          // ---- the layers ---------------------------------------------------
          let mapL = null, mapR = null, ovL = null, ovR = null;
          let hover = null;  // the hovered cell (hex string), mirrored on both panes
          // deck's layers go under the basemap labels
          const labelSlot = () => cfg.labels_slot || "watername_ocean";
          const slot = labelSlot;
          const ring = (h) => { try { return cellToBoundary(h, true); } catch (e) { return null; } };
          // Overture divisions as PMTiles on Source Cooperative (cboettig/
          // overturemaps), regions and counties, the world over: one vector
          // source per level on each map and a fill at zero opacity that is
          // never seen, so queryRenderedFeatures can name the region and
          // county under a click from the tiles already on screen. No lines:
          // the division lines came out on 2026-09-02 and stay out.
          const ADMIN = ["regions", "counties"];
          const aSrc = (k) => "admin-" + k + "-src", aFill = (k) => "admin-" + k + "-fill";
          const adminSync = (m) => {
            if (!m || !cfg.admin_pm) return;
            for (const k of ADMIN) {
              if (m.getSource(aSrc(k))) continue;
              try {
                m.addSource(aSrc(k), {type: "vector", url: "pmtiles://" + cfg.admin_pm + "/" + k + ".pmtiles"});
                m.addLayer({id: aFill(k), type: "fill", source: aSrc(k), "source-layer": k,
                  filter: ["==", ["get", "class"], "land"], paint: {"fill-opacity": 0}}, slot());
              } catch (e) { console.error("admin " + k, e); }
            }
          };
          // the region and county under a point, from the tiles on screen.
          // Empty when the tiles are not in yet; the kernel's query fills
          // the same levels then.
          const adminAt = (m, pt) => {
            const out = {};
            const one = (k) => {
              if (!m.getLayer(aFill(k))) return null;
              const fs = m.queryRenderedFeatures(pt, {layers: [aFill(k)]});
              return fs && fs.length ? fs[0].properties : null;
            };
            try {
              const r = one("regions");
              if (r) out.region = r.name_en || r["names.primary"] || r.name || null;
              const c = one("counties");
              if (c) out.county = c.name_en || c["names.primary"] || c.name || null;
            } catch (e) {}
            return out;
          };
          const outline = (id, h, color, width) => {
            const r = h ? ring(h) : null;
            if (!r) return null;
            return new PathLayer({id, data: [r], getPath: (d) => d, getColor: color,
              widthUnits: "pixels", getWidth: width, widthMinPixels: 1, beforeId: slot()});
          };
          const mkRaster = (src, year, maxZ, extent, visible) => new TileLayer({
            // the S2 id carries the scale generation: a bump is a NEW layer to
            // deck, so its tile cache goes and every tile in view is asked for
            // again at the new gain
            id: src + "-" + year + (src === "s2" && cfg.s2_gen ? "-s" + cfg.s2_gen : ""),
            getTileData: getTileDataFor(src, year),
            onTileError: (e) => { if (!e || e.name !== "AbortError") say(src + " tile: " + ((e && e.message) || e)); },
            tileSize: cfg.tile || 256,
            minZoom: 0, maxZoom: maxZ,
            extent: extent || null,
            visible: visible,
            refinementStrategy: "best-available",
            beforeId: slot(),
            renderSubLayers: (p) => {
              if (!p.data) return null;
              const {west, south, east, north} = p.tile.bbox;
              return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]});
            },
          });
          const hexZoomOk = () => !!mapR && mapR.getZoom() >= (cfg.hex_zoom || 9);
          function layersLeft() {
            const out = [];
            out.push(mkRaster("s2", cfg.s2_year, 14, null, true));
            const h = outline("hover-l", hover, [255, 255, 255, 255], 2);
            if (h) out.push(h);
            const pk = cfg.hit ? outline("picked-l", cfg.hit, [255, 200, 40, 255], 3) : null;
            if (pk) out.push(pk);
            return out;
          }
          function layersRight() {
            const out = [];
            // the WSF raster: below the hexagon zoom (the pyramid, the earliest
            // built-up date under each pixel) and whenever the fill is `built`
            // (the record itself, level 0 from zoom 13; no hexagons then)
            const rasterOn = !hexZoomOk() || !dataObj || fill === "built";
            out.push(mkRaster("wsf", 0, 14, cfg.extent || null, rasterOn));
            if (dataObj && fill !== "built") out.push(new H3HexagonLayer({
              id: "hexes",
              data: {length: N},
              getHexagon: (_, {index}) => hexes[index],
              getFillColor: (_, {index}) => [colors[4 * index], colors[4 * index + 1], colors[4 * index + 2], colors[4 * index + 3]],
              updateTriggers: {getFillColor: [dataObj], getHexagon: [dataObj]},
              filled: true, stroked: false, extruded: false,
              highPrecision: true,
              pickable: false,
              beforeId: slot(),
            }));
            const h = outline("hover-r", hover, [255, 255, 255, 255], 2);
            if (h) out.push(h);
            const pk = cfg.hit ? outline("picked-r", cfg.hit, [255, 200, 40, 255], 3) : null;
            if (pk) out.push(pk);
            return out;
          }
          function update() {
            if (ovL) ovL.setProps({layers: layersLeft()});
            if (ovR) ovR.setProps({layers: layersRight()});
          }
          function updateHover() {
            if (ovL) ovL.setProps({layers: layersLeft()});
            if (ovR) ovR.setProps({layers: layersRight()});
          }

          function labels(on) {
            for (const m of [mapL, mapR]) {
              if (!m || !m.isStyleLoaded()) continue;
              const st = m.getStyle();
              if (!st || !st.layers) continue;
              st.layers.forEach((l) => {
                if (l.layout && l.layout["text-field"] !== undefined)
                  m.setLayoutProperty(l.id, "visibility", on ? "visible" : "none");
              });
            }
          }

          let seq = 0, lastView = "";
          function sendView() {
            if (!mapL) return;
            const c = mapL.getCenter();
            const v = {longitude: c.lng, latitude: c.lat, zoom: mapL.getZoom(), w: L.mapEl.clientWidth, h: L.mapEl.clientHeight};
            const key = JSON.stringify(v);
            if (key === lastView) return;
            lastView = key;
            v.n = ++seq;
            model.set("view", JSON.stringify(v));
            model.save_changes();
            // FOLDING (Stephen, 2026-09-02: "when it's calculating hexagons, I
            // wanna know that"): the strip's status line says folding the
            // instant a view past hex zoom goes up, before the server answers;
            // the server's next status replaces it
            if (v.zoom >= (cfg.hex_zoom || 9)) say("folding hexagons\u2026");
          }

          const cellAt = (lngLat) => {
            if (res < 0) return null;
            try { const h = latLngToCell(lngLat.lat, lngLat.lng, res); return hexIndex.has(h) ? h : null; }
            catch (e) { return null; }
          };

          function boot() {
            const home = cfg.home || {longitude: -96, latitude: 38.5, zoom: 4};
            const mk = (elm) => new maplibregl.Map({
              container: elm, style: STYLE,
              center: [home.longitude, home.latitude], zoom: home.zoom,
              // the credit in full (Carto, OpenStreetMap): compact hid it behind
              // an (i) that read as nothing (Stephen, 2026-09-02)
              attributionControl: {compact: false},
            });
            mapL = mk(L.mapEl); mapR = mk(R.mapEl);
            // maplibre's keyboard handler is OFF on both maps: a clicked map
            // (the pick, which the hexagon highlight needs) holds focus, and its
            // arrows panned and its - = zoomed, under the widget's own keys
            // (Stephen, 2026-09-03: "I can't then left right arrow move the S2").
            // The arrows now step the S2 year; the mouse pans and zooms.
            mapL.keyboard.disable(); mapR.keyboard.disable();
            // maplibre's own controls: full screen takes the WHOLE widget (both
            // panes and the strip), not the one map it sits on
            mapR.addControl(new maplibregl.FullscreenControl({container: root}), "top-right");
            // the zoom buttons under full screen, the default maplibre column
            mapR.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
            ovL = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck L: " + (e && e.message ? e.message : e))});
            ovR = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck R: " + (e && e.message ? e.message : e))});
            mapL.addControl(ovL); mapR.addControl(ovR);
            // one camera: whichever map moves, the other jumps to it (guarded
            // against the echo). moveend on either sends the view (deduped).
            let syncing = false;
            const follow = (a, b) => () => {
              if (syncing) return;
              syncing = true;
              b.jumpTo({center: a.getCenter(), zoom: a.getZoom(), bearing: a.getBearing(), pitch: a.getPitch()});
              syncing = false;
            };
            mapL.on("move", follow(mapL, mapR));
            mapR.on("move", follow(mapR, mapL));
            let ready = 0;
            const onLoad = () => { ready++; if (ready === 2) { labels(labelsOn); adminSync(mapL); adminSync(mapR); update(); sendView(); } };
            mapL.on("load", onLoad); mapR.on("load", onLoad);
            mapL.on("moveend", sendView); mapR.on("moveend", sendView);
            mapL.on("zoom", () => update());
            mapR.on("zoom", () => update());
            for (const [m, other] of [[mapL, mapR], [mapR, mapL]]) {
              m.on("mousemove", (e) => {
                const h = cellAt(e.lngLat);
                if (h !== hover) { hover = h; updateHover(); }
              });
              m.on("mouseout", () => { if (hover) { hover = null; updateHover(); } });
              m.on("click", (e) => {
                const h = cellAt(e.lngLat);
                model.set("pick", JSON.stringify({cell: h, lon: e.lngLat.lng, lat: e.lngLat.lat, admin: adminAt(m, e.point), n: ++seq}));
                model.save_changes();
              });
              m.on("error", (ev) => { if (ev && ev.error && ev.error.message) say("map: " + ev.error.message); });
              new ResizeObserver(() => { try { m.resize(); } catch (e) {} }).observe(m.getContainer());
            }
            window.__spTiles = tstat;
            window.__spMaps = () => [mapL, mapR];
            window.__spLayers = () => ({left: layersLeft().map((l) => l.id), right: layersRight().map((l) => l.id), N, res});
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
          model.on("change:config", () => {
            const was = cfg;
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            // the S2 year: the handle is the UI's, the echo only draws the layer
            // (a late echo of an older year snapped the handle back after two
            // quick steps; the kernel never changes the year on its own)
            if (Number(cfg.s2_scale) && Number(cfg.s2_scale) !== s2scale) { s2scale = Number(cfg.s2_scale); scSent = s2scale; styleSc(); }
            if (cfg.fill && cfg.fill !== fill) { fill = cfg.fill; styleFill(); }
            if (cfg.labels !== was.labels) { labelsOn = cfg.labels !== false; labels(labelsOn); }
            update();
          });
          try { grab("cells"); grab("colors"); loadCells(); loadAttrs(); renderLegend(); say(model.get("status")); boot(); }
          catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { try { mapL && mapL.remove(); mapR && mapR.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (PairMap,)


@app.cell
def _(
    ADMIN_PM,
    AEF_FROM0,
    AEF_TO0,
    AEF_YEARS_ALL,
    FILLS,
    FILL_NAMES,
    FILL_SHORT,
    HEX_ZOOM,
    HOME,
    LABELS_SLOT,
    PairMap,
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
    pair = PairMap(config=json.dumps({
        "height": VIEW_H, "home": dict(HOME), "labels": True, "labels_slot": LABELS_SLOT, "tile": RASTER_TILE,
        "s2_year": S2_YEAR0, "s2_scale": S2_SCALE0, "s2_gen": 0, "fill": FILLS[0],
        "s2_years": list(S2_YEARS),
        "aef_from": AEF_FROM0, "aef_to": AEF_TO0, "aef_years": list(AEF_YEARS_ALL),
        "fills": [[f, FILL_SHORT[f], FILL_NAMES[f]] for f in FILLS],
        "hex_zoom": HEX_ZOOM, "extent": list(wsf_bounds),
        "minimal": STRIP_MINIMAL, "admin_pm": ADMIN_PM,
    }))
    HOLD = {
        "frame": None, "sent": None, "box": None, "res": None, "vs": None,
        "busy": False, "pending": None, "pending_force": False, "task": None, "loop": None,
        "s2y": S2_YEAR0, "s2scale": S2_SCALE0, "s2gen": 0, "fill": FILLS[0], "labels": True,
        "y0": AEF_FROM0, "y1": AEF_TO0,
        "hit": None, "pick_n": None, "panel_body": "", "memo": {}, "aef": {}, "wsf": {}, "h_cam": None, "h_ctl": None, "h_pick": None,
        "runs": 0,
    }
    pair
    return HOLD, pair


@app.cell
def _(
    AEF_YEARS_ALL,
    CELL_KM2,
    FILLS,
    FILL_NAMES,
    HEX_ZOOM,
    HOLD,
    HOME,
    NEW_MIN,
    S2_YEARS,
    SETTLE,
    STRIP_MINIMAL,
    aef_fold,
    asyncio,
    build_frame,
    con,
    contains,
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
):
    # ---- wiring: the camera loop and the controls. Re-runs freely. ---------------
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
        try:
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

    def _hexes_off(msg):
        if HOLD["sent"] is not None:
            with pair.hold_sync():
                pair.cells, pair.colors = b"", b""
            HOLD["sent"] = None
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = None, None, None, None
        _cfg(hit=None)
        pair.legend = "[]"
        pair.panel = ""
        _say(msg)

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
        pair.legend = json.dumps(fr["legend"](HOLD["fill"]))
        return True

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        if vsd["zoom"] < HEX_ZOOM:
            _hexes_off(f"zoom {vsd['zoom']:.1f} · WSF raster on the right (the earliest built-up date under each pixel) · zoom in past {HEX_ZOOM:g} for the hexagons")
            return
        view = view_to_bbox(vsd)
        box = pad_box(view)
        inside = HOLD["box"] is not None and contains(HOLD["box"], view)
        if inside and not force:
            if res_for_view(vsd, box) <= HOLD["res"]:
                _say(HOLD.get("last_status", "") + " · held")
                return
        res = res_for_view(vsd, box)
        y0, y1 = HOLD["y0"], HOLD["y1"]
        rbox = tuple(round(v, 3) for v in box)
        key = (y0, y1, res, rbox)
        t0 = time.time()
        years = list(range(y0, y1 + 1))
        _say(f"folding WSF and AlphaEarth {y0} to {y1} ({len(years)} years)…" if STRIP_MINIMAL
             else f"res {res} · folding WSF, AlphaEarth {y0}..{y1} ({len(years)} years)… (wiring run {HOLD['runs']})")
        if key in HOLD["memo"]:
            fr, stats = HOLD["memo"][key]
        else:
            bkey = (res, rbox)  # the fold cache is per res and BOX
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
                _say(f"res {res} · {s1}")
                return
            aef_by_year = {y: HOLD["aef"][(y, bkey)][0] for y in years if (y, bkey) in HOLD["aef"]}
            s2s = " · ".join(HOLD["aef"][(y, bkey)][1] for y in years if (y, bkey) in HOLD["aef"])
            t1 = time.time()
            loop = asyncio.get_running_loop()
            fr = await loop.run_in_executor(None, build_frame, nw, aef_by_year, y0, y1)
            stats = " · ".join(x for x in (f"res {res}", s1, s2s, f"frame {time.time() - t1:.1f} s") if x)
            HOLD["memo"][key] = (fr, stats)
            if len(HOLD["memo"]) > 12:
                HOLD["memo"].pop(next(iter(HOLD["memo"])))
        HOLD["frame"], HOLD["box"], HOLD["res"], HOLD["hit"] = fr, box, res, None
        t2 = time.time()
        _paint()
        st = s2_raster_stats()
        HOLD["last_status"] = (
            f"{stats} · {fr['score']}"
            + f" · send {time.time() - t2:.2f} s · {time.time() - t0:.1f} s"
            f" · S2 tiles {st['served']:,} served, {st['blank']:,} empty"
            + "".join(f" · S2 {y}: {100 * v:.0f}% of pixels backfilled from the temporal median" for y, v in sorted(st["fill"].items()))
        )
        _say(HOLD["last_status"])

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

    # where: the place ladder under the click, from Overture divisions. The
    # region and county come with the click from the PMTiles in the browser;
    # the whole ladder, local_type included, is a query against Source
    # Cooperative off the loop, and lands after if this click is still the
    # last one. Each level is printed the way Overture names it, with the
    # country's own word beside when it has one (city, town, prefecture).
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

    def _on_pick(change):
        fr = HOLD["frame"]
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        if fr is None:
            return
        try:
            cellh = p.get("cell")
            if not cellh:
                HOLD["hit"] = None
                pair.panel = ""
                _paint()
                return
            cell = int(cellh, 16)
            con.register("cur_cells", fr["cells"])
            r = con.execute(
                "SELECT p_built, p_new, byear, byear_name, first_date, disp, disp_max, when_name "
                "FROM cur_cells WHERE cell = ?", [cell]
            ).fetchone()
            ci = int(np.searchsorted(fr["cellid"], np.uint64(cell)))
            row_steps = fr["steps"][:, ci] if r is not None and ci < len(fr["cellid"]) else []
            lat, lon = p.get("lat"), p.get("lon")
            where = f" at {lat:.4f}, {lon:.4f}" if lat is not None and lon is not None else ""
            n, adm = p.get("n"), p.get("admin") or {}
            HOLD["pick_n"] = n
            l0 = _place_line(_place_from_tiles(adm), pending=True) if lon is not None else ""
            if r is None:
                HOLD["hit"] = None
                body = f"<span style='opacity:.7'>{cellh}{where}: not in the current frame</span>"
            else:
                HOLD["hit"] = cell if HOLD["hit"] != cell else None
                pb, pn, by, byn, fd, dsp, dmx, wn = r
                y0, y1 = fr["y0"], fr["y1"]
                d0 = fr["D0"]
                # BLUNT (Stephen): one plain sentence per source, then the numbers small
                if pb <= 0:
                    l1 = "WSF sees no built-up ground here."
                elif by >= 0:
                    l1 = (f"WSF: <b>{100 * pb:.0f}%</b> of this hexagon is built-up by the end of {y1}; "
                          f"<b>{100 * pn:.1f}%</b> of it was built {y0 + 1} to {y1}, most of that in <b>{by}</b>.")
                elif pn > 0:
                    l1 = (f"WSF: <b>{100 * pb:.0f}%</b> of this hexagon is built-up by the end of {y1}; "
                          f"{100 * pn:.2f}% of it was built {y0 + 1} to {y1} (under the {100 * NEW_MIN:g}% that counts as growth).")
                else:
                    l1 = f"WSF: <b>{100 * pb:.0f}%</b> of this hexagon is built-up, all of it before {y0 + 1} (first seen {fd})."
                # BLUNT (Stephen): the fill paints the end-to-end shift, so the sentence
                # speaks to that first; a year is named only when one step stood out
                if wn.startswith("no AlphaEarth") or dsp is None or np.isnan(dsp):
                    l2 = "AlphaEarth has no embedding here."
                else:
                    s_lo, s_hi = fr.get("shift_lo", 0.0), fr.get("shift_hi", 1.0)
                    t = (float(dsp) - s_lo) / max(s_hi - s_lo, 1e-9)
                    if t >= 0.75:
                        how = "moved <b>significantly</b>"
                    elif t >= 0.4:
                        how = "moved <b>a fair amount</b>"
                    elif t >= 0.15:
                        how = "moved <b>a little</b>"
                    else:
                        how = "<b>barely moved</b>"
                    l2 = f"AlphaEarth: the ground's fingerprint {how} from {y0} to {y1} (shift {_f(dsp)}, this view runs {_f(s_lo)} to {_f(s_hi)})"
                    if wn.startswith("no single year"):
                        l2 += ", spread across the years rather than in any one of them." if t >= 0.4 else "; no single year stood out."
                    else:
                        yr = wn.split("changed in ")[1].split(" ")[0]
                        l2 += f", most sharply in <b>{yr}</b>."
                detail = (
                    f"shift {y0} to {y1} {_f(dsp)} · steps "
                    + " · ".join(f"{ya}→{yb} {_f(v)}" for (ya, yb), v in zip(fr["step_years"], row_steps))
                    + (f" · a step counts as change above {_f(d0)}" if not np.isnan(d0) else "")
                    + f" · first built-up {fd} · {CELL_KM2.get(HOLD['res'], 0):.3f} km²{where}"
                )
                body = (
                    f"<div style='font-size:14px;line-height:1.5'>{l1}<br>{l2}</div>"
                    + ("" if STRIP_MINIMAL else f"<div style='font-size:12px;color:#777'>{detail}</div>")
                )
            HOLD["panel_body"] = body
            pair.panel = l0 + body
            if l0:
                async def _place_later(n=n, lon=lon, lat=lat, adm=adm):
                    try:
                        d = await asyncio.to_thread(division_at, lon, lat)
                    except Exception as e:
                        d = []
                        _say(f"place (fused/overture): {e}")
                    if HOLD.get("pick_n") != n:
                        return
                    pair.panel = _place_line(d or _place_from_tiles(adm), pending=False) + HOLD["panel_body"]

                _spawn(_place_later())
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

    def _on_ctl_body(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        act = c.get("act")
        if act == "s2":
            y = int(c.get("s2y", HOLD["s2y"]))
            if y in S2_YEARS and y != HOLD["s2y"]:
                HOLD["s2y"] = y
                _cfg(s2_year=y)
                _say((HOLD.get("last_status") or "") + f" · Sentinel-2 {y}")
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
                _say((HOLD.get("last_status") or "") + f" · Sentinel-2 scale {v:.1f}× · tiles re-served")
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
                if _paint():
                    _say((HOLD.get("last_status") or "") + f" · {FILL_NAMES[f]}")
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

    if HOLD["frame"] is None and not HOLD["busy"]:
        _request()
    else:
        _paint()
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
    """)
    return


@app.cell
def _(mo):
    tables_btn = mo.ui.run_button(label="tables for the current view")
    tables_btn
    return (tables_btn,)


@app.cell
def _(HOLD, con, mo, tables_btn):
    mo.stop(not tables_btn.value or HOLD["frame"] is None, mo.md("*no view folded yet*"))
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
