# AEF, Sentinel-2 and landcover explorer

AlphaEarth Foundations embeddings folded to H3 to show where the ground
changed, read against ESA WorldCover 2021 for what is on the ground, with
the Earth Genome Sentinel-2 yearly mosaics as the imagery for checking the
change. Three notebooks:

- `aef-s2-landcover-explorer.py`: where the ground changed, and when.
  [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-landcover-explorer.py)
- `aef-s2-new-construction.py`: only new construction, learned in each
  view from the World Settlement Footprint and found in AlphaEarth,
  including places WSF did not record.
  [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-new-construction.py)
- `aef-s2-kinds-of-change.py`: the ground that moved most, grouped by the
  way it moved, from AlphaEarth alone, with its land cover read from
  AlphaEarth every year.
  [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-kinds-of-change.py)

## Datasets

| Dataset | Producer | Where | License |
| --- | --- | --- | --- |
| AlphaEarth Foundations Satellite Embedding, annual, 2017 to 2025 | Google and Google DeepMind ([dataset page](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)) | [tge-labs/aef](https://source.coop/tge-labs/aef), [tge-labs/aef-mosaic](https://source.coop/tge-labs/aef-mosaic) | CC BY 4.0 |
| World Settlement Footprint (WSF) Tracker, 10 m, mid-2016 to the end of 2025 (new construction notebook) | DLR and MindEarth | [mindearth/wsf](https://source.coop/mindearth/wsf) ([DOI 10.5281/zenodo.20424537](https://doi.org/10.5281/zenodo.20424537)) | CC BY 3.0 IGO |
| Impact Observatory, Microsoft and Esri 10 m annual land use and land cover v02, 2017 to 2023 (kinds of change notebook) | Impact Observatory, Microsoft and Esri | Microsoft Planetary Computer, collection `io-lulc-annual-v02` | CC BY 4.0 |
| Overture Maps transportation and land use (PMTiles release 2026-08-19.0, kinds of change notebook) | Overture Maps Foundation, from OpenStreetMap | Overture's PMTiles (`overturemaps-extras-us-west-2`, `tiles/2026-08-19.0/transportation.pmtiles` and `base.pmtiles`) | ODbL |
| ESA WorldCover 10 m 2021 v200 | ESA WorldCover consortium, from Copernicus Sentinel data | `s3://esa-worldcover/v200/2021/map` (AWS open data) | CC BY 4.0 |
| Overture Maps divisions (place names: PMTiles release 2026-08-19.0, and GeoParquet release 2026-05-20.0 via [fused/overture](https://source.coop/fused/overture)) | Overture Maps Foundation, from OpenStreetMap, geoBoundaries, Esri Community Maps contributors and LINZ | Overture's release bucket, Source Cooperative | ODbL (the geoBoundaries, Esri and LINZ parts CC BY 4.0); see [Overture attribution](https://docs.overturemaps.org/attribution/) |
| Photon place search, over OpenStreetMap | komoot | [photon.komoot.io](https://photon.komoot.io/) | ODbL (OpenStreetMap data) |
| Sentinel-2 yearly mosaics (true color, 2022 to 2025) | Earth Genome, from Copernicus Sentinel data | Found through Earth Genome's STAC API ([stac.earthgenome.org](https://stac.earthgenome.org/), collection `sentinel2-yearly-mosaics`); the COGs are read from [earthgenome/earthindeximagery](https://source.coop/earthgenome/earthindeximagery) on Source Cooperative | CC BY 4.0 |
| Sentinel-2 temporal mosaics (fills holes in the yearly mosaic, 2022 and 2023) | Earth Genome, from Copernicus Sentinel data | Earth Genome's STAC, collection `sentinel2-temporal-mosaics`; COGs from [earthgenome/sentinel2-temporal-mosaics](https://source.coop/earthgenome/sentinel2-temporal-mosaics) | CC BY 4.0 |

**Earth Genome's STAC.** The imagery is not found through a static
catalog: each imagery tile asks Earth Genome's own STAC API
(`stac.earthgenome.org/search`) which mosaic items cover it, then reads the
COGs its assets point to on Source Cooperative. Both mosaics are CC BY 4.0,
as their READMEs on Source Cooperative state
([yearly](https://data.source.coop/earthgenome/earthindeximagery/README.md),
[temporal](https://data.source.coop/earthgenome/sentinel2-temporal-mosaics/README.md)).
The STAC's record for `sentinel2-yearly-mosaics` puts `proprietary` in its
license field, with no license link; the Source Cooperative README is the
statement of the license.

## The change notebook

`aef-s2-landcover-explorer.py`: AlphaEarth change hexagons in viridis (how
much the ground changed over the years read, or, in YlOrBr, the year its
change stood out most against that year's usual change in view), ESA
WorldCover 2021 class shares per hexagon, and the Earth Genome Sentinel-2
mosaic on demand. Below zoom 9 the map is WorldCover itself, drawn as tiles
from ESA's COGs in a palette without red; the hexagons start at zoom 9.
Hold space (the map still pans) to swap the hexagons for
the imagery; scroll while holding to step the year. Click a hexagon for its
year-to-year steps, its land cover and the Overture divisions it sits in
(gold outline, click again to clear). Click on the imagery with space held
for the cell's H3 string and lat, long, each copyable. The search box
takes a place name or an H3 string: it flies there and outlines the cell
in blue. The full key list is at the top of the notebook.

The hexagons reach the browser as map tiles in which each pixel names the
hexagons near it; the shader draws each edge from the H3 boundary, so
edges stay smooth at any zoom, and switching the color mode only updates a
small color table.

### How change is aggregated

AlphaEarth pixels are folded to H3 in DataFusion (xarray-sql, with an
h3ronpy UDF) and averaged per cell, one fold per year. Change is 1 minus
the cosine between a cell's normalized vectors for two years.

Averaging every pixel in a hexagon and measuring the change of that
average would let a small site that changed a lot inside a hexagon of
quiet ground be averaged away when zoomed out. This notebook
folds one H3 level finer than the hexagons drawn (about one pixel of the
read per finer cell), works out the change per finer cell, then uses
h3ronpy's `change_resolution` to group the finer cells under their hexagon.
Each hexagon takes its most-changed finer cell whole: its change, its year
of biggest change and its year-to-year steps. A hexagon shows its strongest
spot, not its average.

[Open it in molab](https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-landcover-explorer.py): it runs in the same region as the data, where
the AlphaEarth reads are several times faster. Locally (dependencies are
declared inline, PEP 723):

```
uv run marimo run aef-s2-landcover-explorer.py --sandbox
```

## The new construction notebook

`aef-s2-new-construction.py` is a copy of the change notebook that shows
only new construction (`A`, the default; its raw AEF Change mode, `S`, is
still there, and the change year moved to the card). The World Settlement Footprint tracker dates, every half year,
when each 10 m pixel first read as built-up. For the view on screen it is
folded to the same finer H3 cells as AlphaEarth. A finer cell is an
example of new construction when at least 20% of its WSF samples first
read as built inside the years read, and of unchanged ground when under
2% were built from the first year on. A logistic regression on each
cell's first- and last-year AlphaEarth vectors (128 numbers, fit with
numpy) learns the difference and scores every cell, WSF's or not. Each
hexagon takes its best-scoring finer cell, and hexagons scoring 0.5 or
more are drawn, colored in Oranges by the year their change stood out. Half
of the unchanged examples are ground that moved in AlphaEarth as much as
new construction does (fields, roads, cleared land, water), weighted up, so
the model learns built against other change rather than change against
quiet ground. A held-out fifth of the examples says how many of WSF's new
places the model finds and how many of its picks WSF also calls new; the
status line reports it.

Zoomed in past about 13, every AlphaEarth year (2017 to 2025) is read in
the background once the window's hexagons are up.
Each hexagon's change is judged by what the ground did around it: one step
that held, came back, changes this much most years, kept changing after,
or too recent to tell. The card says which. New construction leaves out
the ones that came back or change most years; AEF Change stays raw.

The model is learned where there is enough to learn from: a view with at
least 25 new places by WSF. Zoomed in under that view it is kept, so a
close view with no WSF building of its own is still scored; zooming out,
moving off it or changing the years read teaches it again. Nothing is
saved to disk. Clicking a hexagon shows its score, WSF's reading of it
(newly built share and year, or nothing new) and the AlphaEarth steps.
Below zoom 9 the map is the plain basemap; WorldCover stays on the
hexagons, in the card.

```
uv run marimo run aef-s2-new-construction.py --sandbox
```

## The kinds of change notebook

`aef-s2-kinds-of-change.py` is a copy of the new construction notebook
with WSF switched off (`USE_WSF = False`; the WSF model is still in the
code for later). `A` is **Kinds of change**, from AlphaEarth alone: the
quarter of the view that moved most, less the change the whole view made,
grouped by the direction it moved into six kinds (spherical k-means), in
Okabe-Ito colors with quiet ground faint gray. The key gives each kind's
hexagons, most common year and the land cover it has more of than the
view; click a kind to hide it. `S` is raw AEF Change.

The land cover is read from AlphaEarth every year: a logistic regression
per year on the 64 numbers, taught per finer cell by Impact Observatory's
annual land cover (its own year on all ground, 2017 to 2023; 2024 and 2025
only on ground that barely moved) and by Overture roads, rail and land use
(the last year on all ground, earlier years only where the ground barely
moved). ESA WorldCover 2021 stands in until those are read.

[Open it in molab](https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-kinds-of-change.py), or locally:

```
uv run marimo run aef-s2-kinds-of-change.py --sandbox
```

## Run

Dependencies are declared inline (PEP 723):

```
uv run marimo edit aef-s2-landcover-explorer.py --sandbox
uv run marimo edit aef-s2-new-construction.py --sandbox
uv run marimo edit aef-s2-kinds-of-change.py --sandbox
```

## Attribution

AlphaEarth Foundations Satellite Embedding dataset by Google and Google
DeepMind (CC BY 4.0). WSF Tracker (c) DLR and MindEarth, via Source
Cooperative (mindearth/wsf, DOI 10.5281/zenodo.20424537), CC BY 3.0 IGO.
ESA WorldCover 10 m 2021 v200 (c) ESA WorldCover project,
contains modified Copernicus Sentinel data (2021) processed by the ESA
WorldCover consortium (CC BY 4.0). Impact Observatory, Microsoft and
Esri 10 m annual land use and land cover v02, via Microsoft Planetary
Computer (CC BY 4.0). Overture Maps transportation, land use and divisions: (c) OpenStreetMap
contributors, Overture Maps Foundation (ODbL), with geoBoundaries, Esri
Community Maps contributors and Land Information New Zealand (LINZ)
(CC BY 4.0).
Sentinel-2 yearly and temporal mosaics by Earth Genome (CC BY 4.0),
found through Earth Genome's STAC API. Search by Photon (komoot)
over OpenStreetMap data (ODbL). Basemap by Carto. Contains modified
Copernicus Sentinel data.
