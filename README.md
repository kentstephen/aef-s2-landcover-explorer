# AEF, Sentinel-2 and landcover explorer

[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-landcover-explorer.py)

AlphaEarth Foundations embeddings folded to H3 to show where the ground
changed, read against ESA WorldCover 2021 for what is on the ground, with
the Earth Genome Sentinel-2 yearly mosaics as the imagery for checking the
change.

## Datasets

| Dataset | Producer | Where | License |
| --- | --- | --- | --- |
| AlphaEarth Foundations Satellite Embedding, annual, 2017 to 2025 | Google and Google DeepMind ([dataset page](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)) | [tge-labs/aef](https://source.coop/tge-labs/aef), [tge-labs/aef-mosaic](https://source.coop/tge-labs/aef-mosaic) | CC BY 4.0 |
| ESA WorldCover 10 m 2021 v200 | ESA WorldCover consortium, from Copernicus Sentinel data | `s3://esa-worldcover/v200/2021/map` (AWS open data) | CC BY 4.0 |
| Overture Maps divisions (place names: PMTiles, and GeoParquet via [fused/overture](https://source.coop/fused/overture)) | Overture Maps Foundation | Overture's release bucket, Source Cooperative | ODbL |
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

## The notebook

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

## Run

Dependencies are declared inline (PEP 723):

```
uv run marimo edit aef-s2-landcover-explorer.py --sandbox
```

## Attribution

AlphaEarth Foundations Satellite Embedding dataset by Google and Google
DeepMind (CC BY 4.0). ESA WorldCover 10 m 2021 v200 (c) ESA WorldCover project,
contains modified Copernicus Sentinel data (2021) processed by the ESA
WorldCover consortium (CC BY 4.0). Overture Maps divisions (ODbL).
Sentinel-2 yearly and temporal mosaics by Earth Genome (CC BY 4.0),
found through Earth Genome's STAC API. Search by Photon (komoot)
over OpenStreetMap data (ODbL). Basemap by Carto. Contains modified
Copernicus Sentinel data.
