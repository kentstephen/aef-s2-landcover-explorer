# Sentinel-2 / World Settlement Footprint / AlphaEarth Foundations / Overture Maps pair

[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/s2-wsf-aef-overture-pair/blob/main/s2-wsf-aef-overture-pair.py)

Settlement growth read two ways on one H3 grid, in a single marimo notebook.

Two maps share one camera. On the left, Earth Genome's [Sentinel-2 yearly
mosaic](https://source.coop/earthgenome/sentinel2-yearly-mosaics) (true colour,
2022 to 2025), rendered by the kernel from the COGs. On
the right, one H3 fill per hexagon from two independent records of change:

- **[WSF Tracker](https://source.coop/mindearth/wsf)** (World Settlement
  Footprint, DLR and MindEarth): the half-year each 10 m pixel was first
  observed as built-up. Per hexagon: the share built-up, the share that grew
  inside the window, and the year most of that new ground arrived.
- **[AlphaEarth Foundations](https://source.coop/tge-labs/aef)** (Google
  DeepMind): the mean 64-dimensional
  embedding per hexagon per year. How far it moved between the ends of the
  window (1 minus cosine similarity), and the first year it jumped past a
  quiet level set by the hexagons WSF says did not grow.

A click on a hexagon names its place from [Overture Maps
divisions](https://docs.overturemaps.org/guides/divisions/), locality up to
country. Everything is read live from Source Cooperative; nothing is
downloaded ahead of time. Every raster is folded onto H3 cells by the H3 UDF
inside DataFusion (xarray-sql), and the browser receives only cell ids and
colours.

## Run

[Open in molab](https://molab.marimo.io/github/github.com/kentstephen/s2-wsf-aef-overture-pair/blob/main/s2-wsf-aef-overture-pair.py).
molab runs in the same region as the data and is the faster place to open it.
Locally, the dependencies are declared inline (PEP 723):

```
uv run marimo edit s2-wsf-aef-overture-pair.py --sandbox
```

Controls, the fills and the click panel are described in the notebook's first
cell.

## Datasets

Everything is read live from [Source Cooperative](https://source.coop).

| Dataset | Producer | Source Cooperative | Licence |
| --- | --- | --- | --- |
| Sentinel-2 yearly mosaics (true colour, 2022 to 2025) | Earth Genome, from Copernicus Sentinel data | [earthgenome/sentinel2-yearly-mosaics](https://source.coop/earthgenome/sentinel2-yearly-mosaics), via the [Earth Genome STAC](https://stac.earthgenome.org) | CC BY 4.0 |
| Sentinel-2 temporal mosaics (fills nodata holes in the yearly) | Earth Genome, from Copernicus Sentinel data | [earthgenome/sentinel2-temporal-mosaics](https://source.coop/earthgenome/sentinel2-temporal-mosaics) | CC BY 4.0 |
| World Settlement Footprint (WSF) Tracker, 10 m, 2016 to 2026 | DLR and MindEarth | [mindearth/wsf](https://source.coop/mindearth/wsf) ([DOI 10.5281/zenodo.20424537](https://doi.org/10.5281/zenodo.20424537)) | CC BY 3.0 IGO |
| AlphaEarth Foundations Satellite Embedding, annual, 2017 to 2025 | Google and Google DeepMind ([dataset page](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)) | [tge-labs/aef](https://source.coop/tge-labs/aef) (index) and [tge-labs/aef-mosaic](https://source.coop/tge-labs/aef-mosaic) (COG overviews) | CC BY 4.0 |
| Overture Maps divisions (PMTiles, regions and counties) | Overture Maps Foundation | [cboettig/overturemaps](https://source.coop/cboettig/overturemaps) | ODbL |
| Overture Maps divisions (GeoParquet, point-in-polygon on click) | Overture Maps Foundation | [fused/overture](https://source.coop/fused/overture) | ODbL |

Also used: place search by [Photon](https://photon.komoot.io/) (komoot) over
OpenStreetMap data (ODbL); basemap by [Carto](https://carto.com/basemaps/).

## Attribution

WSF Tracker (c) DLR and MindEarth, via Source Cooperative
([mindearth/wsf](https://source.coop/mindearth/wsf), DOI
[10.5281/zenodo.20424537](https://doi.org/10.5281/zenodo.20424537)). The
[AlphaEarth Foundations Satellite Embedding
dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
is produced by Google and Google DeepMind (CC BY 4.0). Sentinel-2
[yearly](https://source.coop/earthgenome/sentinel2-yearly-mosaics) and
[temporal](https://source.coop/earthgenome/sentinel2-temporal-mosaics) mosaics
by Earth Genome from Copernicus Sentinel data (CC BY 4.0). Overture Maps
divisions (ODbL) via Source Cooperative
([cboettig/overturemaps](https://source.coop/cboettig/overturemaps),
[fused/overture](https://source.coop/fused/overture)). Place search by
[Photon](https://photon.komoot.io/) (komoot) over OpenStreetMap data (ODbL).
Basemap by [Carto](https://carto.com/basemaps/).
