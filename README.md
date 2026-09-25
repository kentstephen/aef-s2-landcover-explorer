# AEF, Sentinel-2 and landcover explorer

[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-landcover-explorer.py)

AlphaEarth Foundations embeddings read against two independent records of
what is on the ground: Overture Maps landcover and the World Settlement
Footprint, with Sentinel-2 as the imagery for tracking change: the Earth
Genome yearly mosaics or the EOPF Sentinel-2 zarr.

This repo steps back from the earlier Overture buildings work
([s2-wsf-aef-overture-pair](https://github.com/kentstephen/s2-wsf-aef-overture-pair))
and narrows the sources to these.

## Datasets

| Dataset | Producer | Where | License |
| --- | --- | --- | --- |
| AlphaEarth Foundations Satellite Embedding, annual, 2017 to 2025 | Google and Google DeepMind ([dataset page](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)) | [tge-labs/aef](https://source.coop/tge-labs/aef), [tge-labs/aef-mosaic](https://source.coop/tge-labs/aef-mosaic) | CC BY 4.0 |
| ESA WorldCover 10 m 2021 v200 | ESA WorldCover consortium, from Copernicus Sentinel data | `s3://esa-worldcover/v200/2021/map` (AWS open data) | CC BY 4.0 |
| Overture Maps divisions (place names: PMTiles, and GeoParquet via [fused/overture](https://source.coop/fused/overture)) | Overture Maps Foundation | Overture's release bucket, Source Cooperative | ODbL |
| World Settlement Footprint (WSF) Tracker, 10 m, 2016 to 2026 | DLR and MindEarth | [mindearth/wsf](https://source.coop/mindearth/wsf) ([DOI 10.5281/zenodo.20424537](https://doi.org/10.5281/zenodo.20424537)) | CC BY 3.0 IGO |
| Sentinel-2 yearly mosaics (true color, 2022 to 2025) | Earth Genome, from Copernicus Sentinel data | [earthgenome/sentinel2-yearly-mosaics](https://source.coop/earthgenome/sentinel2-yearly-mosaics) | CC BY 4.0 |
| Sentinel-2 L2A, EOPF zarr | ESA, Copernicus | [EOPF Sentinel Zarr Samples](https://zarr.eopf.copernicus.eu/) | Copernicus open license |

## The notebook

`aef-s2-landcover-explorer.py`: AlphaEarth change hexagons in viridis (how
much the ground changed over the years read, or, in YlOrBr, the year its
change stood out most against that year's usual change in view), ESA
WorldCover 2021 class shares per hexagon, and the Earth Genome Sentinel-2
mosaic on demand. Hold space (the map still pans) to swap the hexagons for
the imagery; scroll while holding to step the year. Click a hexagon for its
year-to-year steps, its land cover and the Overture divisions it sits in
(gold outline, click again to clear). Click on the imagery with space held
for the cell's H3 string and lat, long, each copyable. The full key list is
at the top of the notebook.

[Open it in molab](https://molab.marimo.io/github/github.com/kentstephen/aef-s2-landcover-explorer/blob/main/aef-s2-landcover-explorer.py): it runs in the same region as the data, where
the AlphaEarth reads are several times faster. Locally (dependencies are
declared inline, PEP 723):

```
uv run marimo run aef-s2-landcover-explorer.py --sandbox
```

## Carried over

The notebooks from the earlier project are still here as a starting point
(`s2-wsf-aef-overture-pair.py`, `s2-wsf-aef-overture-slider.py`,
`s2-wsf-aef-overture-atlas.py`). They hold the AEF and WSF readers and the
H3 folding in DataFusion, and the Earth Genome Sentinel-2 mosaic reader.
They also read Overture buildings and divisions, which are outside this
repo's scope.

## Run

Dependencies are declared inline (PEP 723):

```
uv run marimo edit <notebook>.py --sandbox
```

## Attribution

AlphaEarth Foundations Satellite Embedding dataset by Google and Google
DeepMind (CC BY 4.0). WSF Tracker (c) DLR and MindEarth, via Source
Cooperative. ESA WorldCover 2021 (CC BY 4.0). Overture Maps divisions (ODbL).
Sentinel-2 mosaics by Earth Genome (CC BY 4.0). Contains modified
Copernicus Sentinel data.
