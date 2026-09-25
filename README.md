# AEF, landcover and settlements

AlphaEarth Foundations embeddings read against two independent records of
what is on the ground: Overture Maps landcover and the World Settlement
Footprint. Possibly the EOPF Sentinel-2 zarr as the imagery for tracking
change.

This repo steps back from the earlier Overture buildings work
([s2-wsf-aef-overture-pair](https://github.com/kentstephen/s2-wsf-aef-overture-pair))
and narrows the sources to these.

## Datasets

| Dataset | Producer | Where | Licence |
| --- | --- | --- | --- |
| AlphaEarth Foundations Satellite Embedding, annual, 2017 to 2025 | Google and Google DeepMind ([dataset page](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)) | [tge-labs/aef](https://source.coop/tge-labs/aef), [tge-labs/aef-mosaic](https://source.coop/tge-labs/aef-mosaic) | CC BY 4.0 |
| Overture Maps landcover (base theme, `land_cover`) | Overture Maps Foundation, from ESA WorldCover | [Overture release bucket](https://docs.overturemaps.org/guides/base/) | CC BY 4.0 |
| World Settlement Footprint (WSF) Tracker, 10 m, 2016 to 2026 | DLR and MindEarth | [mindearth/wsf](https://source.coop/mindearth/wsf) ([DOI 10.5281/zenodo.20424537](https://doi.org/10.5281/zenodo.20424537)) | CC BY 3.0 IGO |
| Sentinel-2 L2A, EOPF zarr (possible) | ESA, Copernicus | [EOPF Sentinel Zarr Samples](https://zarr.eopf.copernicus.eu/) | Copernicus open licence |

## Carried over

The notebooks from the earlier project are still here as a starting point
(`s2-wsf-aef-overture-pair.py`, `s2-wsf-aef-overture-slider.py`,
`s2-wsf-aef-overture-atlas.py`). They hold the AEF and WSF readers and the
H3 folding in DataFusion. They also read Overture buildings and the Earth
Genome Sentinel-2 mosaics, which are outside this repo's scope.

## Run

Dependencies are declared inline (PEP 723):

```
uv run marimo edit <notebook>.py --sandbox
```

## Attribution

AlphaEarth Foundations Satellite Embedding dataset by Google and Google
DeepMind (CC BY 4.0). WSF Tracker (c) DLR and MindEarth, via Source
Cooperative. Overture Maps landcover by the Overture Maps Foundation.
Contains modified Copernicus Sentinel data.
