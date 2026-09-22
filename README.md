# s2-wsf-aef-overture-pair

[![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/kentstephen/s2-wsf-aef-overture-pair/blob/main/s2-wsf-aef-overture-pair.py)

Settlement growth read two ways on one H3 grid, in a single marimo notebook.

Two maps share one camera. On the left, Earth Genome's Sentinel-2 yearly
mosaic (true colour, 2022 to 2025), rendered by the kernel from the COGs. On
the right, one H3 fill per hexagon from two independent records of change:

- **WSF Tracker** (DLR and MindEarth): the half-year each 10 m pixel was first
  observed as built-up. Per hexagon: the share built-up, the share that grew
  inside the window, and the year most of that new ground arrived.
- **AlphaEarth Foundations** (Google DeepMind): the mean 64-dimensional
  embedding per hexagon per year. How far it moved between the ends of the
  window (1 minus cosine similarity), and the first year it jumped past a
  quiet level set by the hexagons WSF says did not grow.

A click on a hexagon names its place from Overture Maps divisions, locality up
to country. Everything is read live from Source Cooperative; nothing is
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

## Attribution

WSF Tracker (c) DLR and MindEarth, via Source Cooperative (mindearth/wsf, DOI
10.5281/zenodo.20424537). The AlphaEarth Foundations Satellite Embedding
dataset is produced by Google and Google DeepMind (CC BY 4.0). Sentinel-2
yearly and temporal mosaics by Earth Genome from Copernicus Sentinel data (CC
BY 4.0). Overture Maps divisions (ODbL) via Source Cooperative
(cboettig/overturemaps, fused/overture). Place search by Photon (komoot) over
OpenStreetMap data (ODbL). Basemap by Carto.
