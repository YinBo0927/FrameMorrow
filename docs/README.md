# FrameMorrow project page

Static research project page. No build step or dependencies.

## Preview

```sh
cd docs
python3 -m http.server 8765
```

Open http://localhost:8765. Edit `index.html`, `style.css`, and `script.js`.

## GitHub Pages

In repository **Settings → Pages**, choose **Deploy from a branch**, then **main /docs**. `.nojekyll` enables plain static serving. All asset paths are relative and work at the existing project URL.

## Content

- Three synchronized native / FrameMorrow video comparisons: Causal Forcing, LongLive 2.0, and WorldMem.
- Conceptual introduction figure and editable-project overview export.
- Three expandable result tables: MovieGenBench, interactive video generation (aggregate metrics), and action-conditioned world models. Values were checked against Tables 1, 2 and 4 of the September 25 manuscript export (`ICLR2027_FrameMorrow.pdf`).
- Video sources: the user's curated FrameMorrow supplementary clips. Website copies are H.264 re-encodes at CRF 20 with fast-start metadata, original resolution and timing, and no audio.
- The introduction image is a conceptual illustration, not an experimental comparison.
- The title and evaluation scope follow the provided manuscript. The author list and affiliations were supplied by the user. The paper button links to https://arxiv.org/abs/2609.38839. The Code button links to the repository.

Run `python3 check_site.py` to check local assets and basic page structure. The shared player supports task switching, playback speed and synchronized scrubbing.
