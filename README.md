# FrameMorrow project page

Static research project page. No build step or dependencies.

## Preview

```sh
python3 -m http.server 8765
```

Open http://localhost:8765. Edit `index.html`, `style.css`, and `script.js`.

## GitHub Pages

In repository **Settings → Pages**, choose **Deploy from a branch**, then **main / (root)**. `.nojekyll` enables plain static serving. All asset paths are relative and work at a project URL.

## Content

- Three synchronized native / FrameMorrow video comparisons: Causal Forcing, LongLive 2.0, and WorldMem.
- Conceptual introduction figure and editable-project overview export.
- Video sources: the user's curated FrameMorrow supplementary clips. Website copies are H.264 re-encodes at CRF 20 with fast-start metadata, original resolution and timing, and no audio.
- The introduction image is a conceptual illustration, not an experimental comparison.
- The title and evaluation scope follow the provided manuscript. The author list, affiliations and paper/PDF link were supplied by the user. The code button shows coming soon until the research implementation is released. This repository contains the project website, not the research implementation.

Run `python3 check_site.py` to check local assets and basic page structure. The shared player supports task switching, playback speed and synchronized scrubbing.
