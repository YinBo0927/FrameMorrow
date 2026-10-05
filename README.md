<div align="center">

# FrameMorrow: Future-Guided Frame Selection with Prospective Tokens for Long-Horizon Video Generation

<p>
  <a href="https://arxiv.org/abs/2609.38839"><img src="https://img.shields.io/badge/arXiv-2609.38839-b31b1b.svg" alt="arXiv"></a>
  <a href="https://yinbo0927.github.io/FrameMorrow/"><img src="https://img.shields.io/badge/Project-Page-4d8199.svg" alt="Project page"></a>
  <img src="https://img.shields.io/badge/code-coming%20soon-lightgrey.svg" alt="Code coming soon">
</p>

<p>
  <a href="https://openreview.net/profile?id=~Bo_Yin2">Bo Yin</a>,
  <a href="https://openreview.net/profile?id=~Xiaobin_Hu1">Xiaobin Hu</a>,
  <a href="https://openreview.net/profile?id=~Jiaqi_Zhao3">Jiaqi Zhao</a>,
  <a href="https://openreview.net/profile?id=~Shuicheng_YAN3">Shuicheng Yan</a>
</p>

<p>Official repository for <strong>FrameMorrow</strong>. Code will be released soon.</p>

</div>

---

## Framework

<p align="center">
  <img src="assets/framemorrow-overview.png" alt="FrameMorrow inference and future-guided ranking distillation" width="900">
</p>

<p align="center">
  <em>FrameMorrow predicts prospective tokens to select historical frames for future generation. Explicit frames enable plug-and-play integration through each generator's existing conditioning interface.</em>
</p>

## Main Results

<p align="center"><strong>Long video generation on MovieGenBench.</strong></p>

<p align="center">
  <img src="assets/main-results-long-video.png" alt="VBench-Long results on 128 MovieGenBench prompts across three backbones" width="900">
</p>

<br>

<p align="center"><strong>Interactive video generation under single-shot and multi-shot conditions.</strong></p>

<p align="center">
  <img src="assets/main-results-interactive-video.png" alt="Interactive video generation results over 60 seconds" width="900">
</p>

<br>

<p align="center"><strong>Action-conditioned interactive world models.</strong></p>

<p align="center">
  <img src="assets/main-results-world-models.png" alt="Visual quality, temporal quality and action alignment across three world models" width="900">
</p>

## Citation

```bibtex
@article{yin2026framemorrow,
  title={FrameMorrow: Future-guided Frame Selection with Prospective Tokens for Long-Horizon Video Generation},
  author={Yin, Bo and Hu, Xiaobin and Zhao, Jiaqi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2609.38839},
  year={2026}
}
```

## Project Website

The website lives in [`docs/`](docs/). See its [README](docs/README.md) for local preview instructions.
