# Robust Online Video Instance Segmentation with Track Queries

## Abstract

<!-- [ABSTRACT] -->

Recently, transformer-based methods have achieved impressive results on Video Instance Segmentation (VIS). However, most of these top-performing methods run in an offline manner by processing the entire video clip at once to predict instance mask volumes. This makes them incapable of handling the long videos that appear in challenging new video instance segmentation datasets like UVO and OVIS. We propose a fully online transformer-based video instance segmentation model that performs comparably to top offline methods on the YouTube-VIS 2019 benchmark and considerably outperforms them on UVO and OVIS. This method, called Robust Online Video Segmentation (ROVIS), augments the Mask2Former image instance segmentation model with track queries, a lightweight mechanism for carrying track information from frame to frame, originally introduced by the TrackFormer method for multi-object tracking. We show that, when combined with a strong enough image segmentation architecture, track queries can exhibit impressive accuracy while not being constrained to short videos.

<!-- [IMAGE] -->

<div align="center">
  <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRiYh6wm4ogny5sIOkK4ENQT76UHNq-NQ5IZZFG8aIxZupJC1h0bwc_2gGELlENEtjHsmiMNdVZeacc/embed?start=false&loop=true&delayms=3000" frameborder="0" width="948" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@misc{https://doi.org/10.48550/arxiv.2211.09108,
  doi = {10.48550/ARXIV.2211.09108},
  url = {https://arxiv.org/abs/2211.09108},
  author = {Zhan, Zitong and McKee, Daniel and Lazebnik, Svetlana},
  title = {Robust Online Video Instance Segmentation with Track Queries},
  publisher = {arXiv},
  year = {2022},
}
```

## Results and models of ROVIS on YouTube-VIS 2019 validation dataset

The checkpoint provided below is the best one from two experiments.

| Method | Base detector | Backbone |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AP  |                          Config                           |                                                                                          Download                                                                                           |
| :----: | :-----------: | :------: | :-----: | :-----: | :------: | :------------: | :-: | :-------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ROVIS  |  Mask2Former  |   R-50   | pytorch |   3e    |   5.71   |       -        |   45.8  | [config](rovis_mask2former_r50_2xb8_5e_youtubevis2019.py) | [model](https://drive.google.com/file/d/1J0T3IfDPk17HAkbIuntbkVWnKr4W1owb/view?usp=share_link) [log](https://drive.google.com/file/d/1xpk1xKWWpwVsB_P9SBunMJp11ktcl2zd/view?usp=share_link) |
