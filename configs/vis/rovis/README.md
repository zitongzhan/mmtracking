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

<!-- ## Results and models of MaskTrack R-CNN on YouTube-VIS 2019 validation dataset

As mentioned in [Issues #6](https://github.com/youtubevos/MaskTrackRCNN/issues/6#issuecomment-502503505) in MaskTrack R-CNN, the result is kind of unstable for different trials, which ranges from 28 AP to 31 AP when using R-50-FPN as backbone.
The checkpoint provided below is the best one from two experiments.

|    Method    |    Base detector    |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) |  AP  | Config | Download |
| :-------------: | :-------------: | :-------------: | :-----: | :-----: | :------: | :------------: |:----:| :------: | :--------: |
| MaskTrack R-CNN |    Mask R-CNN    |    R-50-FPN     |  pytorch  |   12e    | 1.61        | -            | 30.2 | [config](masktrack_rcnn_r50_fpn_12e_youtubevis2019.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830.log.json) |
| MaskTrack R-CNN |    Mask R-CNN    |    R-101-FPN     |  pytorch  |   12e    |  2.27       | -            | 32.2 | [config](masktrack_rcnn_r101_fpn_12e_youtubevis2019.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2019/masktrack_rcnn_r101_fpn_12e_youtubevis2019_20211023_150038-454dc48b.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2019/masktrack_rcnn_r101_fpn_12e_youtubevis2019_20211023_150038.log.json) |
| MaskTrack R-CNN |    Mask R-CNN    |    X-101-FPN     |  pytorch  |   12e    | 3.69        | -            | 34.7 | [config](masktrack_rcnn_x101_fpn_12e_youtubevis2019.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2019/masktrack_rcnn_x101_fpn_12e_youtubevis2019_20211023_153205-fff7a102.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2019/masktrack_rcnn_x101_fpn_12e_youtubevis2019_20211023_153205.log.json) |

## Results and models of MaskTrack R-CNN on YouTube-VIS 2021 validation dataset

The checkpoint provided below is the best one from two experiments.

|    Method    |    Base detector    |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) |  AP  | Config | Download |
| :-------------: | :-------------: | :-------------: | :-----: | :-----: | :------: | :------------: |:----:| :------: | :--------: |
| MaskTrack R-CNN |    Mask R-CNN    |    R-50-FPN     |  pytorch  |   12e    | 1.61        | -            | 28.7 | [config](masktrack_rcnn_r50_fpn_12e_youtubevis2021.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2021/masktrack_rcnn_r50_fpn_12e_youtubevis2021_20211026_044948-10da90d9.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2021/masktrack_rcnn_r50_fpn_12e_youtubevis2021_20211026_044948.log.json) |
| MaskTrack R-CNN |    Mask R-CNN    |    R-101-FPN     |  pytorch  |   12e    | 2.27         | -            | 31.3 | [config](masktrack_rcnn_r101_fpn_12e_youtubevis2021.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2021/masktrack_rcnn_r101_fpn_12e_youtubevis2021_20211026_045509-3c49e4f3.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r101_fpn_12e_youtubevis2021/masktrack_rcnn_r101_fpn_12e_youtubevis2021_20211026_045509.log.json) |
| MaskTrack R-CNN |    Mask R-CNN    |    X-101-FPN     |  pytorch  |   12e    | 3.69         | -            | 33.5 | [config](masktrack_rcnn_x101_fpn_12e_youtubevis2021.py) | [model](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2021/masktrack_rcnn_x101_fpn_12e_youtubevis2021_20211026_095943-90831df4.pth) &#124; [log](https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_x101_fpn_12e_youtubevis2021/masktrack_rcnn_x101_fpn_12e_youtubevis2021_20211026_095943.log.json) | -->
