# Copyright (c) OpenMMLab. All rights reserved.
from mmtrack.datasets.youtube_vis_dataset import YouTubeVISDataset
from mmtrack.registry import DATASETS


@DATASETS.register_module()
class OVISDataset(YouTubeVISDataset):
    """OVIS dataset for video instance segmentation."""

    def __init__(self, dataset_version, *args, **kwargs):
        self.set_dataset_classes(dataset_version)
        super().__init__(dataset_version, *args, **kwargs)

    @classmethod
    def set_dataset_classes(cls, dataset_version):
        CLASSES_OVIS = ('Person', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep',
                        'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe',
                        'Poultry', 'Giant_panda', 'Lizard', 'Parrot', 'Monkey',
                        'Rabbit', 'Tiger', 'Fish', 'Turtle', 'Bicycle',
                        'Motorcycle', 'Airplane', 'Boat', 'Vehical')
        if dataset_version == 'ovis2021':
            cls.METAINFO = dict(CLASSES=CLASSES_OVIS)
        else:
            raise NotImplementedError('Not supported UVO dataset'
                                      f'version: {dataset_version}')
