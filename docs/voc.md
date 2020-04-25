!!! warning
    **When use PreImutils VOC segmentation it's easier to put your data in below pattern**
```sh
.
└── ${YOUR_DATASET_DIR}
    ├── ImageSets
    │   └── Segmentation
    │       └── train.txt
    │       └── val.txt
    │       └── trainval.txt
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassRaw
    ├── SegmentationObject
    └── labelmap.txt


```
**If you have custom dataset just use below method**

```python
from preimutils.segmentations.voc import utils


utils.custom_to_voc('YOUR_MASKS_DIR','YOUR_IMAGES_DIR','TARGET_TO_SAVE_VOC_DS')
```

# preimutils.segmentation.voc.utils

## Modules 

::: preimutils.segmentations.voc.utils
    :docstring:
    :members:

# preimutils.segmentation.voc.Dataset

## Class

::: preimutils.segmentations.voc.Dataset
    :docstring:
    :members:
# preimutils.segmentation.voc.LabelMap

## Class

::: preimutils.segmentations.voc.LabelMap
    :docstring:
    :members:

# preimutils.segmentation.voc.SegmentationAug

## Class
::: preimutils.segmentations.voc.SegmentationAug
    :docstring:
    :members: