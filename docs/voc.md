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
## convert your custom dataset to VOC format

If you want to convert your custom dataset to VOC format use