
![Bounding Box](imgs/object_detection.jpg)
## Prepare Your dataset

!!! warning
    **When use PreImutils object detection please put your data in this pattern**
```sh
+dataset
    -label.json
    +images
    +annotations
```
**sample of label.json file**

```json
{
    "1": "object1",
    "2": "object2",
    "3": "object3",
    "4": "object4",
    "5": "object5",
    "6": "object6",
    "7": "object7",
    "8": "object8",
    "9": "object9",
    "10": "object10",
    "11": "object11",
    "12": "object12",
    "13": "object13"
}
```
## Download dataset 
For downloading your dataset I suggest you to use  [google_images_download](https://github.com/hardikvasa/google-images-download)  package easy to use 
`pip install google_images_download`
## Labeling

>**For labeling (bounding box) I suggest you to use  [labelImg](https://github.com/tzutalin/labelImg) and suggest to label in PASCAL_VOC mode because you can easily work on in and convert to coco and YOLO.**

## Convert to Yolo
**For converting to YOLO use [convert2Yolo](https://github.com/ssaru/convert2Yolo)**

## Rename image path in Annotations file
When You move your dataset files from some place to another you need to change image path in the .xml file
this function find the related image file in your image path and replace the annotation path

**after moving**
```xml
<annotation>
    <folder>voc2012</folder>
    <filename>000001</filename>
    <path>~/old_path/000001.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>353</width>
        <height>500</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>sample_object</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>129</xmin>
            <ymin>31</ymin>
            <xmax>298</xmax>
            <ymax>227</ymax>
        </bndbox>
    </object>
</annotation>
```

### How to use in code
```python
from preimutils.object_detection import xml_address_changer
xml_address_changer(xmls_dir, images_dir)
```

### How to use terminal
```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
IMAGES_DIR=~/YOUR_IMAGES_DIR/
FUNCTION=xml_address_changer

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR --images_dir $IMAGES_DIR
```


## Checking for valid labels

### label_checker
Imaging your labeling team label some wrong label on your picture you can find with this function and replace with the correct one with [Replace label](#replace-label)
#### How to use in code
```python
from preimutils.object_detection import label_checker
from preimutils.object_detection import LabelHandler


label = LabelHandler(json_path)
classes_array = label.json_label_array()
# or you can Prepare it manually
# if there were problem print the file and wrong label for and write the statistic of each label in another json file
label_checker(xmls_dir,classes_array)
replace_label(xmls_dir,from_label,dst_label)
```

#### How to use terminal
```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
IMAGES_DIR=~/YOUR_IMAGES_DIR/
FUNCTION=label_checker

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR  --images_dir $IMAGES_DIR
```

### Replace label 
Imaging your labeling team label some wrong label on your picture you can replace it with the currect one.
for instance: instead of object some of label write object.
#### How to use in code
```python
from preimutils.object_detection import replace_label,label_checker

label = LabelHandler(json_path)
classes_array = label.json_label_array()
# or you can Prepare it manually
# if there were problem print the file and wrong label for and write the statistic of each label in another json file
label_checker(xmls_dir,classes_array)
replace_label(xmls_dir,from_label,dst_label)
```

#### How to use terminal
```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
DST_LABEL=YOUR_CORRECT_LABEL
SOURCE_LABEL=YOUR_WRONG_LABEL
FUNCTION=replace_label

preimutils --function $FUNCTION --label_json_path $JSON_PATH  --xmls_dir $XMLS_DIR --label $SOURCE_LABEL --dst_label $DST_LABEL
```

## Crop from point
Some time you label your images but you need the pure images for instance for training haarcascade method with OpenCV or training simple convolutional neural network
this function crop images with their bbox and separating to their related object name.

### How to use in code
```python
from preimutils.object_detection import cut_with_object_names

if __name__ == "__main__":
  cut_with_object_names(images_dir, xmls_dir,dst_save, labels)
```

### How to use terminal

```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
IMAGES_DIR=~/YOUR_IMAGES_DIR/
DST_SAVE=~/YOUR_DESTINATION_DIR/
FUNCTION=cut_with_object_names

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR --dst_save $DST_SAVE --images_dir $IMAGES_DIR
```

## Separate with label
Separate images and their related annotations files on their object name file
after working on your separated dataset you can gather all of them together with `gather_together(label_array, DATASET_PATH)`

**after run**
```sh
+dataset
    -label.json
    +images
    +annotations
    +object
      +images
      +annotations
    +object2
      +images
      +annotations
    +objectN
      +images
      +annotations
```


### How to use in code
```python
from preimutils.object_detection import separate_with_label,gather_together
separate_with_label(XML_PATH, IMAGE_PATH, label_array)
# after working on your separated dataset you can gather all of them together
gather_together(label_array, DATASET_PATH)
```
### How to use terminal
```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
IMAGES_DIR=~/YOUR_IMAGES_DIR/
FUNCTION=separate_with_label

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR --images_dir $IMAGES_DIR
```

## Shuffle dataset images and annotations
One of the easiest but **the most important** point for pretraining, if you don't shuffle your images and their annotation your neural network won't get accurate result.

This function shuffling images and their related annotations and save in the destination directory

### How to use in code
```python
from preimutils.object_detection as shuffle_img_xml
shuffle_img_xml(XMLS_DIR, IMAGES_DIR, DST_PATH)
```
### How to use shell
```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
IMAGES_DIR=~/YOUR_IMAGES_DIR/
DST_SAVE=~/YOUR_DESTINATION_DIR/
FUNCTION=shuffle_img_xml

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR --images_dir $IMAGES_DIR --dst_save $DST_SAVE
```

## Image augmentation
One of the most important item for machine learning, CNN or other neural networks is augmenting your dataset in different situations.

I've used all the augmentations method and highly recommend you to use this package [albumentations]('https://github.com/albumentations-team/albumentations').

Write a good wrapper for this package with the best filter of this package that calculate the number of each object and then augment all of them in amount that you want.

### How to use in code
```python
from preimutils.object_detection import AMRLImageAug

img_aug = AMRLImageAug(json_path, xmls_dir, images_dir)
img_aug.auto_augmentation(quantity)
```
If you want to resize your images set resized param `True` and pass the `width` and `height` in parameters.

**Point**
As you know, if you use resize with other functions such as `cv2.resized()` your bounding box will be disarrange.

```python
from preimutils.object_detection import AMRLImageAug

img_aug = AMRLImageAug(json_path, xmls_dir, images_dir)
img_aug.auto_augmentation(quantity, resized = True, width = 300, height = 300)
```
### How to use terminal

1. Without resize
```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
IMAGES_DIR=~/YOUR_IMAGES_DIR/
FUNCTION=auto_augmentation
QUANTITY=1000 # the amount of each object to create

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR --images_dir $IMAGES_DIR --quantity $QUANTITY
```
2. Resize
```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
IMAGES_DIR=~/YOUR_IMAGES_DIR/
FUNCTION=auto_augmentation
QUANTITY=1000 # the amount of each object to create
RESIZE=True # If you want to resize You should set WIDTH and WIDTH param
WIDTH=300
HEIGHT=300

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR --images_dir $IMAGES_DIR --quantity $QUANTITY --resize $RESIZE --width $WIDTH --height $HEIGHT
```

## Train validate separator
We provide tool for separating your images for train and validation dataset with their related annotations.
### How to use in code
```python
from preimutils.object_detection import separate_test_val
separate_test_val(IMAGES_DIR,XMLS_DIR,DST_VALIDATION_PATH,DST_TRAIN_PATH,validation_percentage=0.3)
```
### How to use terminal
```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
IMAGES_DIR=~/YOUR_IMAGES_DIR/
DATASET_PATH=~/DATASET_PATH/
VALIDATION_PERSENT=0.3
FUNCTION=separate_test_val

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR --images_dir $IMAGES_DIR --dataset_dir $DATASET_PATH --validation_persent $VALIDATION_PERSENT
```

## XML to csv converting
This function convert pascal-voc format to csv



### How to use terminal
```sh
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
DST_SAVE=~/YOUR_DESTINATION_DIR/
VALIDATION_PERSENT=0.3
FUNCTION=xml_to_csv

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR --dst_save $DST_SAVE
```

## Statistics of your Dataset labels
Get statistics of dataset with their labels with their xmls and images files path
return
```json
        dict{   "label1": {
            "count":0,
            "xmls_paths":[],
            "images_paths":[],
                    },
        "labelN": {
            "count":0,
            "xmls_paths":[],
            "images_paths":[],
                    },
        }
```
### How to use in code
```python
from preimutils.object_detection import export_path_count_for_each_label
export_path_count_for_each_label(xmls_dir, images_dir, labels)
```
