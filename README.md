# PreImutils
---
[![pipeline status](https://gitlab.com/mrl-amrl/preimutils/badges/master/pipeline.svg)](https://gitlab.com/mrl-amrl/preimutils/-/commits/master)
> AMRL lab utils for Pretrain your dataset specialy for PASCAL_VOC format

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made in MRL](https://img.shields.io/badge/Made%20in-Mechatronic%20Research%20Labratories-red.svg)](https://www.qiau.ac.ir/)
## Feature
- [# PreImutils](#h1-id%22preimutils-55%22preimutilsh1)
- [Feature](#feature)
- [Installation](#installation)
  - [Build from source](#build-from-source)
  - [Get from PyPI](#get-from-pypi)
- [Prepare Your dataset](#prepare-your-dataset)
- [Download dataset](#download-dataset)
- [Labeling](#labeling)
- [Rename image path in Annotations file](#rename-image-path-in-annotations-file)
  - [How to use in code](#how-to-use-in-code)
  - [How to use terminal](#how-to-use-terminal)
- [Checking for valid labels](#checking-for-valid-labels)
  - [label_checker](#labelchecker)
    - [How to use in code](#how-to-use-in-code-1)
    - [How to use terminal](#how-to-use-terminal-1)
  - [Replace label](#replace-label)
    - [How to use in code](#how-to-use-in-code-2)
    - [How to use terminal](#how-to-use-terminal-2)
- [Crop from point](#crop-from-point)
  - [How to use in code](#how-to-use-in-code-3)
  - [How to use terminal](#how-to-use-terminal-3)
- [Separate with label](#separate-with-label)
  - [How to use in code](#how-to-use-in-code-4)
  - [How to use terminal](#how-to-use-terminal-4)
- [Shuffle dataset images and annotations](#shuffle-dataset-images-and-annotations)
  - [How to use in code](#how-to-use-in-code-5)
  - [How to use shell](#how-to-use-shell)
- [Image augmentation](#image-augmentation)
  - [How to use in code](#how-to-use-in-code-6)
  - [How to use terminal](#how-to-use-terminal-5)
- [Train validate separator](#train-validate-separator)
  - [How to use in code](#how-to-use-in-code-7)
  - [How to use terminal](#how-to-use-terminal-6)
- [XML to csv converting](#xml-to-csv-converting)
  - [How to use terminal](#how-to-use-terminal-7)
- [Statistics of your Dataset labels](#statistics-of-your-dataset-labels)
  - [How to use in code](#how-to-use-in-code-8)

## Installation 
### Build from source
```sh 
git clone https://github.com/mrl-amrl/preimutils.git
cd preimutils
sudo pip3 install -r requirements.txt
```
### Get from PyPI
```sh
sudo pip3 install preimutils
```

## Prepare Your dataset

Everything that you need to preprocess your data is here.
One of the most important item for machine learning or CNN or other neural networks is preparing your dataset.
1.  The amount of your dataset is really important. Not very few that lose the accuracy not great number of that lose your time and cause to overfitting, more than 4000 image per object is enough that mostly. depend on how much your feature is hard.
2.  The amount of each object image is important if objects sample count not equal your neural network forget the lower object count for instance if you have 3 object each one should have 4000 sample.  
    - No:
        | object   | Sample Count |
        | -------- | :----------: |
        | object 1 |     2000     |
        | object 2 |     1000     |
        | object 3 |     4000     |
    
    - Yes :
        | object   | Sample Count |
        | -------- | :----------: |
        | object 1 |     3900     |
        | object 2 |     4100     |
        | object 3 |     4000     |

3. Don't forget to shuffle your dataset if you don't do that you never ever don't get good accuracy on all of your objects 
4. If you want to detect your object from all angles don't forget to put sample from other angle
First of all create a json file like this that contain all of your labels in a .json file to use this package

**When use this utils please put your data in this pattern**
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
from preimutils import xml_address_changer
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
from preimutils import label_checker
from preimutils import LabelHandler


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
from preimutils import replace_label,label_checker

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
from preimutils import cut_with_object_names

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
from preimutils import separate_with_label,gather_together
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
from preimutils as shuffle_img_xml
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
from preimutils import AMRLImageAug

img_aug = AMRLImageAug(json_path, xmls_dir, images_dir)
img_aug.auto_augmentation(quantity)
```
If you want to resize your images set resized param `True` and pass the `width` and `height` in parameters.

**Point**
As you know, if you use resize with other functions such as `cv2.resized()` your bounding box will be disarrange.

```python
from preimutils import AMRLImageAug

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
from preimutils import separate_test_val
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
from preimutils import export_path_count_for_each_label
export_path_count_for_each_label(xmls_dir, images_dir, labels)
```
