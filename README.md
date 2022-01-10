# PreImutils
![PreImutils](docs/imgs/preimutils.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made in MRL](https://img.shields.io/badge/Made%20in-Mechatronic%20Research%20Labratories-red.svg)](https://www.qiau.ac.ir/)

PreImutils is a python library built to empower developers, reseachers and students to prepare and preprocessing image datasets for applications and systems with Deep Learning and Computer Vision capabilities using simple and few lines of code. This documentation is provided to provide detailed insight into all the classes and functions available in PreImutils, coupled with a number of code examples.

The Official GitHub Repository of PreImutils is https://github.com/mrl-amrl/preimutils

For read the full doc please visit   
https://a-sharifi.github.io/preimutils-doc/

## Easy to use:

## Why we need PreImutils?
Everything that you need to preprocess your image dataset is here.
One of the most important item for machine learning or CNN or other neural networks is preparing your dataset.

    - It's easy to use.
    - You can use both in terminal and code.
    - It has separate classes for object-detection and segmentation datasets.

## For Object detection
### Use in Code

```python
from preimutils.object_detection import AMRLImageAug

img_aug = AMRLImageAug(json_path, xmls_dir, images_dir)
img_aug.auto_augmentation(quantity, resized = True, width = 300, height = 300)
```
### Use in Terminal

```bash
JSON_PATH=~/YOUR_JSON_PATH/label.json
XMLS_DIR=~/YOUR_ANNOTATION_DIR/
IMAGES_DIR=~/YOUR_IMAGES_DIR/
FUNCTION=auto_augmentation
QUANTITY=1000 # the amount of each object to create

preimutils --function $FUNCTION --label_json_path $JSON_PATH --xmls_dir $XMLS_DIR --images_dir $IMAGES_DIR --quantity $QUANTITY
```
## For segmentation task

```python
from preimutils.segmentations.voc import Dataset
from preimutils.segmentations.voc import SegmentationAug


dataset = Dataset('./VOC2012', images_extention='jpg')

# First checking to have valid dataset(All dirs exist all mask's image exist or no)
dataset.check_valid_dataset()
seg = SegmentationAug(dataset.label_map_path,dataset.masks_dir,dataset.images_dir, images_extention='jpg')

# At the end, there must be 2000 of each object
seg.auto_augmentation(2000)
# Seprate dataset into validation 10% test 10% and train 80% and save it to train.txt,trainval.txt ,val.txt,test.txt
dataset.seprate_dataset(shuffle=True,valid_persent=0.10,test_persent=0.10,save=True)
```
## Some points


    1.  The amount of your dataset is really important. Not very few that lose the accuracy not great number of that lose your time and cause to overfitting, more than 4000 image per object is enough that mostly. depend on how much your feature is hard.
    2.  The amount of each object image is important if objects sample count not equal your neural network forget the lower object count for instance if you have 3 object each one should have 4000 sample.


    3. Don't forget to shuffle your dataset if you don't do that you never ever don't get good accuracy on all of your objects.
    4. If you want to detect your object from all angles don't forget to put sample from other angle
   
### attention
  - **No**:


      | object   | Sample Count |
      | -------- | :----------: |
      | object 1 |     2000     |
      | object 2 |     1000     |
      | object 3 |     4000     |
  

  - **Yes**:


      | object   | Sample Count |
      | -------- | :----------: |
      | object 1 |     3900     |
      | object 2 |     4100     |
      | object 3 |     4000     |
  

**PreImutils** help you to do these points in few line of code. 

## How should I use Preimutils?

We prepare a full documentation in https://a-sharifi.github.io/preimutils-doc/ .



