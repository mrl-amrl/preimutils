from dataset_info import Dataset
from utils import find_image_from_mask,find_maxmin_size_images
import glob
import os
from augment import aug
from coco_ds.coco_manager import COCOHandler

# ds = Dataset('/home/amir/segmentation/pascal_voc_seg/VOCdevkit/VOC2012')
# images = ds.images_dir
# masks = glob.glob(os.path.join(ds.segmentations_class_dir, '*.png'))[0:5]
# image = find_image_from_mask(masks[0],images)
# aug(image,masks[0])



if __name__ == "__main__":
    ann_file = '/home/amir/segmentation/hazmat-segmentation/annotations/small.json'
    images_dir = '/home/amir/segmentation/hazmat-segmentation/images'
    coco_handler = COCOHandler(ann_file)
    # coco_handler.get_image_path_from_id(images_dir,0)
    # a = coco_handler.coco.getImgIds()
    # b = coco_handler.coco.getAnnIds(0)
    # print(b)
    # a = coco_handler.get_last_image_id()
    # print(a)
    # coco_handler.augment_image(images_dir,0)
    
    # coco_handler.save_dataset()
    # coco_handler.ds_statistic()
    coco_handler.remove_image_from_ds(0)
    # print(find_maxmin_size_images(images_dir))
    # print(max([]))