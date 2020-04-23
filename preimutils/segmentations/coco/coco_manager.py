from pycocotools.coco import COCO
from glob import glob
import os
import pycocotools.mask as mask_utils
import json
import albumentations as A
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm


class COCOHandler:
    def __init__(self, annotations_path):
        self.coco = COCO(annotations_path)
        self.annotations_path = annotations_path
        self.aug_filter = [
            A.VerticalFlip(p=1),
            A.HorizontalFlip(p=1),
            A.RandomRotate90(p=1),
            A.ShiftScaleRotate(p=1),
            A.OneOf([A.HueSaturationValue(p=0.5),
                     A.RGBShift(p=0.7),
                     A.RandomBrightnessContrast(p=0.2)], p=1),
            A.PadIfNeeded(p=1, min_height=1024, min_width=1024),
            A.RandomRotate90(p=0.2),
            A.RandomShadow(p=0.1),
            A.RandomSnow(snow_point_lower=0.1,
                         snow_point_upper=0.15, p=0.1),
            A.RGBShift(p=0.2),
            A.CLAHE(p=0.2),

            A.HueSaturationValue(
                p=0.1, hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(p=0.2)]),
            A.ISONoise(p=0.2),
            A.Posterize(p=0.2),
            A.IAAPerspective(p=0.1),
            A.IAAPiecewiseAffine(p=0.1, scale=(0.01, 0.02)),
            A.IAAEmboss(p=0.2),
        ]
        self.aug = self.create_transformer(self.aug_filter)

    def add_aug_filter(self, aug_filter):
        self.aug_filter.append(aug_filter)
        self.aug = self.create_transformer(self.aug_filter)

    def pad_if_need_ds(self, images_dir, new_ds_path):
        img_ids = self.coco.getImgIds()
        for img_id in tqdm(img_ids):
            try:
                self.pad_if_need_image(images_dir, img_id)
            except ValueError as e : 
                print(e,img_id)
                continue
        self.save_dataset(new_ds_path)

    def visualize_by_image_id(self, images_dir ,image_id):
        image_info = self.coco.loadImgs(image_id)
        image_path = self.get_image_path_from_id(
            images_dir, image_id)
        anns_index = self.coco.getAnnIds(image_id)
        anns_info = self.coco.loadAnns(anns_index)
        all_points = []
        for ann in anns_info:
            ann['seg_len'] = len(ann['segmentation'][0])
            all_points.extend(ann['segmentation'][0])

        image, points = self.prepare_image_point(
            image_path, all_points)
        self.visualize_points(image, points)

    def pad_if_need_image(self, images_dir, image_id):
        image_info = self.coco.loadImgs(image_id)
        image_path = self.get_image_path_from_id(
            images_dir, image_id)
        anns_index = self.coco.getAnnIds(image_id)
        anns_info = self.coco.loadAnns(anns_index)
        all_points = []
        for ann in anns_info:
            ann['seg_len'] = len(ann['segmentation'][0])
            all_points.extend(ann['segmentation'][0])

        image, points = self.prepare_image_point(
            image_path, all_points)

        self.aug_filter = [A.PadIfNeeded(
            p=1, min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT)]
        self.aug = self.create_transformer(self.aug_filter)
        transformed = self.aug(image=image, keypoints=points)
        keypoints = transformed['keypoints']
        new_image = transformed['image']
        
        # remove old image and related annotations
        self.remove_image_from_ds(image_id)
        new_key_point = self.points_to_segmentations(keypoints)
        new_height, new_width, _ = new_image.shape
        new_image_id = image_info[0]['id']
        new_image_name = str(new_image_id) + '.jpg'
        self.add_image(new_width,
                       new_height,
                       new_image_id,
                       new_image_name
                       )
        start_idx = 0
        end_idx = 0
        new_seg = []
        for ann in anns_info:
            start_idx = start_idx + end_idx
            end_idx = end_idx + ann['seg_len']
            new_seg = new_key_point[start_idx:end_idx]
            try:
                new_area = self.segment_area([new_seg], new_width, new_height)
                new_bbox = self.segment_to_bbox(
                    [new_seg], new_width, new_height)
            except Exception:
                print("ERROR", ann)
            self.add_annotation(
                ann['id'],
                new_image_id,
                ann['category_id'],
                new_seg,
                float(new_area),
                list(new_bbox)
            )
        new_image_path = os.path.join(
            images_dir, "test", new_image_name)
        cv2.imwrite(new_image_path, new_image)

    def get_last_ann_id(self):
        last_id = 0
        ids = self.coco.getAnnIds()
        if len(ids):
            last_id = max(ids)
        return last_id

    def get_last_image_name(self):
        ann_id = max(self.coco.getImgIds())
        ann_info = self.coco.loadImgs(ann_id)
        name = ann_info[0]['file_name']
        return name

    def get_last_image_id(self):
        image_id = 0
        ids = self.coco.getImgIds()
        if len(ids):
            image_id = max(ids)
        return image_id

    def make_new_ds(self):
        return {
            "annotations": [],
            "categories": [],
            "licenses": [
                {
                    "id": 0,
                    "url": "",
                    "name": ""
                }
            ],
            "info": {
                "date_created": "",
                "description": "",
                "url": "",
                "year": "",
                "contributor": "",
                "version": ""
            },
            "images": []
        }

    def segment_area(self, segmentation, width, height):
        """get the area from coco annotation

        Args:

            segmentation: ann['segmentation'],
            width: image width,
            height: image height,

        Returns:

            area : area of segment points  -> float.
        """
        if type(segmentation) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif type(segmentation['counts']) == list:
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segmentation, height, width)

        # return rle
        area = mask_utils.area(rle)
        return area

    def segment_to_bbox(self, segmentation, width, height):
        """get the area from coco annotation

        Args:

            segmentation: ann['segmentation'],
            width: image width,
            height: image height,

        Returns:
        
            area : area of segment points  -> float.
        """
        if type(segmentation) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif type(segmentation['counts']) == list:
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segmentation, height, width)

        # return rle
        bbox = mask_utils.toBbox(rle)
        return bbox

    def get_image_path_from_id(self, images_dir, image_id):
        image_info = self.coco.loadImgs(image_id)
        image_name = image_info[0]['file_name']
        image_path = glob(os.path.join(images_dir, image_name))
        if not os.path.exists(image_path[0]):
            raise ValueError('image {} not exist in {} directory'.format(
                image_name, images_dir))
        return image_path[0]

    @staticmethod
    def segmentation_to_pair_points(segmentation):
        points = []
        for i in range(0, len(segmentation), 2):
            points.append(tuple(segmentation[i: i+2]))
        return points

    @staticmethod
    def points_to_segmentations(points):
        flat_list = []
        for sublist in points:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    def add_image(self, width, height, img_id, file_name, date_captured=0, flickr_url="", coco_url="", ds_license=0):
        """Add new image into your dataset

        Args:
            id: int -> image id should be unique,
            width: int,
            height: int,
            file_name: str,
            license: int,
            flickr_url: str,
            coco_url: str,
            date_captured: datetime,

        Returns:
            image path : path of the input mask  -> string.
        """

        d = {
            "width": width,
            "date_captured": date_captured,
            "flickr_url": flickr_url,
            "id": img_id,
            "file_name": file_name,
            "coco_url": coco_url,
            "license": ds_license,
            "height": height
        }
        self.coco.dataset['images'].append(d)
        self.coco.createIndex()

    def add_annotation(self, ann_id, img_id, cat_id, segmentation, area, bbox, iscrowed=0):
        """Add new annotation into your dataset

        Args:

            read more in http://cocodataset.org/#format-data
            ann_id: int -> annotation id (this should be unique can use get_last_id() + 1),
            image_id: int -> in which image you add new annotation,
            category_id: int ,
            segmentation: RLE or [polygon],
            area: float,
            bbox: [x, y, width, height],
            iscrowd: 0 or 1,

        Returns:
        
            image path : path of the input mask  -> string.
        """
        d = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "segmentation": [segmentation],
            "area": float(area),
            "bbox": list(bbox),
            "iscrowd": iscrowed,
        }
        self.coco.dataset['annotations'].append(d)
        self.coco.createIndex()

    def prepare_image_point(self, image_path, segment):
        img = cv2.imread(image_path)
        pair_points = self.segmentation_to_pair_points(segment)
        return img, pair_points

    def augment_image(self, images_dir, image_id):
        image_info = self.coco.loadImgs(image_id)
        image_path = self.get_image_path_from_id(
            images_dir, image_id)
        anns_index = self.coco.getAnnIds(image_id)
        anns_info = self.coco.loadAnns(anns_index)
        all_points = []
        for ann in anns_info:
            ann['seg_len'] = len(ann['segmentation'][0])
            all_points.extend(ann['segmentation'][0])

        image, points = self.prepare_image_point(
            image_path, all_points)
        # self.visualize_points(image,points)
        transformed = self.aug(image=image, keypoints=points)
        keypoints = transformed['keypoints']
        new_image = transformed['image']
        print(new_image.shape)
        # self.visualize_points(new_image, keypoints)

        new_key_point = self.points_to_segmentations(keypoints)
        new_height, new_width, _ = new_image.shape
        new_image_id = self.get_last_image_id() + 1

        self.add_image(new_width,
                       new_height,
                       new_image_id,
                       str(new_image_id) + '.jpg'
                       )
        start_idx = 0
        end_idx = 0
        new_seg = []
        for ann in anns_info:
            start_idx = start_idx + end_idx
            end_idx = end_idx + ann['seg_len']
            new_seg = new_key_point[start_idx:end_idx]
            new_area = self.segment_area([new_seg], new_width, new_height)
            new_bbox = self.segment_to_bbox([new_seg], new_width, new_height)
            self.add_annotation(
                self.get_last_ann_id() + 1,
                new_image_id,
                ann['category_id'],
                new_seg,
                float(new_area),
                list(new_bbox)
            )
        new_image_path = os.path.join(
            images_dir, "test", str(new_image_id))+".jpg"
        print(new_image_path)

    def save_dataset(self, path):
        with open(path, 'w') as dataset_file:
            json.dump(self.coco.dataset, dataset_file)


    def create_transformer(self, transformations):
        return A.Compose(transformations, p=1,
                         keypoint_params=A.KeypointParams(format='xy'))

    def visualize_points(self, image, points, diameter=15):
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        im = image.copy()
        for (x, y) in points:
            cv2.circle(im, (int(x), int(y)), diameter, (0, 255, 0), -1)
        cv2.imshow('img', im)
        cv2.waitKey(0)

    def ds_statistic(self):
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)
        statics = {}
        for cat in cats:
            statics[cat['name']] = 0

        for cat in tqdm(cats):
            cat_name = cat["name"]
            anns = self.coco.getAnnIds(catIds=cat["id"])
            statics[cat['name']] += len(anns)

        return statics

    def remove_image_from_ds(self, image_id):
        for image in self.coco.dataset['images']:
            if image["id"] == image_id:
                self.coco.dataset['images'].remove(image)
        image_anns = []
        for ann in tqdm(self.coco.dataset['annotations'], desc='finding annotations'):
            if ann['image_id'] == image_id:
                image_anns.append(ann)
        for ann in tqdm(image_anns, desc='removing annotations'):
            self.coco.dataset['annotations'].remove(ann)
        self.coco.createIndex()

