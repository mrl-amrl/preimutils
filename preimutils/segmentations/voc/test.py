from dataset_info import Dataset,LabelMap
from count import export_path_count_for_each_label
from augment import SegmentationAug
if __name__ == "__main__":
    PATH = '/Users/amir/segmentations/hazmat-dataset/VOC2012/'
    dataset = Dataset(PATH)
    label_handler = LabelMap(PATH + '/labelmap.txt')
    # a = export_path_count_for_each_label(label_handler.color_label(),dataset.images_dir,dataset.segmentations_class_dir)
    seg_aug = SegmentationAug(PATH + '/labelmap.txt',dataset.segmentations_class_dir,dataset.images_dir)
    seg_aug.auto_augmentation(1000)