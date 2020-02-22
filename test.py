import os
from preimutils import LabelHandler, AMRLImageAug, label_checker, replace_label, cut_with_object_names, xml_address_changer
from preimutils import separate_with_label, gather_together, shuffle_img_xml,separate_test_val,separate_test_val,xml_csv_save


JSON_PATH = os.path.join('test','label.json')
XMLS_PATH = os.path.join('test','annotations')
IMAGES_DIR = os.path.join('test','images')
quantity = 10
img_aug = AMRLImageAug(JSON_PATH, XMLS_PATH, IMAGES_DIR)
img_aug.auto_augmentation(quantity)