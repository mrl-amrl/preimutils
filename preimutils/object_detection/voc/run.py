import argparse
import os
from .label_json import LabelHandler
from preimutils.object_detection import separate_with_label, gather_together, shuffle_img_xml, separate_test_val, separate_test_val, xml_csv_save
from .check_valid_label import label_checker, replace_label
from .img_aug import AMRLImageAug
from .rename_xml_path import xml_address_changer
from .crop_from_point import cut_with_object_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images_dir", required=False,
                    help="path to images file")
    ap.add_argument("-x", "--xmls_dir", required=False,
                    help="path to xmls file")
    ap.add_argument("-j", "--label_json_path", required=True,
                    help="path to label.json file")
    # For auto augment needed
    ap.add_argument("-q", "--quantity", required=False, type=int,
                    help="the amount that you want to each object to auto augment")
    ap.add_argument("-r", "--resize", required=False, default=False, type=bool,
                    help="If you want to resize your images set it true")
    ap.add_argument("-w", "--width", required=False, default=0, type=int,
                    help="If you set resize True set the width")
    ap.add_argument("-he", "--height", required=False, default=0, type=int,
                    help="If you set resize True set the height")
    # For replace_label only
    ap.add_argument("-l", "--label", required=False,
                    help="source label required to replace_label")
    # For replace_label only
    ap.add_argument("-d", "--dst_label", required=False,
                    help="destination label required to replace_label")
    # For cut with object_name & shuffle_image_xml
    ap.add_argument("-ds", "--dst_save", required=False,
                    help="destination path for save files")
    # For gathering together, train_validation_sep
    ap.add_argument("-p", "--dataset_dir", required=False,
                    help="path to your dataset file")
    # For train_validation_sep
    ap.add_argument("-vp", "--validation_persent", type=float, required=False,
                    help="percentage for validation from all of your dataset")

    ap.add_argument("-f", "--function", required=True,
                    choices=["auto_augmentation", "label_checker", "replace_label",
                             "cut_with_object_names", "separate_with_label",
                             "gather_together", "shuffle_img_xml",
                             "separate_test_val", "xml_to_csv", 'xml_address_changer'], help="function that you want to run")

    args = vars(ap.parse_args())
    # Auto augment
    quantity = args["quantity"]
    resize = args["resize"]
    width = args["width"]
    height = args["height"]
    # Other func
    images_dir = args['images_dir']
    xmls_dir = args["xmls_dir"]
    json_path = args['label_json_path']

    function = args["function"]
    dst_save = args["dst_save"]
    dataset_dir = args["dataset_dir"]
    # For replace_label only
    source_label = args["label"]
    # For replace_label only
    destination_label = args["dst_label"]
    assert os.path.exists(json_path), 'json file path not exist'
    label_handler = LabelHandler(json_path)
    persentage = args['validation_persent']
    # assert os.path.exists(images_dir), 'images_dir not exist'
    if function == "auto_augmentation":
        assert os.path.exists(xmls_dir), 'xmls path not exist'
        assert os.path.exists(images_dir), 'images path not exist'
        assert quantity > 0, 'Quantity < 0 not acceptable'
        img_aug = AMRLImageAug(json_path, xmls_dir, images_dir)
        img_aug.auto_augmentation(
            quantity, resize=resize, width=width, height=height)

    if function == "label_checker":
        assert os.path.exists(xmls_dir), 'xmls path not exist'
        label_array = label_handler.json_label_array()
        print("label array is ", label_array)
        label_checker(xmls_dir, label_array)

    if function == "replace_label":
        assert os.path.exists(xmls_dir), 'xmls path not exist'
        assert not destination_label == None, 'destinaton label param not exist'
        assert not source_label == None, 'source label param not exist'
        replace_label(xmls_dir, source_label, destination_label)

    if function == "cut_with_object_names":
        assert os.path.exists(xmls_dir), 'xmls path not exist'
        assert os.path.exists(images_dir), 'images path not exist'
        assert os.path.exists(dst_save), 'dst path not exist'
        labels_array = label_handler.json_label_array()
        cut_with_object_names(images_dir, xmls_dir, dst_save, labels_array)

    if function == "xml_address_changer":
        assert os.path.exists(xmls_dir), 'xmls path not exist'
        assert os.path.exists(images_dir), 'images path not exist'
        xml_address_changer(xmls_dir, images_dir)

    if function == "separate_with_label":
        assert os.path.exists(xmls_dir), 'xmls path not exist'
        assert os.path.exists(images_dir), 'images path not exist'
        labels_array = label_handler.json_label_array()
        separate_with_label(xmls_dir, images_dir, labels_array)

    if function == "gather_together":
        assert os.path.exists(dataset_dir), 'dataset path not exist'
        labels_array = label_handler.json_label_array()
        gather_together(label_array, dataset_dir)

    if function == "shuffle_img_xml":
        assert os.path.exists(xmls_dir), 'xmls path not exist'
        assert os.path.exists(images_dir), 'images path not exist'
        assert os.path.exists(dst_save), 'dst path not exist'
        labels_array = label_handler.json_label_array()
        shuffle_img_xml(xmls_dir, images_dir, dst_save)

    if function == "separate_test_val":
        assert os.path.exists(xmls_dir), 'xmls path not exist'
        assert os.path.exists(images_dir), 'images path not exist'
        assert persentage > 0, 'persentage LE than 0 not acceptable'
        train_path = os.path.join(dataset_dir, 'train')
        validation_path = os.path.join(dataset_dir, 'validation')
        separate_test_val(images_dir, xmls_dir,
                          validation_path, train_path, persentage)

    if function == "xml_to_csv":
        assert os.path.exists(xmls_dir), 'xmls path not exist'
        assert os.path.exists(dst_save), 'dst path not exist'
        xml_csv_save(xmls_dir, os.path.join(dst_save, 'dataframe.csv'))


if __name__ == "__main__":
    main()
