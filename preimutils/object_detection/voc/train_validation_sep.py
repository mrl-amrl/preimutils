import random
import os
import glob
import shutil
from tqdm import tqdm


def separate_test_val(images_dir, xmls_dir, dst_validatoion_dir, dst_train_dir, validation_percentage = 0.2):
    """
    Seperating Train and validation to their related directories
    Args:
        xmls_dir: all xmls files directory.
        images_dir: your images directory.
        dst_validatoion_dir:destination directory to save validations images and xmls after seperating
        dst_train_dir:destination directory to save train images and xmls after seperating
    Returns:
        No return
    """
    if not os.path.exists(dst_validatoion_dir):
        os.makedirs(dst_validatoion_dir)
    if not os.path.exists(dst_train_dir):
        os.makedirs(dst_train_dir)
    assert os.path.exists(images_dir),'images path not exist'
    assert os.path.exists(xmls_dir),'xmls path not exist'
    a = glob.glob(os.path.join(xmls_dir , '*.xml'))
    all_xmls = random.sample(a, len(a))
    validation_max_index = int(validation_percentage * len(all_xmls))
    validation_xmls_path = all_xmls[:validation_max_index]
    train_xmls_path = all_xmls[validation_max_index:]
    
    
    dst_validation_path_images = os.path.join(dst_validatoion_dir,'images')
    dst_validation_path_annotations = os.path.join(dst_validatoion_dir,'annotations')

    dst_train_path_images = os.path.join(dst_train_dir,'images')
    dst_train_path_annotations = os.path.join(dst_train_dir,'annotations')

    if not os.path.exists(dst_validation_path_images):
        os.mkdir(dst_validation_path_images)

    if not os.path.exists(dst_validation_path_annotations):
        os.mkdir(dst_validation_path_annotations)

    if not os.path.exists(dst_train_path_images):
        os.mkdir(dst_train_path_images)

    if not os.path.exists(dst_train_path_annotations):
        os.mkdir(dst_train_path_annotations)
    
    # # seprate tests
    for xml_path in tqdm(validation_xmls_path):
        xml_basename = os.path.basename(xml_path)
        jpg_basename = xml_basename[:-4] + '.jpg'
        shutil.copy2(xml_path, dst_validation_path_annotations)
        image_path = glob.glob(os.path.join(images_dir,jpg_basename))
        try:
            image_path = str(image_path[0])
        except IndexError:
            print("related image file {} not exist".format(os.path.join(images_dir,jpg_basename)))
        assert os.path.exists(image_path),'image file for {} not exist \n'.format(xml_path)
        shutil.copy2(image_path,dst_validation_path_images)

    # # seprate validations
    for xml_path in tqdm(train_xmls_path):
        xml_basename = os.path.basename(xml_path)
        jpg_basename = xml_basename[:-4] + '.jpg'
        shutil.copy2(xml_path,dst_train_path_annotations)
        image_path = glob.glob(os.path.join(images_dir,jpg_basename))
        image_path = str(image_path[0])
        assert os.path.exists(image_path),'image file for {} not exist \n'.format(xml_path)
        shutil.copy2(image_path, dst_train_path_images)
    