from utils import xml_to_dict, load_images_from_folder

def load_datas():
    xml_dir = 'C:\\Users\\Seyed\\Desktop\\pants-keypoint\\data\\annotations.xml'
    IMAGE_DIRECTORY = 'C:\\Users\\Seyed\\Desktop\\pants-keypoint\\data\\images'
    ann_dict = xml_to_dict(xml_dir)

    keypoint_names = [ann_dict['annotations']['meta']['task']['labels']['label'][i]['name'] for i in
                      range(len(ann_dict['annotations']['meta']['task']['labels']['label']))]
    colours = [ann_dict['annotations']['meta']['task']['labels']['label'][i]['color'] for i in
               range(len(ann_dict['annotations']['meta']['task']['labels']['label']))]

    images_name = [ann_dict['annotations']['image'][i]['@name'] for i in range(len(ann_dict['annotations']['image']))]
    images = load_images_from_folder(IMAGE_DIRECTORY, images_name)

    keypoints = []
    for i in range(len(ann_dict['annotations']['image'])):
        kp = []
        for p in range(len(ann_dict['annotations']['image'][i]['points'])):
            xy = ann_dict['annotations']['image'][i]['points'][p]['@points']
            xy = xy.split(',')
            x = float(xy[0])
            y = float(xy[1])
            kp.append((x, y))
        keypoints.append(kp)

    return images, keypoints, keypoint_names
