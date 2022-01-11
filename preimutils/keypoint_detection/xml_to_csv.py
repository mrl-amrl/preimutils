import pandas as pd

from utils import xml_to_dict


def xml_to_csv(xml_dir: str) -> pd.DataFrame:
    """

    :param xml_dir: Directory of xml file
    :return: Pandas DataFrame created from xml file
    """
    ann_dict = xml_to_dict(xml_dir)
    images_name = [ann_dict['annotations']['image'][i]['@name'] for i in range(len(ann_dict['annotations']['image']))]

    keypoints = []
    for i in range(len(ann_dict['annotations']['image'])):
        kp = []
        for p in range(len(ann_dict['annotations']['image'][i]['points'])):
            xy = ann_dict['annotations']['image'][i]['points'][p]['@points']
            kplabel = ann_dict['annotations']['image'][i]['points'][p]['@label']
            xy = xy.split(',')
            x = float(xy[0])
            y = float(xy[1])
            kp.append((x,y))
        keypoints.append(kp)

    data_dict = {}

    for i in range(len(images_name)):
        data_dict[images_name[i]] = keypoints[i]
    print(data_dict)
    #df = pd.DataFrame(data_dict)
    df = pd.DataFrame.from_dict(data_dict, orient='index',)
    return df


def xml_csv_save(xml_dir: str, save_dir: str) -> None:
    """
    :param xml_dir: Directory of xml file
    :param save_dir: Directory to save csv file
    """
    df = xml_to_csv(xml_dir)
    try:
        df.to_csv(save_dir)
        print(f'Saved: {save_dir}')
    except Exception as e:
        print(e)

xml_csv_save("C:\\Users\\Seyed\\Desktop\\keypoint\\data\\annotations.xml", "C:\\Users\\Seyed\\Desktop\\keypoint\\data\\annotations.csv")