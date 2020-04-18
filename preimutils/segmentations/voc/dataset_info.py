import os

class LabelMap:
    def __init__(self,label_map_path):
        # self.lines = []
        with open(label_map_path) as f:
            lines = f.readlines()
        self.lines = lines[1:]

    def label_color(self):
        label_map = {}
        for line in self.lines:
            line = line.split(':')
            color = tuple(line[1].split(','))
            color = (int(color[0]),int(color[1]),int(color[2]))
            label_map[line[0]] = color
        return label_map

    def color_label(self):
        label_map = {}
        for line in self.lines:
            line = line.split(':')
            color = tuple(line[1].split(','))
            color = (int(color[0]),int(color[1]),int(color[2]))
            label_map[line[0]] = color
            new_dic = {y:x for x,y in label_map.items()}
        return new_dic

    def color_list(self):
        colors = []
        for line in self.lines:
            line = line.split(':')
            color = tuple(line[1].split(','))
            color = (int(color[0]),int(color[1]),int(color[2]))
            colors.append(color)
        return colors
class Dataset:
    def __init__(self,dataset_dir):
        self.dataset_dir = dataset_dir
        self.segmentations_class_dir = os.path.join(dataset_dir,'SegmentationClass')
        self.images_dir = os.path.join(dataset_dir,'JPEGImages')
        self.segmentations_object_dir = os.path.join(dataset_dir,'SegmentationObject')
        self.label_map_path = os.path.join(dataset_dir,'labelmap.txt')

if __name__ == "__main__":
    PATH = '/Users/amir/segmentations/hazmat-dataset/VOC2012/'
    label_handler = LabelMap(PATH + '/labelmap.txt')
    print(label_handler.color_list())