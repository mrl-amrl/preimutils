import os
import cv2


def read_image_and_annotation(image_path: str, label_path: str):
    assert os.path.exists(image_path), f'{image_path} does not exist'
    assert os.path.exists(label_path), f'{label_path} does not exist'

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    with open(label_path, "r") as f:
        labels = f.read().splitlines()

    label_list = []
    boxes = []
    for i in range(len(labels)):
        class_name, cx, cy, w, h = labels[i].split(' ')
        cx = float(cx)
        cy = float(cy)
        w = float(w)
        h = float(h)
        label_list.append(class_name)
        boxes.append([cx, cy, w, h])

    return img, label_list, boxes

def write_image_and_annotations(image_path, txt_path, image, bboxes, bbox_classes):
    cv2.imwrite(image_path, image)
    lines = []
    for i in range(len(bboxes)):
        lines.append(f"{bbox_classes[i]} {bboxes[i][0]} {bboxes[i][1]} {bboxes[i][2]} {bboxes[i][3]}")

    open(txt_path, 'w').write('\n'.join(lines))
