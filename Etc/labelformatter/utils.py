import json
from PIL import Image
from pathlib import Path
import sys
import cv2
import os


def labelme2hubble(json_data, file_name, class_list = None):
    boxes = json_data['shapes']
    hubble_set = list(dict() for _ in range(len(boxes)))
    for ind in range(len(boxes)):
        hubble_set[ind]['defectTypeName'] = boxes[ind]['label']
        ul = boxes[ind]['points'][0]
        br = boxes[ind]['points'][1]
        ur = [br[0], ul[1]]
        bl = [ul[0], br[1]]
        hubble_set[ind]['data'] = {'coordinateList' : [ul, ur, br, bl]}
    with open(file_name, 'w') as w_file:
        json.dump(hubble_set, w_file, indent = 4)

def labelme2yolo(data, file_path, class_list):
    def convert(label, size, box):
        c = label
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh

        st = str(c)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)
        return st

    defect_label = {class_list[i]: i for i in range(len(class_list))}
    txt_file_name = str(file_path)[:-4] + 'txt'
    path = str(file_path)[:-4] + 'png'

    if len(data['shapes']) > 1:
        st_list = []
        for i in range(len(data['shapes'])):
            label = defect_label[data['shapes'][i]['label']]
            points = data['shapes'][i]['points']

            xmin = min(points[0][0], points[1][0])
            ymin = min(points[0][1], points[1][1])
            xmax = max(points[0][0], points[1][0])
            ymax = max(points[0][1], points[1][1])

            im = Image.open(str(path))

            w = int(im.size[0])
            h = int(im.size[1])

            # print(xmin, xmax, ymin, ymax) #define your x,y coordinates
            b = (xmin, xmax, ymin, ymax)
            st = convert(label, (w, h), b)
            st_list.append(st)
            st_list.append('\n')

        with open(txt_file_name, "w") as f:
            f.writelines(st_list[:-1])
            f.close()

        # print(st)
    else:
        label = defect_label[(data['shapes'][0]['label'])]
        points = data['shapes'][0]['points']

        xmin = min(points[0][0], points[1][0])
        ymin = min(points[0][1], points[1][1])
        xmax = max(points[0][0], points[1][0])
        ymax = max(points[0][1], points[1][1])

        print(str(path))

        im = Image.open(str(path))

        w = int(im.size[0])
        h = int(im.size[1])

        # print(xmin, xmax, ymin, ymax) #define your x,y coordinates
        b = (xmin, xmax, ymin, ymax)
        st = convert(label, (w, h), b)

        with open(txt_file_name, "w") as f:
            f.write(st)
            f.close()

    return defect_label

def yolo2labelme(file_path, class_list):
    def convert_yolo_to_labelme(txt_file_path, img_file_path, labelme_file_path):
        with open(txt_file_path, 'r') as f:
            yolo_lines = f.readlines()

        label_dict = {class_list[i]: i for i in range(len(class_list))}
        # 이미지 파일을 읽어서 이미지 크기를 가져옵니다.
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape

        labelme_data = {"version": "4.5.7", "flags": {}, "shapes": [], "imagePath": "", "imageData": None}

        # LabelMe 데이터 구조에 이미지 크기를 설정합니다.
        labelme_data["imageWidth"] = width
        labelme_data["imageHeight"] = height

        # 이미지 파일명에서 .jpg 확장자를 제거한 이름을 이미지 경로로 설정합니다.
        img_file_name = os.path.splitext(os.path.basename(img_file_path))[0]
        labelme_data["imagePath"] = img_file_name + ".png"

        # YOLO 파일의 각 줄마다 반복합니다.
        for line in yolo_lines:
            label, x_center, y_center, box_width, box_height = line.strip().split(' ')

            label = [k for k, v in label_dict.items() if v == 0] 
            assert len(label) == 1, "label 길이가 1 넘음"
            # YOLO 형식의 좌표 값을 float형으로 변환합니다.
            x_center, y_center, box_width, box_height = float(x_center), float(y_center), float(box_width), float(box_height)

            # 좌표 값을 x1, y1, x2, y2로 변환합니다.
            x1 = ((x_center - box_width / 2) * width) 
            y1 = ((y_center - box_height / 2) * height)
            x2 = ((x_center + box_width / 2) * width)
            y2 = ((y_center + box_height / 2) * height)
            
            # LabelMe 모양 데이터 구조를 만듭니다.
            shape = {"label": label[0], "points": [[x1, y1], [x2, y2]], "group_id": None, "shape_type": "rectangle", "flags": {}}

            # LabelMe 데이터 구조의 모양 목록에 모양을 추가합니다.
            labelme_data["shapes"].append(shape)

        # LabelMe 데이터 구조를 .json 파일로 쓰기
        labelme_file_path = f"{labelme_file_path}"
        with open(labelme_file_path, 'w') as f:
            json.dump(labelme_data, f, indent = 4)

    name = file_path[:-4]
    txt_file_path = f'{name}.txt'
    img_file_path = f'{name}.png'
    labelme_file_path = f'{name}.json'
    convert_yolo_to_labelme(txt_file_path, img_file_path, labelme_file_path)


    