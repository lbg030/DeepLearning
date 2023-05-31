import json
from PIL import Image
from pathlib import Path
import sys


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