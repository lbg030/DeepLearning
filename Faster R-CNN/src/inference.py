## Defect 500개 Normal 500개

import numpy as np
import cv2
import torch
import glob as glob

# 이미지 출력하게 해주는 라이브러리
# from google.colab.patches import cv2_imshow

from model import create_model

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=2).to(device)
model.load_state_dict(torch.load(
    '../outputs/model34.pth', map_location=device
))
model.eval()


threshold_list = []
detection_threshold = 0.75
folder_list = ['defect','normal']


def testing(folder):
    acc = 0
    DIR_TEST = f"test_data/{folder}"
    defect_test_image = glob.glob(f"{DIR_TEST}/*")
    print(f"Test instances: {len(defect_test_image)}")
    
    CLASSES = ['양품', 'crack',]
    
    for i in range(len(defect_test_image)):
        # get the image file name for saving output later on
        image_name = defect_test_image[i].split('/')[-1].split('.')[0]
        
        image = cv2.imread(defect_test_image[i])
        orig_image = image.copy()
        
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float64)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        flag = False

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        labels = outputs[0]['labels']

        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            
            if(len(list(boxes))) > 0:
                flag = True

                draw_boxes = boxes.copy()
                # get all the predicited class names
                pred_classes = [CLASSES[i] for i in labels.cpu().numpy()]

                # draw the bounding boxes and write the class name on top of it
                for j, box in enumerate(draw_boxes):
                    cv2.rectangle(orig_image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 0, 255), 2)
                    cv2.putText(orig_image, pred_classes[j], 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                                2, lineType=cv2.LINE_AA)
                    
        #이미지 보여주기
        # cv2_imshow(orig_image)
        cv2.waitKey(1)
        
        if folder == 'defect':
            if flag:
            # crack 출력
            # cv2.imwrite(f"test_predictions/{image_name}.jpg", orig_image)
                acc += 1

        elif folder == 'normal':
            if not flag:
                acc += 1
                
        print(f"Image {i+1} done...")
        print('-'*50)
    
    return acc

for folder in folder_list:
    threshold_list.append(testing(folder)) # defect, normal 순서


acc = sum(threshold_list) / 10
precision = (threshold_list[0] / (threshold_list[0] + (500 - threshold_list[1]))) * 100
recall = (threshold_list[0] / 500) * 100

print(f"acc = {acc}, precision = {precision}, recall = {recall}, threshold = {detection_threshold}")

