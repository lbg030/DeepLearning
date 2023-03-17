import os
import shutil

# 이미지와 텍스트 파일이 있는 폴더 경로
folder_path = "/Users/gwonsmpro/Downloads/archive/images"

# yolo 폴더 생성
yolo_path = "./yolo/"
if not os.path.exists(yolo_path):
    os.mkdir(yolo_path)

# 이미지와 텍스트 파일을 확인하고 yolo 폴더로 이동
for filename in os.listdir(folder_path):
    name, ext = os.path.splitext(filename)
    if ext == ".txt":
        txt_path = os.path.join(folder_path, filename)
        img_path = os.path.join(folder_path, name + ".png")  # 이미지 확장자에 맞게 수정
        yolo_txt_path = os.path.join(yolo_path, filename)
        yolo_img_path = os.path.join(yolo_path, name + ".png")  # 이미지 확장자에 맞게 수정
        if os.path.exists(img_path):
            shutil.copy(txt_path, yolo_txt_path)
            shutil.copy(img_path, yolo_img_path)
