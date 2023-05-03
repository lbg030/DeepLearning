from ultralytics import YOLO

model = YOLO("yolov8l.pt")

result = model.predict(source="0", show=True)


print(result)