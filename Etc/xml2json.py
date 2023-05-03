import xmltodict
import json

def xml_to_labelme(xml_path, output_path):
    # XML 파일 열기
    with open(xml_path) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())

    # JSON 파일로 변환
    json_data = json.loads(json.dumps(data_dict))

    # 이미지 정보 가져오기
    img_path = json_data["annotation"]["filename"]
    img_height = int(json_data["annotation"]["size"]["height"])
    img_width = int(json_data["annotation"]["size"]["width"])

    # Labelme 형식으로 변환
    labelme_data = {}
    labelme_data["version"] = "4.5.6"
    labelme_data["flags"] = {}
    labelme_data["imagePath"] = img_path
    labelme_data["imageData"] = None
    labelme_data["imageHeight"] = img_height
    labelme_data["imageWidth"] = img_width
    labelme_data["shapes"] = []

    for obj in json_data["annotation"]["object"]:
        shape = {}
        shape["label"] = obj["name"]
        shape["points"] = [[float(obj["bndbox"]["xmin"]), float(obj["bndbox"]["ymin"])], [float(obj["bndbox"]["xmax"]), float(obj["bndbox"]["ymax"])]]
        shape["group_id"] = None
        shape["shape_type"] = "rectangle"
        shape["flags"] = {}
        labelme_data["shapes"].append(shape)

    # JSON 파일로 저장
    with open(output_path, "w") as output_file:
        json.dump(labelme_data, output_file, indent=4)
