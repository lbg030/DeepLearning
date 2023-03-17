import os
import xml.etree.ElementTree as ET

def convert_annotation(xml_path, yolo_path):
    xml_dir = os.listdir(xml_path)
    for i, xml_file in enumerate(xml_dir):
        if xml_file.endswith(".xml"):
            with open(os.path.join(xml_path, xml_file)) as f:
                tree = ET.parse(f)
                root = tree.getroot()

                txt_file = xml_file[:-4] + ".txt"
                with open(os.path.join(yolo_path, txt_file), "w") as out_file:
                    for obj in root.iter("object"):
                        cls = obj.find("name").text
                        if cls != "With Helmet":
                            continue
                        xml_box = obj.find("bndbox")
                        b = (
                            float(xml_box.find("xmin").text),
                            float(xml_box.find("ymin").text),
                            float(xml_box.find("xmax").text),
                            float(xml_box.find("ymax").text),
                        )
                        w = int(root.find("size").find("width").text)
                        h = int(root.find("size").find("height").text)
                        bb = (b[0], b[1], b[2], b[3])
                        out_file.write(
                            "{} {} {} {} {}\n".format(0, bb[0] / w, bb[1] / h, bb[2] / w, bb[3] / h)
                        )

xml_path = "/Users/gwonsmpro/Downloads/archive/annotations"
yolo_path = "/Users/gwonsmpro/Downloads/archive/images"

convert_annotation(xml_path, yolo_path)