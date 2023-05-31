import sys
import os
import json
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QComboBox, QFileDialog, QMessageBox

from utils import labelme2yolo, labelme2hubble


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.combo = QComboBox(self)
        self.combo.addItem("Labelme2YOLO")
        self.combo.addItem("Labelme2Hubble")

        # TODO: add other formats
        # self.combo.addItem("YOLO2Labelme")
        # self.combo.addItem("lens2YOLO")
        # self.combo.addItem("lens2labelme")
        # self.combo.addItem("lens2Hubble")

        self.lbl = QLabel('Label', self)
        self.btn_dir = QPushButton('Select Directory', self)
        self.btn_dir.clicked.connect(self.select_directory)

        self.btn_format = QPushButton('Format Labels', self)
        self.btn_format.clicked.connect(self.format_label)
        self.btn_format.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addWidget(self.combo)
        vbox.addWidget(self.lbl)
        vbox.addWidget(self.btn_dir)
        vbox.addWidget(self.btn_format)

        self.setLayout(vbox)

        self.setWindowTitle('Label Formatter')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def select_directory(self):
        self.directory = QFileDialog.getExistingDirectory(
            self, "Select Directory")
        if self.directory:
            self.btn_format.setEnabled(True)
            self.class_list = [name for name in os.listdir(
                self.directory) if os.path.isdir(os.path.join(self.directory, name))]

    def format_label(self):
        format_type = self.combo.currentText()

        try:
            for dir_name in self.class_list:
                subdir_path = os.path.join(self.directory, dir_name)
                json_files = [f for f in os.listdir(
                    subdir_path) if f.endswith('.json')]

                for file_name in json_files:
                    file_path = os.path.join(subdir_path, file_name)
                    with open(file_path, 'r') as file:
                        data = json.load(file)

                    if format_type == "Labelme2YOLO":
                        formatted_data = self.format_to_yolo(
                            data, file_path, self.class_list)

                    elif format_type == "Labelme2Hubble":
                        formatted_data = self.format_to_hubble(
                            data, file_path, self.class_list)

                    # elif format_type == "YOLO2Labelme":
                    #     formatted_data = self.format_to_labelme(
                    #         data, file_path, self.class_list)

                    # self.save_formatted_data(dir_name, file_name, formatted_data)

            self.lbl.setText(f"Saved Complete | Total : {len(json_files)}, Label : {str(formatted_data)}") if formatted_data else self.lbl.setText(
                f"Saved Complete | Total : {len(json_files)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def format_to_yolo(self, data, file_path, class_list):
        defect_label = labelme2yolo(data, file_path, class_list)

        return defect_label

    def format_to_hubble(self, data, file_path, class_list=None):
        defect_label = labelme2hubble(data, file_path, class_list)

        return defect_label

    # def format_to_labelme(self, data):
    #     return "Labelme Formatted: " + str(data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
