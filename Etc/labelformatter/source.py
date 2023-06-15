import sys
import os
import json
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QComboBox, QFileDialog, QMessageBox

from utils import labelme2yolo, labelme2hubble, yolo2labelme


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.combo = QComboBox(self)
        self.combo.addItem("Labelme2YOLO")
        self.combo.addItem("Labelme2Hubble")
        self.combo.addItem("YOLO2Labelme")
        
        # Add this line to reset label when combo selection changes
        self.combo.currentIndexChanged.connect(self.reset_label)

        self.lbl = QLabel('Not Executed', self)
        self.btn_dir = QPushButton('Select Directory', self)
        self.btn_dir.clicked.connect(self.select_directory)

        self.btn_format = QPushButton('Convert format', self)
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

    def reset_label(self):  # This function is called whenever the combo selection changes
        self.lbl.setText('Not Executed')

    def format_label(self):
        format_type = self.combo.currentText()
        # Initialize count dictionary
        converted_files = {cls: 0 for cls in self.class_list}

        if format_type == "Labelme2YOLO" or format_type == "Labelme2Hubble":
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

                        converted_files[dir_name] += 1

                # self.lbl.setText(f"Saved Complete")
                self.lbl.setText(
                    f"Conversion complete. Class-wise counts: {converted_files}")

            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
        else:
            try:
                for dir_name in self.class_list:
                    subdir_path = os.path.join(self.directory, dir_name)
                    txt_files = [f for f in os.listdir(
                        subdir_path) if f.endswith('.txt')]

                    for file_name in txt_files:
                        file_path = os.path.join(subdir_path, file_name)
                        # with open(file_path, 'r') as file:
                        #     data = json.load(file)

                        if format_type == "YOLO2Labelme":
                            formatted_data = self.format_to_labelme(
                                file_path, self.class_list)

                        converted_files[dir_name] += 1
                # self.lbl.setText(f"Saved Complete ")
                self.lbl.setText(
                    f"Conversion complete. Class-wise counts: {converted_files}")

            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def format_to_yolo(self, data, file_path, class_list):
        defect_label = labelme2yolo(data, file_path, class_list)

        return defect_label

    def format_to_hubble(self, data, file_path, class_list=None):
        defect_label = labelme2hubble(data, file_path, class_list)

        return defect_label

    def format_to_labelme(self, data, class_list):
        defect_label = yolo2labelme(data, class_list)

        return defect_label


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
