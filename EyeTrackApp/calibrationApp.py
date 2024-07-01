import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import Qt

class PointDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super(PointDisplayWidget, self).__init__(parent)
        self.setGeometry(100, 100, 600, 600)
        self.points = []

    def paintEvent(self, event):
        painter = QPainter(self)
        self.draw_grid_points(painter)
        self.draw_custom_points(painter)

    def draw_grid_points(self, painter):
        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
        width, height = self.width(), self.height()
        step_x, step_y = width // 3, height // 3

        for i in range(4):
            for j in range(4):
                x, y = i * step_x, j * step_y
                painter.drawEllipse(x - 5, y - 5, 10, 10)


    def draw_custom_points(self, painter):
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        for point in self.points:
            painter.drawEllipse(point[0] - 5, point[1] - 5, 10, 10)

    def add_point(self, x, y):
        self.points.append((x, y))
        self.update()

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, 800, 800)
        self.point_widget = PointDisplayWidget(self)

        self.btn_add_point = QPushButton('Add Point', self)
        self.btn_add_point.clicked.connect(self.add_point)

        layout = QVBoxLayout()
        layout.addWidget(self.point_widget)
        layout.addWidget(self.btn_add_point)
        self.setLayout(layout)

    def add_point(self):
        # Example of adding a point at (150, 150)
        self.point_widget.add_point(150, 150)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
