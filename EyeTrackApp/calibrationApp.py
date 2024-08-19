import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import Qt


'''
这段代码创建了一个简单的基于PyQt5的图形用户界面（GUI）应用程序。该应用程序的主要功能是显示一个3x3网格，并允许用户通过点击按钮在指定位置添加自定义的红色点。

### 主要功能
1. **显示网格**：
   - 应用程序窗口中显示了一个3x3的点阵网格。每个网格点是一个黑色的小圆圈。
   - 这些网格点是固定的，表示在窗口的各个位置上的交点。

2. **添加自定义点**：
   - 用户可以点击“Add Point”按钮，在指定的位置（代码中示例位置为(150, 150)）添加一个红色的自定义点。
   - 这些点是由用户动态添加的，添加的点会显示在窗口中，并且与网格点一起显示。

3. **绘图机制**：
   - 通过重写`paintEvent`方法实现绘图操作。`QPainter`对象用于在窗口小部件上绘制图形。
   - `draw_grid_points`方法绘制黑色的固定网格点，而`draw_custom_points`方法绘制用户添加的红色自定义点。

4. **布局管理**：
   - 应用程序的布局使用垂直布局管理器(`QVBoxLayout`)，将用于显示点的`PointDisplayWidget`和用于添加点的按钮`QPushButton`进行排列。

### 程序运行流程
1. **启动应用**：
   - 当程序启动时，创建并显示一个包含网格的窗口，同时在窗口下方有一个按钮用于添加点。
   
2. **添加点**：
   - 用户点击“Add Point”按钮时，程序会在指定的位置（示例中为(150, 150)）添加一个红色点，并触发窗口重绘，显示新添加的点。

3. **窗口更新**：
   - 每次添加新点时，`PointDisplayWidget`的小部件都会重新绘制，包括网格点和所有已添加的自定义点。

这个程序展示了如何使用PyQt5构建简单的GUI应用，并且演示了基本的绘图和事件处理功能。
'''


# 自定义的窗口小部件类，用于显示点阵
class PointDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super(PointDisplayWidget, self).__init__(parent)
        # 设置窗口小部件的初始大小和位置
        self.setGeometry(100, 100, 600, 600)
        # 初始化一个列表，用于存储用户添加的自定义点
        self.points = []

    # 重写paintEvent方法，处理窗口重绘事件
    def paintEvent(self, event):
        # 创建QPainter对象，用于绘制图形
        painter = QPainter(self)
        # 绘制网格点
        self.draw_grid_points(painter)
        # 绘制自定义添加的点
        self.draw_custom_points(painter)

    # 绘制3x3的网格点
    def draw_grid_points(self, painter):
        # 设置画刷为黑色，实心填充
        painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
        # 获取窗口的宽度和高度
        width, height = self.width(), self.height()
        # 计算每个网格单元的宽度和高度
        step_x, step_y = width // 3, height // 3

        # 使用双层循环遍历网格的每个交点
        for i in range(4):
            for j in range(4):
                # 计算当前网格点的坐标
                x, y = i * step_x, j * step_y
                # 在当前坐标处绘制一个圆点（网格点）
                painter.drawEllipse(x - 5, y - 5, 10, 10)

    # 绘制用户添加的自定义点
    def draw_custom_points(self, painter):
        # 设置画刷为红色，实心填充
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        # 遍历所有自定义点并绘制它们
        for point in self.points:
            painter.drawEllipse(point[0] - 5, point[1] - 5, 10, 10)

    # 添加一个新的自定义点，并触发重绘
    def add_point(self, x, y):
        # 将新的点添加到points列表中
        self.points.append((x, y))
        # 更新窗口以触发重绘
        self.update()

# 主窗口类，包含整个应用程序的布局和逻辑
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        # 设置主窗口的初始大小和位置
        self.setGeometry(100, 100, 800, 800)
        # 创建一个PointDisplayWidget实例，用于显示点
        self.point_widget = PointDisplayWidget(self)

        # 创建一个按钮，用于添加新的自定义点
        self.btn_add_point = QPushButton('Add Point', self)
        # 将按钮的点击事件连接到add_point方法
        self.btn_add_point.clicked.connect(self.add_point)

        # 使用垂直布局管理器排列PointDisplayWidget和按钮
        layout = QVBoxLayout()
        layout.addWidget(self.point_widget)
        layout.addWidget(self.btn_add_point)
        self.setLayout(layout)

    # 添加自定义点的事件处理函数
    def add_point(self):
        # 示例：在(150, 150)处添加一个新点
        self.point_widget.add_point(150, 150)

# 程序的主入口
if __name__ == "__main__":
    # 创建应用程序对象
    app = QApplication(sys.argv)
    # 创建并显示主窗口
    mainWindow = MainWindow()
    mainWindow.show()
    # 进入应用程序主循环
    sys.exit(app.exec_())