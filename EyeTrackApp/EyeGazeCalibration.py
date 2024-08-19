import numpy as np
from scipy.interpolate import griddata

'''
这段代码实现了一个基于插值方法的眼动标定系统，用于将眼动坐标转换为屏幕坐标。它的目的是通过收集若干已知的屏幕点和对应的眼动坐标，建立一个插值模型，然后通过该模型预测新的眼动坐标对应的屏幕位置。

1. **类的定义：`EyeGazeCalibration`**
   - **`__init__` 构造函数**:
     - 初始化标定系统，创建存储屏幕点和眼动点的列表，并定义插值网格的分辨率。
     - 使用 `np.mgrid` 创建了一个分辨率为 `resolution x resolution` 的网格，用于后续的插值计算。

   - **`add_calibration_point` 方法**:
     - 该方法用于添加标定数据，包括屏幕点和对应的眼动点。

   - **`calibrate` 方法**：
     - 这个方法将所有添加的标定数据转换为 NumPy 数组，然后使用 `scipy.interpolate.griddata` 函数进行插值。
     - 插值的目的是在给定的眼动坐标和屏幕坐标之间建立映射关系，从而可以预测新的眼动坐标对应的屏幕位置。
     - `self.interpolated_x` 和 `self.interpolated_y` 分别存储了在网格上插值后的 x 和 y 坐标。

   - **`predict` 方法**：
     - 给定一个新的眼动点，`predict` 方法会找到在插值网格中最接近的点，然后返回该点对应的屏幕坐标预测值。
     - 使用 `np.clip` 保证计算出的网格索引在有效范围内。

2. **主程序部分：`if __name__ == '__main__':`**
   - **创建 `EyeGazeCalibration` 实例**：
     - 实例化一个 `EyeGazeCalibration` 对象，用于执行眼动标定。

   - **添加标定点**：
     - 定义了一组屏幕点及其对应的眼动坐标，并通过 `add_calibration_point` 方法将这些点添加到标定系统中。

   - **执行标定**：
     - 调用 `calibrate` 方法进行插值计算，生成眼动坐标到屏幕坐标的映射模型。

   - **预测新的眼动坐标对应的屏幕点**：
     - 使用已标定好的模型，预测给定眼动坐标 `(0.4, 0.7)` 对应的屏幕点。
     - 打印预测结果。

### 代码用途
- 这个代码片段主要用于开发一个眼动追踪系统的标定部分，通过记录用户在屏幕上的注视点与相应的眼动坐标，建立一个模型。
- 之后，系统可以通过这个模型预测用户在其他未知点的注视位置，实现眼动数据与屏幕位置的映射。

**在代码中的应用**

在你的代码中，插值模型用于将一组离散的眼动坐标点映射到屏幕上的位置。具体来说：

- 通过 `griddata` 函数，插值模型根据你提供的眼动坐标和屏幕坐标的数据点，计算并生成一个函数或网格。
- 这个函数能够在任意新的眼动坐标点预测对应的屏幕坐标。

这种方法特别有用，因为它允许在没有直接测量或观测值的情况下，对未知点进行合理的估计，从而填补数据之间的空白。

'''


class EyeGazeCalibration:
    def __init__(self, resolution=200):
        # 初始化标定系统
        self.screen_points = []      # `screen_points` 存储屏幕上的标定点
        self.eye_gaze_points = []    # `eye_gaze_points` 存储相应的眼动点
        self.resolution = resolution        # `resolution` 定义插值网格的分辨率
        # `grid_x`, `grid_y` 创建了在0到1之间的网格，分辨率为200x200
        self.grid_x, self.grid_y = np.mgrid[0:1:complex(0, resolution), 0:1:complex(0, resolution)]

    def add_calibration_point(self, screen_point, eye_gaze_point):
        # 添加一个屏幕点和对应的眼动点到标定数据中
        self.screen_points.append(screen_point)
        self.eye_gaze_points.append(eye_gaze_point)

    def calibrate(self):
        # 将屏幕点和眼动点转换为 NumPy 数组
        self.screen_points = np.array(self.screen_points)
        self.eye_gaze_points = np.array(self.eye_gaze_points)
        
        # 使用 `griddata` 函数进行插值，将眼动点映射到屏幕点
        # `method='linear'` 表示使用线性插值
        self.interpolated_x = griddata(self.eye_gaze_points, self.screen_points[:, 0], (self.grid_x, self.grid_y), method='linear')
        self.interpolated_y = griddata(self.eye_gaze_points, self.screen_points[:, 1], (self.grid_x, self.grid_y), method='linear')

    def predict(self, eye_gaze_point):
        # 给定一个眼动点，预测它对应的屏幕点
        ex, ey = eye_gaze_point
        # 将眼动点坐标转换为插值网格上的索引值，并确保索引在有效范围内
        ix = np.clip(int(ex * self.resolution), 0, self.resolution - 1)
        iy = np.clip(int(ey * self.resolution), 0, self.resolution - 1)
        # 返回插值结果，即预测的屏幕点
        return (self.interpolated_x[ix, iy], self.interpolated_y[ix, iy])

'''
if __name__ == '__main__': 是 Python 中的一种常见结构，用来确保某些代码块只有在直接运行脚本时才会执行，而在该脚本被作为模块导入到其他脚本时，这些代码块不会执行。

	•	__name__: 在 Python 中，每个模块都有一个内置的属性 __name__。当模块被直接运行时，__name__ 的值被设置为 '__main__'。当模块被导入到其他模块中时，__name__ 的值是模块的实际名称（通常是文件名）。
	•	if __name__ == '__main__':: 这行代码检查当前模块的 __name__ 是否等于 '__main__'。如果是，表示该脚本是直接运行的，而不是被导入的，此时就会执行这个条件块中的代码。
'''

if __name__ == '__main__':
    # 示例使用
    calibration = EyeGazeCalibration()

    # 添加标定点及其对应的眼动坐标
    calibration_points = [
        ((0.0, 0.0), (0.0, 0.0)), ((0.25, 0.0), (0.2, 0.1)), ((0.5, 0.0), (0.4, 0.2)), ((0.75, 0.0), (0.6, 0.3)),
        ((1.0, 0.0), (0.8, 0.4)), ((0.0, 0.25), (0.1, 0.2)), ((0.25, 0.25), (0.1, 0.3)), ((0.5, 0.25), (0.3, 0.4)),
        ((0.75, 0.25), (0.5, 0.5)), ((1.0, 0.25), (0.7, 0.6)), ((0.0, 0.5), (0.2, 0.4)), ((0.25, 0.5), (0.0, 0.5)),
        ((0.5, 0.5), (0.2, 0.6)), ((0.75, 0.5), (0.4, 0.7)), ((1.0, 0.5), (0.6, 0.8)), ((0.0, 0.75), (0.3, 0.6)),
        ((0.25, 0.75), (0.1, 0.7)), ((0.5, 0.75), (0.1, 0.8)), ((0.75, 0.75), (0.3, 0.9)), ((1.0, 0.75), (0.5, 1.0)),
        ((0.0, 1.0), (0.4, 0.8)), ((0.25, 1.0), (0.2, 0.9)), ((0.5, 1.0), (0.0, 1.0)), ((0.75, 1.0), (0.2, 1.1)),
        ((1.0, 1.0), (0.4, 1.0)), ((0.1, 0.1), (0.1, 0.1)), ((0.9, 0.1), (0.9, 0.1)),
        ((0.1, 0.9), (0.1, 0.9)), ((0.9, 0.9), (0.9, 0.9)), ((0.5, 0.2), (0.5, 0.2))
    ]

    for screen_point, eye_gaze_point in calibration_points:
        # 将标定点添加到标定系统中
        calibration.add_calibration_point(screen_point, eye_gaze_point)

    # 进行标定
    calibration.calibrate()

    # 使用标定模型预测新的眼动坐标对应的屏幕点
    eye_gaze_point = (0.4, 0.7)
    predicted_screen_point = calibration.predict(eye_gaze_point)

    # 输出预测结果
    print(f"Predicted screen point for eye gaze {eye_gaze_point}: {predicted_screen_point}")