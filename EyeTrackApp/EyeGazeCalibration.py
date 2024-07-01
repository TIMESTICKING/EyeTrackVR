import numpy as np
from scipy.interpolate import griddata

class EyeGazeCalibration:
    def __init__(self, resolution=200):
        self.screen_points = []
        self.eye_gaze_points = []
        self.resolution = resolution
        self.grid_x, self.grid_y = np.mgrid[0:1:complex(0, resolution), 0:1:complex(0, resolution)]  # 创建网格进行插值

    def add_calibration_point(self, screen_point, eye_gaze_point):
        self.screen_points.append(screen_point)
        self.eye_gaze_points.append(eye_gaze_point)

    def calibrate(self):
        # 转换为numpy数组
        self.screen_points = np.array(self.screen_points)
        self.eye_gaze_points = np.array(self.eye_gaze_points)
        
        # 使用griddata进行插值
        self.interpolated_x = griddata(self.eye_gaze_points, self.screen_points[:, 0], (self.grid_x, self.grid_y), method='cubic')
        self.interpolated_y = griddata(self.eye_gaze_points, self.screen_points[:, 1], (self.grid_x, self.grid_y), method='cubic')

    def predict(self, eye_gaze_point):
        ex, ey = eye_gaze_point
        # 在插值网格中找到最近的点
        ix = np.clip(int(ex * self.resolution), 0, self.resolution - 1)
        iy = np.clip(int(ey * self.resolution), 0, self.resolution - 1)
        return (self.interpolated_x[ix, iy], self.interpolated_y[ix, iy])


if __name__ == '__main__':
        
    # 示例使用
    calibration = EyeGazeCalibration()

    # 添加16个均匀分布的标定点及对应的眼动坐标
    calibration_points = [
        ((0.0, 0.0), (0.0, 0.0)), ((0.25, 0.0), (0.2, 0.1)), ((0.5, 0.0), (0.4, 0.2)), ((0.75, 0.0), (0.6, 0.3)),
        ((1.0, 0.0), (0.8, 0.4)), ((0.0, 0.25), (-0.1, 0.2)), ((0.25, 0.25), (0.1, 0.3)), ((0.5, 0.25), (0.3, 0.4)),
        ((0.75, 0.25), (0.5, 0.5)), ((1.0, 0.25), (0.7, 0.6)), ((0.0, 0.5), (-0.2, 0.4)), ((0.25, 0.5), (0.0, 0.5)),
        ((0.5, 0.5), (0.2, 0.6)), ((0.75, 0.5), (0.4, 0.7)), ((1.0, 0.5), (0.6, 0.8)), ((0.0, 0.75), (-0.3, 0.6)),
        ((0.25, 0.75), (-0.1, 0.7)), ((0.5, 0.75), (0.1, 0.8)), ((0.75, 0.75), (0.3, 0.9)), ((1.0, 0.75), (0.5, 1.0)),
        ((0.0, 1.0), (-0.4, 0.8)), ((0.25, 1.0), (-0.2, 0.9)), ((0.5, 1.0), (0.0, 1.0)), ((0.75, 1.0), (0.2, 1.1)),
        ((1.0, 1.0), (0.4, 1.0))
    ]

    for screen_point, eye_gaze_point in calibration_points:
        calibration.add_calibration_point(screen_point, eye_gaze_point)

    # 标定
    calibration.calibrate()

    # 预测新的眼动坐标对应的屏幕点
    eye_gaze_point = (0.8, 0.4)
    predicted_screen_point = calibration.predict(eye_gaze_point)

    print(f"Predicted screen point for eye gaze {eye_gaze_point}: {predicted_screen_point}")
