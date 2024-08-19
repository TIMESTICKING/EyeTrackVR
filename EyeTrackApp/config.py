from typing import Union  # 用于定义联合类型
import os.path  # 用于处理文件路径
import json  # 用于处理 JSON 数据
from pydantic import BaseModel  # 用于数据验证和管理
from osc import EyeId  # 导入 EyeId 类

# 配置文件名称常量
CONFIG_FILE_NAME: str = "eyetrack_settings.json"

# 定义相机配置类
class EyeTrackCameraConfig(BaseModel):
    threshold: int = 50  # 阈值
    rotation_angle: int = 0  # 旋转角度
    roi_window_x: int = 0  # ROI 窗口 X 坐标
    roi_window_y: int = 0  # ROI 窗口 Y 坐标
    roi_window_w: int = 0  # ROI 窗口宽度
    roi_window_h: int = 0  # ROI 窗口高度
    focal_length: int = 30  # 焦距
    capture_source: Union[int, str, None] = None  # 捕获源
    gui_circular_crop: bool = False  # GUI 圆形裁剪

# 定义设置配置类
class EyeTrackSettingsConfig(BaseModel):
    gui_flip_x_axis_left: bool = False  # 左眼 X 轴翻转
    gui_flip_x_axis_right: bool = False  # 右眼 X 轴翻转
    gui_flip_y_axis: bool = False  # Y 轴翻转
    gui_blob_fallback: bool = True  # Blob 回退
    gui_min_cutoff: str = "0.0004"  # 最小截止频率
    gui_speed_coefficient: str = "0.9"  # 速度系数
    gui_osc_address: str = "127.0.0.1"  # OSC 地址
    gui_osc_port: int = 9000  # OSC 端口
    gui_osc_receiver_port: int = 9001  # OSC 接收端口
    gui_osc_recenter_address: str = "/avatar/parameters/etvr_recenter"  # OSC 重新居中地址
    gui_osc_recalibrate_address: str = "/avatar/parameters/etvr_recalibrate"  # OSC 重新校准地址
    gui_blob_maxsize: float = 25  # Blob 最大尺寸
    gui_blob_minsize: float = 10  # Blob 最小尺寸
    gui_recenter_eyes: bool = False  # 重新居中眼睛
    gui_eye_falloff: bool = False  # 眼睛衰减
    tracker_single_eye: int = 0  # 单眼跟踪
    gui_blink_sync: bool = False  # 眨眼同步

# 定义眼动追踪配置类
class EyeTrackConfig(BaseModel):
    version: int = 1  # 版本
    right_eye: EyeTrackCameraConfig = EyeTrackCameraConfig()  # 右眼配置
    left_eye: EyeTrackCameraConfig = EyeTrackCameraConfig()  # 左眼配置
    settings: EyeTrackSettingsConfig = EyeTrackSettingsConfig()  # 设置配置
    eye_display_id: EyeId = EyeId.RIGHT  # 眼睛显示 ID

    @staticmethod
    def load():
        """加载配置"""
        if not os.path.exists(CONFIG_FILE_NAME):
            print("No settings file, using base settings")
            return EyeTrackConfig()
        with open(CONFIG_FILE_NAME, "r") as settings_file:
            return EyeTrackConfig(**json.load(settings_file))

    def save(self):
        """保存配置"""
        with open(CONFIG_FILE_NAME, "w+") as settings_file:
            json.dump(obj=self.dict(), fp=settings_file)

"""
代码分析

	1.	导入模块：导入需要的模块和类，包括 typing、os.path、json、pydantic 和 EyeId。
	2.	常量定义：
	    •	CONFIG_FILE_NAME:定义配置文件的名称。
	3.	EyeTrackCameraConfig 类：
	    •	这是一个用于相机配置的数据模型，包含各种配置参数。
	4.	EyeTrackSettingsConfig 类：
	    •	这是一个用于其他设置配置的数据模型，包含 GUI 和 OSC 相关的参数。
	5.	EyeTrackConfig 类：
	    •	这是主配置类，包含版本信息、左眼和右眼的相机配置，以及其他设置。
	    •	load 方法：用于从 JSON 文件加载配置。如果文件不存在，则返回默认配置。
	    •	save 方法：用于将当前配置保存到 JSON 文件。
"""