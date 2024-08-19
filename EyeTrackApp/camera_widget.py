import PySimpleGUI as sg  # 导入 PySimpleGUI 库，用于创建 GUI
from config import EyeTrackConfig  # 导入眼球追踪配置类
from config import EyeTrackSettingsConfig  # 导入眼球追踪设置配置类
from threading import Event, Thread  # 导入事件和线程类，用于多线程控制
from eye_processor import EyeProcessor, InformationOrigin  # 导入眼球处理器和信息源类型
from enum import Enum  # 导入枚举类
from queue import Queue, Empty  # 导入队列类，用于线程间通信
from camera import Camera, CameraState  # 导入摄像头类及其状态类型
from osc import EyeId  # 导入眼球ID枚举类，用于区分左右眼
import cv2  # 导入 OpenCV 库，用于图像处理
from winsound import PlaySound, SND_FILENAME, SND_ASYNC  # 导入声音播放函数，用于播放音效
import traceback  # 导入 traceback 模块，用于错误跟踪


'''
这个类 CameraWidget 是一个用于管理和控制眼球跟踪系统的核心组件。
它结合了图形用户界面(GUI)元素、摄像头捕获、图像处理以及事件驱动的操作，专门为眼球跟踪应用程序服务。


代码详解

1. 初始化方法 (__init__)

	•	GUI 元素定义：该方法定义了一系列 GUI 元素的唯一标识符,主要用于配置和显示眼球跟踪的相关参数。
            这些元素包括摄像头地址、阈值滑块、旋转滑块、ROI(感兴趣区域)选择和图像显示区域等。
	•	配置管理：通过 main_config 对象获取和设置左右眼的配置参数（如阈值、旋转角度等），
        并根据 widget_id（标识眼睛ID，左眼或右眼）决定当前小部件对应哪只眼睛。
	•	线程控制：初始化了一些用于线程控制的事件对象和队列，如 cancellation_event 和 capture_queue，
        这些对象用于管理线程的启动、停止和图像数据的传输。
	•	眼球处理器与摄像头初始化：初始化 EyeProcessor 和 Camera 对象，这两个对象分别负责眼球图像的处理和摄像头的图像捕获。

2. 图形用户界面布局定义

	•	ROI 模式布局:定义了一个用于选择感兴趣区域(ROI)的 GUI 布局，用户可以在图像上通过拖动选择区域。
	•	跟踪模式布局：包括阈值滑块、旋转滑块以及一些控制按钮，用户可以通过这些控件调整眼球跟踪的参数。
	•	总体布局：综合了 ROI 和跟踪模式的布局，提供了用户输入摄像头地址、切换模式和保存配置的界面。

3. 启动与停止方法

	•	started 方法：用于检查当前小部件是否已经启动，通过检查 cancellation_event 的状态来确定。如果该事件被设置（set），说明小部件已停止，否则处于运行状态。
	•	start 方法：启动眼球处理器和摄像头的线程。如果小部件未启动，清除 cancellation_event 事件并启动相应的处理线程。
	•	stop 方法：停止眼球处理器和摄像头的线程。通过设置 cancellation_event 来通知线程停止，并等待线程结束。

整体功能概述

CameraWidget 类主要用于管理眼球跟踪系统的 GUI 交互和图像处理流程。它提供了一个集成的界面，允许用户调整跟踪参数、切换不同的操作模式（如 ROI 选择与跟踪模式），并实时显示处理后的图像。通过线程控制，CameraWidget 实现了摄像头图像的捕获、处理和展示，同时确保了系统的响应性。

该类是眼球跟踪应用程序中的一个关键组件，结合了用户交互和图像处理的功能，确保系统能够准确地跟踪并展示眼睛的运动。

'''



class CameraWidget:
    def __init__(self, widget_id: EyeId, main_config: EyeTrackConfig, osc_queue: Queue):
        # 初始化 CameraWidget 类

        # 定义 GUI 元素的唯一标识符（key），这些标识符用于在 GUI 中引用这些元素
        self.gui_camera_addr = f"-CAMERAADDR{widget_id}-"
        self.gui_threshold_slider = f"-THREADHOLDSLIDER{widget_id}-"
        self.gui_rotation_slider = f"-ROTATIONSLIDER{widget_id}-"
        self.gui_roi_button = f"-ROIMODE{widget_id}-"
        self.gui_roi_layout = f"-ROILAYOUT{widget_id}-"
        self.gui_roi_selection = f"-GRAPH{widget_id}-"
        self.gui_tracking_button = f"-TRACKINGMODE{widget_id}-"
        self.gui_save_tracking_button = f"-SAVETRACKINGBUTTON{widget_id}-"
        self.gui_tracking_layout = f"-TRACKINGLAYOUT{widget_id}-"
        self.gui_tracking_image = f"-IMAGE{widget_id}-"
        self.gui_output_graph = f"-OUTPUTGRAPH{widget_id}-"
        self.gui_restart_calibration = f"-RESTARTCALIBRATION{widget_id}-"
        self.gui_recenter_eyes = f"-RECENTEREYES{widget_id}-"
        self.gui_mode_readout = f"-APPMODE{widget_id}-"
        self.gui_circular_crop = f"-CIRCLECROP{widget_id}-"
        self.gui_roi_message = f"-ROIMESSAGE{widget_id}-"

        # 保存传递进来的 OSC 队列，用于在不同线程之间传递数据
        self.osc_queue = osc_queue
        self.main_config = main_config  # 保存主配置对象
        self.eye_id = widget_id  # 保存当前小部件对应的眼睛 ID
        self.settings_config = main_config.settings  # 获取主配置中的设置对象

        # 根据眼睛 ID（左眼或右眼）设置对应的配置
        self.configl = main_config.left_eye      #左眼
        self.configr = main_config.right_eye    #右眼
        self.settings = main_config.settings    # 设置配置
        if self.eye_id == EyeId.RIGHT:
            self.config = main_config.right_eye
        elif self.eye_id == EyeId.LEFT:
            self.config = main_config.left_eye
        else:
            raise RuntimeError("Cannot have a camera widget represent both eyes!")  # 不能同时代表两只眼睛

        # 定义 ROI（感兴趣区域）模式下的 GUI 布局，包括一个用于选择区域的图形元素
        self.roi_layout = [
        [
            sg.Graph(
                (640, 480),  # 图形元素的尺寸，宽度为640像素，高度为480像素
                (0, 480),  # 图形坐标系的底部左角坐标，(x=0, y=480)
                (640, 0),  # 图形坐标系的顶部右角坐标，(x=640, y=0)
                key=self.gui_roi_selection,  # 为这个图形元素设置一个唯一标识符，用于事件处理
                drag_submits=True,  # 启用拖动提交事件，允许用户通过拖动在图形上选择区域
                enable_events=True,  # 启用事件处理，使得用户的操作能够触发事件回调
                background_color='#424042',  # 设置图形的背景颜色为深灰色
            )
        ]
    ]

        # 定义跟踪模式下的 GUI 布局，包括阈值滑块、旋转滑块、按钮等
        self.tracking_layout = [
        [
            # 创建一个显示“Threshold”的文本标签，背景颜色设置为深灰色
            sg.Text("Threshold", background_color='#424042'),   
            sg.Slider(
                range=(0, 110),  # 定义滑块的取值范围为0到110
                default_value=self.config.threshold,  # 设置滑块的默认值为配置中的阈值
                orientation="h",  # 将滑块设置为水平布局
                key=self.gui_threshold_slider,  # 为滑块设置唯一标识符，用于事件处理
                background_color='#424042'  # 设置滑块的背景颜色为深灰色
            ),
        ],
        [
            sg.Text("Rotation", background_color='#424042'),  # 创建一个显示“Rotation”的文本标签，背景颜色设置为深灰色
            sg.Slider(
                range=(0, 360),  # 定义滑块的取值范围为0到360度
                default_value=self.config.rotation_angle,  # 设置滑块的默认值为配置中的旋转角度
                orientation="h",  # 将滑块设置为水平布局
                key=self.gui_rotation_slider,  # 为滑块设置唯一标识符，用于事件处理
                background_color='#424042'  # 设置滑块的背景颜色为深灰色
            ),
        ],
        [
            sg.Button("Restart Calibration", key=self.gui_restart_calibration, button_color='#6f4ca1'),  # 创建一个重启校准的按钮，设置其颜色和唯一标识符
            sg.Button("Recenter Eyes", key=self.gui_recenter_eyes, button_color='#6f4ca1'),  # 创建一个重新校准眼睛位置的按钮，设置其颜色和唯一标识符
        ],
        [
            sg.Text("Mode:", background_color='#424042'),  # 创建一个显示“Mode:”的文本标签，背景颜色设置为深灰色
            sg.Text("Calibrating", key=self.gui_mode_readout, background_color='#424042'),  # 显示当前模式的文本标签，初始值为“Calibrating”，背景颜色设置为深灰色
            sg.Checkbox(
                "Circle crop:",  # 创建一个带有“Circle crop:”标签的复选框
                default=self.config.gui_circular_crop,  # 复选框的默认状态由配置决定
                key=self.gui_circular_crop,  # 为复选框设置唯一标识符，用于事件处理
                background_color='#424042',  # 设置复选框的背景颜色为深灰色
            ),
        ],
        [sg.Image(filename="", key=self.gui_tracking_image)],  # 创建一个图像显示区域，用于显示实时跟踪的图像
        [
            sg.Graph(
                (200, 200),  # 创建一个200x200像素的图形区域
                (-100, 100),  # 设置图形坐标系的范围（x轴从-100到100，y轴从-100到100）
                (100, -100),
                background_color="white",  # 设置图形的背景颜色为白色
                key=self.gui_output_graph,  # 为图形设置唯一标识符，用于事件处理
                drag_submits=True,  # 启用拖动提交事件
                enable_events=True,  # 启用事件处理
            ),
            sg.Text("Please set an Eye Cropping.", key=self.gui_roi_message, background_color='#424042', visible=False),  
            # 添加一个提示用户设置眼睛裁剪区域的文本标签，默认隐藏
        ],
    ]

        # 定义整个小部件的布局，包括摄像头地址输入、模式切换按钮以及布局切换
        self.widget_layout = [
            [
                sg.Text("Camera Address", background_color='#424042'),
                sg.InputText(self.config.capture_source, key=self.gui_camera_addr),
            ],
            [
                sg.Button("Save and Restart Tracking", key=self.gui_save_tracking_button, button_color='#6f4ca1'),
            ],
            [
                sg.Button("Tracking Mode", key=self.gui_tracking_button, button_color='#6f4ca1'),
                sg.Button("Cropping Mode", key=self.gui_roi_button, button_color='#6f4ca1'),
            ],
            [
                sg.Column(self.tracking_layout, key=self.gui_tracking_layout, background_color='#424042'),
                sg.Column(self.roi_layout, key=self.gui_roi_layout, background_color='#424042', visible=False),
            ],
        ]

        # 初始化用于线程控制的事件对象
        self.cancellation_event = Event()
        self.cancellation_event.set()  # 初始化时设置事件为 "set" 状态，以避免阻塞
        self.capture_event = Event()
        self.capture_queue = Queue()  # 创建用于存储捕获图像的队列
        self.roi_queue = Queue()  # 创建用于存储 ROI 模式图像的队列

        self.image_queue = Queue()  # 创建用于存储处理后图像的队列

        # 初始化眼球处理器，用于处理眼球图像
        self.ransac = EyeProcessor(
            self.config,
            self.settings_config,
            self.cancellation_event,
            self.capture_event,
            self.capture_queue,
            self.image_queue,
            self.eye_id,
        )

        # 初始化摄像头对象，用于管理摄像头的连接和图像捕获
        self.camera_status_queue = Queue()  # 创建用于存储摄像头状态的队列
        self.camera = Camera(
            self.config,
            0,
            self.cancellation_event,
            self.capture_event,
            self.camera_status_queue,
            self.capture_queue,
        )

        # 初始化鼠标位置和 ROI 模式的状态
        self.x0, self.y0 = None, None
        self.x1, self.y1 = None, None
        self.figure = None
        self.is_mouse_up = True
        self.in_roi_mode = False

    def started(self):
        # 检查是否已经启动，通过检查 cancellation_event 是否已设置来判断
        return not self.cancellation_event.is_set()

    def start(self):
        # 启动眼球处理器和摄像头的线程
        if not self.cancellation_event.is_set():
            return
        self.cancellation_event.clear()  # 清除事件，表示开始运行
        self.ransac_thread = Thread(target=self.ransac.run)  # 创建并启动眼球处理器线程
        self.ransac_thread.start()
        self.camera_thread = Thread(target=self.camera.run)  # 创建并启动摄像头线程
        self.camera_thread.start()

    def stop(self):
        # 停止线程的运行
        if self.cancellation_event.is_set():
            return
        self.cancellation_event.set()  # 设置事件，表示停止运行
        self.ransac_thread.join()  # 等待眼球处理器线程结束
        self.camera_thread.join()  # 等待摄像头线程结束