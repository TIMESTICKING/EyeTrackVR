import PySimpleGUI as sg
from config import EyeTrackSettingsConfig
from threading import Event, Thread
from eye_processor import EyeProcessor, InformationOrigin
from enum import Enum
from queue import Queue, Empty
from camera import Camera, CameraState
import cv2
from osc import EyeId


'''
这段代码实现了一个用于眼动追踪设置的用户界面（GUI）。它利用了 PySimpleGUI 库来创建窗口和交互元素，让用户能够调整一些配置参数，如眼动追踪的滤波参数、轴翻转、Blob检测等。

代码主要功能

	1.	初始化GUI元素和布局：
	•	使用了PySimpleGUI定义各种用户界面控件，如复选框、滑块和输入框，这些控件用于让用户修改眼动追踪的设置。
	•	每个控件都有一个唯一的键（例如self.gui_flip_x_axis_left），用于在事件处理时识别控件。
	2.	启动和停止事件处理：
	•	使用了Event对象（self.cancellation_event）来管理设置界面的启动和停止。
	•	start方法清除事件，表示开始处理用户输入，而stop方法设置事件，表示停止处理。
	3.	渲染窗口和处理事件：
	•	render方法用于检查用户是否在界面中修改了配置。如果修改了配置，就会更新内部配置并保存这些更改。
	•	例如，当用户修改了OSC端口、滤波器参数、Blob检测参数或其他设置时，render方法会捕获这些更改并相应地更新配置文件。

典型应用场景

	•	该代码适用于需要动态调整眼动追踪算法配置的应用场景。用户可以通过GUI修改各种参数，而不必直接操作配置文件或重新编译代码。这种设置界面可以方便开发人员或用户在不同环境或需求下快速调节眼动追踪的行为。

主要设置选项

	•	Flip Axis（轴翻转）: 用户可以选择是否翻转左眼、右眼的X轴或整体的Y轴。
	•	Blob检测: 允许用户设置Blob检测的最小和最大尺寸，并且可以开启Blob Fallback（当其他方法失败时使用Blob检测）。
	•	滤波参数: 用户可以调整最小频率截止和速度系数来影响信号滤波的行为。
	•	OSC设置: 用户可以配置OSC（开放声音控制）相关的地址、端口以及重新校准的地址。

这段代码本质上为用户提供了一个友好的界面来配置复杂的眼动追踪算法，使其能够适应不同的使用场景和需求。

'''

# SettingsWidget类，用于处理设置界面的交互和配置


class SettingsWidget:
    def __init__(self, widget_id: EyeId, main_config: EyeTrackSettingsConfig, osc_queue: Queue):
        """
        初始化设置窗口。
        
        :param widget_id: 窗口的唯一标识符，用于创建唯一的控件键。
        :param main_config: 主配置对象，包含当前的设置数据。
        :param osc_queue: 用于发送OSC消息的队列。
        """
        # 定义GUI元素的键，用于在事件处理时引用控件
        self.gui_flip_x_axis_left = f"-FLIPXAXISLEFT{widget_id}-"
        self.gui_flip_x_axis_right = f"-FLIPXAXISRIGHT{widget_id}-"
        self.gui_flip_y_axis = f"-FLIPYAXIS{widget_id}-"
        self.gui_general_settings_layout = f"-GENERALSETTINGSLAYOUT{widget_id}-"
        self.gui_osc_address = f"-OSCADDRESS{widget_id}-"
        self.gui_osc_port = f"-OSCPORT{widget_id}-"
        self.gui_osc_receiver_port = f"OSCRECEIVERPORT{widget_id}-"
        self.gui_osc_recenter_address = f"OSCRECENTERADDRESS{widget_id}-"
        self.gui_osc_recalibrate_address = f"OSCRECALIBRATEADDRESS{widget_id}-"
        self.gui_blob_fallback = f"-BLOBFALLBACK{widget_id}-"
        self.gui_blob_maxsize = f"-BLOBMAXSIZE{widget_id}-"
        self.gui_blob_minsize = f"-BLOBMINSIZE{widget_id}-"
        self.gui_speed_coefficient = f"-SPEEDCOEFFICIENT{widget_id}-"
        self.gui_min_cutoff = f"-MINCUTOFF{widget_id}-"
        self.gui_eye_falloff = f"-EYEFALLOFF{widget_id}-"
        self.gui_blink_sync = f"-BLINKSYNC{widget_id}-"

        # 主配置文件和当前设置
        self.main_config = main_config
        self.config = main_config.settings
        self.osc_queue = osc_queue

        # 定义窗口内容的布局
        self.general_settings_layout = [
            # X轴翻转设置
            [
                sg.Checkbox(
                    "Flip Left Eye X Axis",
                    default=self.config.gui_flip_x_axis_left,
                    key=self.gui_flip_x_axis_left,
                    background_color='#424042',
                ),
                sg.Checkbox(
                    "Flip Right Eye X Axis",
                    default=self.config.gui_flip_x_axis_right,
                    key=self.gui_flip_x_axis_right,
                    background_color='#424042',
                ),
            ],
            # Y轴翻转设置
            [
                sg.Checkbox(
                    "Flip Y Axis",
                    default=self.config.gui_flip_y_axis,
                    key=self.gui_flip_y_axis,
                    background_color='#424042',
                ),
            ],
            # 双眼衰减设置
            [
                sg.Checkbox(
                    "Dual Eye Falloff",
                    default=self.config.gui_eye_falloff,
                    key=self.gui_eye_falloff,
                    background_color='#424042',
                ),
            ],
            # 同步眨眼设置
            [
                sg.Checkbox(
                    "Sync Blinks (disables winking)",
                    default=self.config.gui_blink_sync,
                    key=self.gui_blink_sync,
                    background_color='#424042',
                ),
            ],
            # 跟踪算法设置标题
            [
                sg.Text("Tracking Algorithm Settings:", background_color='#242224'),
            ],
            # Blob fallback设置
            [
                sg.Checkbox(
                    "Blob Fallback",
                    default=self.config.gui_blob_fallback,
                    key=self.gui_blob_fallback,
                    background_color='#424042',
                ),
            ],
            # Blob大小设置
            [
                sg.Text("Min blob size:", background_color='#424042'),
                sg.Slider(
                    range=(1, 50),
                    default_value=self.config.gui_blob_minsize,
                    orientation="h",
                    key=self.gui_blob_minsize,
                    background_color='#424042'
                ),
                sg.Text("Max blob size:", background_color='#424042'),
                sg.Slider(
                    range=(1, 50),
                    default_value=self.config.gui_blob_maxsize,
                    orientation="h",
                    key=self.gui_blob_maxsize,
                    background_color='#424042'
                ),
            ],
            # 滤波器参数设置标题
            [
                sg.Text("Filter Parameters:", background_color='#242224'),
            ],
            # 最小频率截止设置
            [
                sg.Text("Min Frequency Cutoff", background_color='#424042'),
                sg.InputText(self.config.gui_min_cutoff, key=self.gui_min_cutoff),
            ],
            # 速度系数设置
            [
                sg.Text("Speed Coefficient", background_color='#424042'),
                sg.InputText(self.config.gui_speed_coefficient, key=self.gui_speed_coefficient),
            ],
            # OSC设置标题
            [
                sg.Text("OSC Settings:", background_color='#242224'),
            ],
            # OSC地址设置
            [
                sg.Text("OSC Address:", background_color='#424042'),
                sg.InputText(self.config.gui_osc_address, key=self.gui_osc_address),
            ],
            # OSC端口设置
            [
                sg.Text("OSC Port:", background_color='#424042'),
                sg.InputText(self.config.gui_osc_port, key=self.gui_osc_port),
            ],
            # OSC接收端口设置
            [
                sg.Text("OSC Receiver Port:", background_color='#424042'),
                sg.InputText(self.config.gui_osc_receiver_port, key=self.gui_osc_receiver_port),
            ],
            # OSC重新中心设置
            [
                sg.Text("OSC Recenter Address:", background_color='#424042'),
                sg.InputText(self.config.gui_osc_recenter_address, key=self.gui_osc_recenter_address),
            ],
            # OSC重新标定设置
            [
                sg.Text("OSC Recalibrate Address:", background_color='#424042'),
                sg.InputText(self.config.gui_osc_recalibrate_address, key=self.gui_osc_recalibrate_address),
            ]
        ]

        # 将整个设置窗口的布局定义为一个单独的列表
        self.widget_layout = [
            [
                sg.Text("General Settings:", background_color='#242224'),
            ],
            [
                sg.Column(self.general_settings_layout, key=self.gui_general_settings_layout, background_color='#424042'),
            ],
        ]

        # 初始化取消事件和图像队列
        self.cancellation_event = Event()  # 事件对象，用于控制窗口的启动和停止
        self.cancellation_event.set()  # 初始化时设置为“已设置”状态，表示窗口处于非活动状态
        self.image_queue = Queue()  # 图像队列，用于存储图像或数据，以便在其他线程中处理

    # 检查设置窗口是否已经启动
    def started(self):
        return not self.cancellation_event.is_set()

    # 启动设置窗口
    def start(self):
        # 如果已经在运行，则返回
        if not self.cancellation_event.is_set():
            return
        self.cancellation_event.clear()

    # 停止设置窗口
    def stop(self):
        # 如果尚未运行，则返回
        if self.cancellation_event.is_set():
            return
        self.cancellation_event.set()

    # 渲染窗口并处理事件
    def render(self, window, event, values):
        """
        检查设置窗口中的配置是否发生变化，并在变化时更新配置。
        
        :param window: PySimpleGUI窗口对象，用于显示设置界面。
        :param event: 触发的事件类型（如按钮点击）。
        :param values: 窗口中所有控件的当前值。
        """
        # 标记设置是否发生了变化
        changed = False

        # 检查OSC端口设置是否发生变化
        if self.config.gui_osc_port != values[self.gui_osc_port]:
            try:
                # 尝试将OSC端口值转换为整数
                int(values[self.gui_osc_port])
                if len(values[self.gui_osc_port]) <= 5:  # 检查端口值是否在有效范围内
                    self.config.gui_osc_port = int(values[self.gui_osc_port])
                    changed = True
                else:
                    print("[ERROR] OSC port value must be an integer 0-65535")
            except:
                print("[ERROR] OSC port value must be an integer 0-65535")

        # 检查OSC接收端口设置是否发生变化
        if self.config.gui_osc_receiver_port != values[self.gui_osc_receiver_port]:
            try:
                # 尝试将OSC接收端口值转换为整数
                int(values[self.gui_osc_receiver_port])
                if len(values[self.gui_osc_receiver_port]) <= 5:  # 检查端口值是否在有效范围内
                    self.config.gui_osc_receiver_port = int(values[self.gui_osc_receiver_port])
                    changed = True
                else:
                    print("[ERROR] OSC receive port value must be an integer 0-65535")
            except:
                print("[ERROR] OSC receive port value must be an integer 0-65535")

        # 检查OSC地址设置是否发生变化
        if self.config.gui_osc_address != values[self.gui_osc_address]:
            self.config.gui_osc_address = values[self.gui_osc_address]
            changed = True

        # 检查OSC重新中心地址设置是否发生变化
        if self.config.gui_osc_recenter_address != values[self.gui_osc_recenter_address]:
            self.config.gui_osc_recenter_address = values[self.gui_osc_recenter_address]
            changed = True

        # 检查OSC重新标定地址设置是否发生变化
        if self.config.gui_osc_recalibrate_address != values[self.gui_osc_recalibrate_address]:
            self.config.gui_osc_recalibrate_address = values[self.gui_osc_recalibrate_address]
            changed = True

        # 检查最小频率截止设置是否发生变化
        if self.config.gui_min_cutoff != values[self.gui_min_cutoff]:
            self.config.gui_min_cutoff = values[self.gui_min_cutoff]
            changed = True

        # 检查速度系数设置是否发生变化
        if self.config.gui_speed_coefficient != values[self.gui_speed_coefficient]:
            self.config.gui_speed_coefficient = values[self.gui_speed_coefficient]
            changed = True

        # 检查X轴右侧翻转设置是否发生变化
        if self.config.gui_flip_x_axis_right != values[self.gui_flip_x_axis_right]:
            self.config.gui_flip_x_axis_right = values[self.gui_flip_x_axis_right]
            changed = True

        # 检查X轴左侧翻转设置是否发生变化
        if self.config.gui_flip_x_axis_left != values[self.gui_flip_x_axis_left]:
            self.config.gui_flip_x_axis_left = values[self.gui_flip_x_axis_left]
            changed = True

        # 检查Y轴翻转设置是否发生变化
        if self.config.gui_flip_y_axis != values[self.gui_flip_y_axis]:
            self.config.gui_flip_y_axis = values[self.gui_flip_y_axis]
            changed = True

        # 检查Blob fallback设置是否发生变化
        if self.config.gui_blob_fallback != values[self.gui_blob_fallback]:
            self.config.gui_blob_fallback = values[self.gui_blob_fallback]
            changed = True

        # 检查双眼衰减设置是否发生变化
        if self.config.gui_eye_falloff != values[self.gui_eye_falloff]:
            self.config.gui_eye_falloff = values[self.gui_eye_falloff]
            changed = True

        # 检查同步眨眼设置是否发生变化
        if self.config.gui_blink_sync != values[self.gui_blink_sync]:
            self.config.gui_blink_sync = values[self.gui_blink_sync]
            changed = True

        # 检查Blob最大尺寸设置是否发生变化
        if self.config.gui_blob_maxsize != values[self.gui_blob_maxsize]:
            self.config.gui_blob_maxsize = values[self.gui_blob_maxsize]
            changed = True

        # 如果设置发生了变化，则保存新的设置
        if changed:
            self.main_config.save()
            
        # 将设置更新的消息放入OSC队列
        self.osc_queue.put((EyeId.SETTINGS))