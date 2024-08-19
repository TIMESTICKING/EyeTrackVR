from config import EyeTrackConfig  # 导入EyeTrackConfig配置类
from enum import Enum  # 导入枚举类型
import threading  # 导入线程模块
import queue  # 导入队列模块
import cv2  # 导入OpenCV库，用于图像处理

'''
这个Camera类的主要功能是:

	1.	连接到指定的摄像头。
	2.	持续从摄像头获取图像。
	3.	将获取到的图像推送到队列中，以供其他部分的程序进行进一步处理。

使用了线程事件(threading.Event)来控制图像的捕获过程，
并通过队列(queue.Queue)来传递摄像头的状态和捕获的图像数据。
代码中的异常处理机制可以应对摄像头连接问题并尝试重新连接。

'''



# 设置等待时间常量
WAIT_TIME = 0.1

# 定义CameraState枚举类，用于表示摄像头的状态
class CameraState(Enum):
    CONNECTING = 0  # 正在连接
    CONNECTED = 1  # 已连接
    DISCONNECTED = 2  # 已断开连接

# 定义Camera类，用于处理摄像头捕获
class Camera:
    def __init__(
        self,
        config: EyeTrackConfig,  # 传入的配置类实例
        camera_index: int,  # 摄像头的索引
        cancellation_event: "threading.Event",  # 取消事件，用于停止线程
        capture_event: "threading.Event",  # 捕获事件，用于控制图像捕获
        camera_status_outgoing: "queue.Queue[CameraState]",  # 输出摄像头状态的队列
        camera_output_outgoing: "queue.Queue",  # 输出捕获图像的队列
    ):
        # 初始化摄像头状态为连接中
        self.camera_status = CameraState.CONNECTING
        self.config = config  # 配置类实例
        self.camera_index = camera_index  # 摄像头索引
        self.camera_address = config.capture_source  # 摄像头地址
        self.camera_status_outgoing = camera_status_outgoing  # 状态输出队列
        self.camera_output_outgoing = camera_output_outgoing  # 图像输出队列
        self.capture_event = capture_event  # 捕获事件
        self.cancellation_event = cancellation_event  # 取消事件
        self.current_capture_source = config.capture_source  # 当前捕获源
        self.wired_camera: "cv2.VideoCapture" = None  # OpenCV视频捕获对象
        self.error_message = "Capture source {} not found, retrying"  # 错误消息

    # 设置图像输出队列
    def set_output_queue(self, camera_output_outgoing: "queue.Queue"):
        self.camera_output_outgoing = camera_output_outgoing

    # 运行摄像头捕获
    def run(self):
        while True:
            if self.cancellation_event.is_set():  # 检查是否触发取消事件
                print("Exiting capture thread")
                return
            should_push = True  # 标记是否应该推送图像到队列
            # 如果摄像头未打开或源发生变化，重新尝试连接摄像头
            if (
                self.config.capture_source is not None and self.config.capture_source != ""
            ):
                if (
                    self.wired_camera is None
                    or not self.wired_camera.isOpened()
                    or self.camera_status == CameraState.DISCONNECTED
                    or self.config.capture_source != self.current_capture_source
                ):
                    print(self.error_message.format(self.config.capture_source))
                    # 等待一段时间再尝试重新连接
                    if self.cancellation_event.wait(WAIT_TIME):
                        return
                    self.current_capture_source = self.config.capture_source
                    self.wired_camera = cv2.VideoCapture(self.current_capture_source)  # 重新打开摄像头
                    should_push = False  # 设置为不推送图像
            else:
                # 如果没有捕获源，等待配置
                if self.cancellation_event.wait(WAIT_TIME):
                    self.camera_status = CameraState.DISCONNECTED
                    return
            # 等待捕获事件触发，获取图像
            if should_push and not self.capture_event.wait(timeout=0.02):
                continue

            self.get_wired_camera_picture(should_push)  # 获取图像
            if not should_push:
                # 如果成功获取图像，设置为已连接状态
                self.camera_status = CameraState.CONNECTED

    # 获取摄像头图像
    def get_wired_camera_picture(self, should_push):
        try:
            ret, image = self.wired_camera.read()  # 从摄像头读取图像
            if not ret:  # 如果读取失败，重置摄像头帧位置，并抛出异常
                self.wired_camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                raise RuntimeError("Problem while getting frame")
            frame_number = self.wired_camera.get(cv2.CAP_PROP_POS_FRAMES)  # 获取当前帧编号
            fps = self.wired_camera.get(cv2.CAP_PROP_FPS)  # 获取帧率
            if should_push:
                self.push_image_to_queue(image, frame_number, fps)  # 推送图像到队列
        except:
            print(
                "Capture source problem, assuming camera disconnected, waiting for reconnect."
            )
            self.camera_status = CameraState.DISCONNECTED  # 设置摄像头为断开状态

    # 推送图像到队列
    def push_image_to_queue(self, image, frame_number, fps):
        # 如果队列中有积压，打印警告消息
        qsize = self.camera_output_outgoing.qsize()
        if qsize > 1:
            print(
                f"CAPTURE QUEUE BACKPRESSURE OF {qsize}. CHECK FOR CRASH OR TIMING ISSUES IN ALGORITHM."
            )
        # 将图像、帧编号和帧率推送到输出队列
        self.camera_output_outgoing.put((image, frame_number, fps))
        self.capture_event.clear()  # 清除捕获事件以等待下一次捕获