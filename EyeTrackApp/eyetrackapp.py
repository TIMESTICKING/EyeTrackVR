import os
from osc import EyeId  # 导入 EyeId 类
from config import EyeTrackConfig  # 导入 EyeTrackConfig 类
from camera_widget import CameraWidget  # 导入 CameraWidget 类
from settings_widget import SettingsWidget  # 导入 SettingsWidget 类
import queue  # 导入队列模块，用于线程间通信
import threading  # 导入线程模块
import PySimpleGUI as sg  # 导入 PySimpleGUI 模块，用于 GUI 创建
import sys  # 导入 sys 模块，用于系统相关操作
from urllib.request import urlopen  # 用于访问网页内容
from bs4 import BeautifulSoup  # 用于解析网页内容

import webbrowser  # 用于打开网页

if sys.platform.startswith("win"):
    from win10toast_click import ToastNotifier  # 导入 Windows 10 通知模块

# 设置环境变量以加快 MSMF 后端打开网络摄像头的速度
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# 窗口和小部件的名称常量
WINDOW_NAME = "EyeTrackApp"
RIGHT_EYE_NAME = "-RIGHTEYEWIDGET-"
LEFT_EYE_NAME = "-LEFTEYEWIDGET-"
SETTINGS_NAME = "-SETTINGSWIDGET-"

LEFT_EYE_RADIO_NAME = "-LEFTEYERADIO-"
RIGHT_EYE_RADIO_NAME = "-RIGHTEYERADIO-"
BOTH_EYE_RADIO_NAME = "-BOTHEYERADIO-"
SETTINGS_RADIO_NAME = '-SETTINGSRADIO-'

# 最新版本发布页面 URL
page_url = 'https://github.com/RedHawk989/EyeTrackVR/releases/latest'

def open_url():
    """打开最新版本发布页面"""
    try: 
        webbrowser.open_new(page_url)
        print('Opening URL...')  
    except: 
        print('Failed to open URL. Unsupported variable type.')


def main():
    # Get Configuration
     # 获取配置
    config: EyeTrackConfig = EyeTrackConfig.load()
    config.save()

    cancellation_event = threading.Event()     # 创建事件对象，用于线程间通信

    # Check to see if we can connect to our video source first. If not, bring up camera finding
    # dialog.

     # 检查视频源连接
     #这段代码的主要功能是从一个给定的URL中获取内容，解析并移除任何 <script> 和 <style> 标签，
     # 然后提取并返回纯文本内容。这样处理后得到的 text 变量会包含该网页或文件中的所有可读文本内容。
    appversion = "0.1.7.2"
    url = "https://raw.githubusercontent.com/RedHawk989/EyeTrackVR-Installer/master/Version-Data/Version_Num.txt"
    html = urlopen(url).read()         # 0.1.8.1
    soup = BeautifulSoup(html, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()                 # 移除脚本和样式标签
    text = soup.get_text()                  # 获取纯文本内容

    # break into lines and remove leading and trailing space on each
    # 分行并去除每行的前后空白
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    # 将多标题拆分为每行一个
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    # 去除空行
    latestversion = '\n'.join(chunk for chunk in chunks if chunk)
    
    # 比较硬编码版本与最新版本
    if appversion == latestversion: # If what we scraped and hardcoded versions are same, assume we are up to date.
        print(f"[INFO] App is up to date! {latestversion}")
    else: 
        print(f"[INFO] You have app version {appversion} installed. Please update to {latestversion} for the newest fixes.")
        if sys.platform.startswith("win"):           # 显示 Windows 通知
            toaster = ToastNotifier()
            toaster.show_toast(  #show windows toast
                "EyeTrackVR has an update.",
                "Click to go to the latest version.",
                icon_path= "Images/logo.ico",
                duration=5,
                threaded=True,
                callback_on_click=open_url
                )
            

    # Check to see if we have an ROI. If not, bring up ROI finder GUI.
     # 检查是否有 ROI。如果没有，则启动 ROI 查找 GUI。

    # Spawn worker threads
    '''
    queue.Queue 是 Python 标准库中的一个类，用来创建队列。这里的 queue.Queue[tuple[bool, int, int]] 
    表示这个队列专门用来存放 三元组（tuple[bool, int, int]），即每个进入队列的元素都是包含三个值的元组：

	•	第一个值 是一个布尔值 (bool)，表示 True 或 False。
	•	第二个值 是一个整数 (int)。
	•	第三个值 也是一个整数 (int)。
    这段代码创建了一个用于存储三元组（布尔值和两个整数）的队列。你可以在多线程程序中使用它来传递和处理数据。
    '''
    osc_queue: queue.Queue[tuple[bool, int, int]] = queue.Queue()
    # osc = VRChatOSC(cancellation_event, osc_queue, config)
    # osc_thread = threading.Thread(target=osc.run)
    # # start worker threads
     # 启动工作线程
    # osc_thread.start()


     # 创建眼动追踪小部件
    eyes = [
        CameraWidget(EyeId.RIGHT, config, osc_queue),
        CameraWidget(EyeId.LEFT, config, osc_queue),
    ]

    # 创建设置小部件
    settings = [
        SettingsWidget(EyeId.SETTINGS, config, osc_queue),
    ]

    # 创建布局
    layout = [
        [
            sg.Radio(
                "Right Eye",
                "EYESELECTRADIO",
                background_color='#292929',
                default=(config.eye_display_id == EyeId.RIGHT),
                key=RIGHT_EYE_RADIO_NAME,
            ),
            sg.Radio(
                "Left Eye",
                "EYESELECTRADIO",
                background_color='#292929',
                default=(config.eye_display_id == EyeId.LEFT),
                key=LEFT_EYE_RADIO_NAME,
            ),
            sg.Radio(
                "Both Eyes",
                "EYESELECTRADIO",
                background_color='#292929',
                default=(config.eye_display_id == EyeId.BOTH),
                key=BOTH_EYE_RADIO_NAME,
            ),
            sg.Radio(
                "Settings",
                "EYESELECTRADIO",
                background_color='#292929',
                default=(config.eye_display_id == EyeId.SETTINGS),
                key=SETTINGS_RADIO_NAME,
            ),
        ],
        [
            sg.Column(
                eyes[1].widget_layout,
                vertical_alignment="top",
                key=LEFT_EYE_NAME,
                visible=(config.eye_display_id in [EyeId.LEFT, EyeId.BOTH]),
                background_color='#424042',
            ),
            sg.Column(
                eyes[0].widget_layout,
                vertical_alignment="top",
                key=RIGHT_EYE_NAME,
                visible=(config.eye_display_id in [EyeId.RIGHT, EyeId.BOTH]),
                background_color='#424042',
            ),
            sg.Column(
                settings[0].widget_layout,
                vertical_alignment="top",
                key=SETTINGS_NAME,
                visible=(config.eye_display_id in [EyeId.SETTINGS]),
                background_color='#424042',
            ),
        ],
    ]


    # 根据配置启动眼动追踪
    if config.eye_display_id in [EyeId.LEFT, EyeId.BOTH]:
        eyes[1].start()
    if config.eye_display_id in [EyeId.RIGHT, EyeId.BOTH]:
        eyes[0].start()

    if config.eye_display_id in [EyeId.SETTINGS, EyeId.BOTH]:
        settings[0].start()

    # the eye's needs to be running before it is passed to the OSC
    # osc_receiver = VRChatOSCReceiver(cancellation_event, config, eyes)
    # osc_receiver_thread = threading.Thread(target=osc_receiver.run)
    # osc_receiver_thread.start()

    # Create the window
    # 创建窗口
    window = sg.Window(f"EyeTrackVR {appversion}" , layout, icon='Images/logo.ico', background_color='#292929')

    # GUI Render loop
     # GUI 渲染循环
    while True:
        # First off, check for any events from the GUI
        # 检查 GUI 事件
        event, values = window.read(timeout=1)

        # If we're in either mode and someone hits q, quit immediately
         # 如果点击退出或窗口关闭，停止所有线程并退出
        if event == "Exit" or event == sg.WIN_CLOSED:
            for eye in eyes:
                eye.stop()
            cancellation_event.set()
            # 关闭工作线程
            # shut down worker threads
            # osc_thread.join()
            # TODO: find a way to have this function run on join maybe??
            # threading.Event() wont work because pythonosc spawns its own thread.
            # only way i can see to get around this is an ugly while loop that only checks if a threading event is triggered
            # and then call the pythonosc shutdown function
            # osc_receiver.shutdown()
            # osc_receiver_thread.join()
            print("Exiting EyeTrackApp")
            return

         # 根据选择更新显示的眼动追踪小部件
        if values[RIGHT_EYE_RADIO_NAME] and config.eye_display_id != EyeId.RIGHT:
            eyes[0].start()
            eyes[1].stop()
            settings[0].stop()
            window[RIGHT_EYE_NAME].update(visible=True)
            window[LEFT_EYE_NAME].update(visible=False)
            window[SETTINGS_NAME].update(visible=False)
            config.eye_display_id = EyeId.RIGHT
            config.settings.tracker_single_eye = 2
            config.save()
        elif values[LEFT_EYE_RADIO_NAME] and config.eye_display_id != EyeId.LEFT:
            settings[0].stop()
            eyes[0].stop()
            eyes[1].start()
            window[RIGHT_EYE_NAME].update(visible=False)
            window[LEFT_EYE_NAME].update(visible=True)
            window[SETTINGS_NAME].update(visible=False)
            config.eye_display_id = EyeId.LEFT
            config.settings.tracker_single_eye = 1
            config.save()
        elif values[BOTH_EYE_RADIO_NAME] and config.eye_display_id != EyeId.BOTH:
            settings[0].stop()
            eyes[0].stop()
            eyes[1].start()
            eyes[0].start()

            window[LEFT_EYE_NAME].update(visible=True)
            window[RIGHT_EYE_NAME].update(visible=True)
            window[SETTINGS_NAME].update(visible=False)
            config.eye_display_id = EyeId.BOTH
            config.settings.tracker_single_eye = 0
            config.save()

        elif values[SETTINGS_RADIO_NAME] and config.eye_display_id != EyeId.SETTINGS:
            eyes[0].stop()
            eyes[1].stop()
            settings[0].start()
            window[RIGHT_EYE_NAME].update(visible=False)
            window[LEFT_EYE_NAME].update(visible=False)
            window[SETTINGS_NAME].update(visible=True)
            config.eye_display_id = EyeId.SETTINGS
            config.save()

        # Otherwise, render all of our cameras
         # 渲染眼动追踪小部件
        for eye in eyes:
            if eye.started():
                eye.render(window, event, values)
        settings[0].render(window, event, values)


if __name__ == "__main__":
    main()
    