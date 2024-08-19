from operator import truth  # 标准库模块，用于操作和比较的函数
from dataclasses import dataclass  # 标准库模块，用于创建数据类
import sys  # 标准库模块，提供对解释器相关功能的访问
import asyncio  # 标准库模块，用于编写并发代码

sys.path.append(".")
from config import EyeTrackCameraConfig
from config import EyeTrackSettingsConfig
from pye3d.camera import CameraModel
from pye3d.detector_3d import Detector3D, DetectorMode
import queue  # 标准库模块，用于线程间的安全队列
import threading  # 标准库模块，用于创建和管理线程
import numpy as np  # 第三方模块，用于科学计算和数组操作
import cv2  # OpenCV库，用于计算机视觉和图像处理
from enum import Enum  # 标准库模块，用于定义枚举类
from one_euro_filter import OneEuroFilter  # 第三方模块，用于信号滤波
if sys.platform.startswith("win"):
    from winsound import PlaySound, SND_FILENAME, SND_ASYNC  # Windows平台特定的模块，用于播放声音


# 枚举类，用于定义应用程序的各种状态
class AppState(Enum):
    calibrate = 0  # 校准状态
    process = 1  # 处理状态
    finish = 2  # 结束状态

'''
详细解释：
1.	EyeInformation 数据类：
	•	@dataclass 是一个用于简化数据类定义的装饰器，它会自动生成常用的特殊方法，如 __init__、__repr__ 等。
	•	info_type:眼部信息的来源类型。这通常可能是枚举类型，表示信息是来自左眼、右眼或某个其他来源。
	•	x 和 y:表示眼睛注视点的横向和纵向坐标，通常用浮点数表示。
	•	pupil_dialation:瞳孔扩张度，通常是一个整数值，表示瞳孔的大小。
	•	blink:一个布尔值，表示当前眼睛是否正在眨眼。
2.	lowb = np.array(0):
	•	创建了一个名为 lowb 的 NumPy 数组，包含一个元素 0。
	•	虽然这里只有一个元素为0的数组，但 lowb 变量名可能预示它会在后续代码中用于表示某种下界或阈值。
    这个值也可以根据需要进一步扩展为更复杂的数组。
'''


# 定义用于存储眼部信息的数据类
@dataclass
class EyeInformation:
    info_type: InformationOrigin  # 眼部信息的来源类型（如左眼或右眼）
    x: float  # 眼睛注视点的x坐标
    y: float  # 眼睛注视点的y坐标
    pupil_dialation: int  # 瞳孔扩张度
    blink: bool  # 是否眨眼的布尔值

# 创建一个包含单一元素的NumPy数组，元素值为0
lowb = np.array(0)


# 定义一个装饰器函数，用于确保函数只运行一次
def run_once(f):
    def wrapper(*args, **kwargs):
        # 检查函数是否已经运行过，如果没有，则运行
        if not wrapper.has_run:
            wrapper.has_run = True  # 标记函数已经运行过
            return f(*args, **kwargs)  # 调用被装饰的函数

    wrapper.has_run = False  # 初始化时设置为False，表示函数还未运行
    return wrapper  # 返回包装后的函数

'''
1.	run_once(f) 装饰器函数：
	•	run_once 是一个装饰器函数，用于确保某个函数在整个程序运行期间只执行一次。
	•	wrapper 是一个包装函数，它会接收任何数量的位置和关键字参数。
	•	wrapper.has_run 是一个布尔值，初始为 False,用于标记函数是否已经运行过。
	•	如果 wrapper.has_run 为 False,则执行被装饰的函数 f，并将 wrapper.has_run 设为 True，防止函数再次运行。
2.	**delayed_setting_change(setting, value)
'''




# 定义一个异步函数，用于延迟更改设置值
async def delayed_setting_change(setting, value):
    await asyncio.sleep(5)  # 等待5秒
    setting = value  # 更改设置值
    # 如果是在Windows平台上，播放完成的提示音
    if sys.platform.startswith("win"):
        PlaySound('Audio/compleated.wav', SND_FILENAME | SND_ASYNC)  # 异步播放音频文件


'''
详细解释：
1.	函数 fit_rotated_ellipse_ransa(c:
	•	这个函数使用随机抽样一致性(RANSAC)算法拟合一个旋转椭圆模型。RANSAC 是一种稳健的拟合方法,
    能够在大量噪声数据中找到合适的模型参数。
2.	参数解释:
	•	data: 包含二维点集的数组，形如 [[x1, y1], [x2, y2], ...]。
	•	iter: 迭代次,RANSAC 算法会执行 iter 次采样和模型拟合，以找到最优模型。
	•	sample_num: 每次迭代时从数据中随机选择的样本点数量。
	•	offset: 判断一个点是否符合模型的阈值，如果点到椭圆的距离小于这个值，则认为该点符合椭圆模型。
3.	RANSAC 过程:
	•	在每次迭代中，随机从数据中选择 sample_num 个样本点，并使用这些点拟合一个旋转椭圆模型。
	•	计算这个模型下所有数据点到模型的误差，并根据 offset 阈值筛选符合模型的点。
	•	记录符合模型的点最多的那组样本，作为最优样本点集。
4.	返回值:
	•	使用 fit_rotated_ellipse 函数对最优样本点集进行精确拟合，得到最终的旋转椭圆模型。

'''

def fit_rotated_ellipse_ransac(
    data, iter=5, sample_num=10, offset=80  # 默认值为 5 次迭代，10 个样本，80 的偏移量
):  
    # 在修改这些值之前，请先了解 RANSAC 算法的工作原理。
    # 如果你想修改这些值，需要注意的是，增加迭代次数会使处理帧的速度变慢。
    
    count_max = 0  # 用于存储最多符合模型的点的数量
    effective_sample = None  # 用于存储最有效的样本点集

    # TODO 这个迭代过程非常慢。
    #
    # 我们需要保持迭代次数较低，或者需要一个专门的工作池来处理这个计算。
    # 这个过程是可以并行化的，所以可以使用像 joblib 这样的工具来优化。
    for i in range(iter):  # 迭代指定次数
        sample = np.random.choice(len(data), sample_num, replace=False)  
        # 随机从数据中选择 sample_num 个样本点

        xs = data[sample][:, 0].reshape(-1, 1)  # 提取 x 坐标
        ys = data[sample][:, 1].reshape(-1, 1)  # 提取 y 坐标

        # 构造线性系统 J * P = Y，用于拟合椭圆模型参数
        J = np.mat(
            np.hstack((xs * ys, ys**2, xs, ys, np.ones_like(xs, dtype=np.float)))
        )
        Y = np.mat(-1 * xs**2)
        P = (J.T * J).I * J.T * Y  # 计算 P 参数向量

        # 椭圆模型的参数形式：a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
        a = 1.0
        b = P[0, 0]
        c = P[1, 0]
        d = P[2, 0]
        e = P[3, 0]
        f = P[4, 0]
        ellipse_model = (
            lambda x, y: a * x**2 + b * x * y + c * y**2 + d * x + e * y + f
        )

        # 根据阈值筛选符合椭圆模型的样本点
        ran_sample = np.array(
            [[x, y] for (x, y) in data if np.abs(ellipse_model(x, y)) < offset]
        )

        if len(ran_sample) > count_max:  
            # 如果当前样本点集符合模型的点数量超过之前的最大值，更新最优样本点集
            count_max = len(ran_sample)
            effective_sample = ran_sample

    return fit_rotated_ellipse(effective_sample)  # 调用函数拟合旋转椭圆并返回结果


'''
 	•	该函数通过最小二乘法拟合数据点的旋转椭圆，提取椭圆的中心、宽度、高度和旋转角度。
	•	data 参数是包含数据点的二维数组，其中每行表示一个 (x, y) 坐标。
	•	椭圆的方程为:a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0。
'''

def fit_rotated_ellipse(data):
    # 将数据的 x 坐标和 y 坐标分开，并将它们转换为列向量
    xs = data[:, 0].reshape(-1, 1)
    ys = data[:, 1].reshape(-1, 1)

    # 构造矩阵 J，其列为 [x*y, y^2, x, y, 1]，Y 为 -x^2
    J = np.mat(np.hstack((xs * ys, ys**2, xs, ys, np.ones_like(xs, dtype=np.float))))
    Y = np.mat(-1 * xs**2)

    # 利用最小二乘法求解 P，其中 P 是包含椭圆方程系数的矩阵
    P = (J.T * J).I * J.T * Y

    # 提取椭圆方程的系数 a, b, c, d, e, f
    a = 1.0
    b = P[0, 0]
    c = P[1, 0]
    d = P[2, 0]
    e = P[3, 0]
    f = P[4, 0]

    # 计算椭圆的旋转角度 theta
    theta = 0.5 * np.arctan(b / (a - c))

    # 计算椭圆的中心 cx, cy
    cx = (2 * c * d - b * e) / (b**2 - 4 * a * c)
    cy = (2 * a * e - b * d) / (b**2 - 4 * a * c)

    # 计算椭圆的参数 cu，用于后续计算椭圆的宽度和高度
    cu = a * cx**2 + b * cx * cy + c * cy**2 - f

    # 计算椭圆的宽度 w 和高度 h
    w = np.sqrt(
        cu
        / (
            a * np.cos(theta)**2
            + b * np.cos(theta) * np.sin(theta)
            + c * np.sin(theta)**2
        )
    )
    h = np.sqrt(
        cu
        / (
            a * np.sin(theta)**2
            - b * np.cos(theta) * np.sin(theta)
            + c * np.cos(theta)**2
        )
    )

    # 定义椭圆模型，用于计算误差
    ellipse_model = lambda x, y: a * x**2 + b * x * y + c * y**2 + d * x + e * y + f

    # 计算所有数据点与拟合椭圆之间的误差总和
    error_sum = np.sum([ellipse_model(x, y) for x, y in data])

    # 返回椭圆的中心 (cx, cy), 宽度 w, 高度 h, 和旋转角度 theta
    return (cx, cy, w, h, theta)

'''
	•	EyeProcessor 类负责处理眼球图像的捕获、裁剪、旋转和跟踪操作。
	•	它使用了 OneEuroFilter 进行数据的平滑处理。
	•	它还处理了跨线程的通信，将处理后的图像和信息通过队列传输。
	•	blob_tracking_fallback 方法提供了对眼球跟踪的一个回退机制，当算法失败时，试图利用 Blob 跟踪继续操作。
'''


class EyeProcessor:
    def __init__(                        # __init__ 方法是 EyeProcessor 类的构造函数，它初始化了类的实例。
        self,
        config: "EyeTrackCameraConfig",          # 配置参数，包含与摄像头相关的设置，如分辨率、ROI（感兴趣区域）等
        settings: "EyeTrackSettingsConfig",       # 设置参数，包含与眼球跟踪算法相关的设置，如阈值、滤波器参数等
        cancellation_event: "threading.Event",  # 线程事件，用于通知线程停止运行
        capture_event: "threading.Event",        # 线程事件，用于通知开始捕获图像
        capture_queue_incoming: "queue.Queue",  # 用于接收图像帧的队列
        image_queue_outgoing: "queue.Queue",    # 用于发送处理后图像帧的队列
        eye_id,                                 # 眼球的标识符，用于区分不同的眼球或摄像头
    ):
        # 初始化配置和状态参数
        self.config = config  # 眼球摄像头配置
        self.settings = settings  # 眼球跟踪设置


            # 跨线程通信管理
            # 跨线程通信管理
        self.capture_queue_incoming = capture_queue_incoming  # 捕获队列，接收图像帧
        self.image_queue_outgoing = image_queue_outgoing  # 输出队列，发送处理后的图像
        self.cancellation_event = cancellation_event  # 取消事件，用于停止线程
        self.capture_event = capture_event  # 捕获事件，用于触发图像捕获
        self.eye_id = eye_id  # 眼球 ID，用于区分不同的眼球或摄像头

        # 跨算法状态
        self.lkg_projected_sphere = None  # 最近一次投影的眼球球体数据
        self.xc = None  # 眼睛在图像中的 x 坐标中心
        self.yc = None  # 眼睛在图像中的 y 坐标中心

        # 图像处理状态
        self.previous_image = None         # 前一帧图像
        self.current_image = None          # 当前帧图像
        self.current_image_gray = None     # 当前帧灰度图像
        self.current_frame_number = None   # 当前帧编号
        self.current_fps = None            # 当前帧率
        self.threshold_image = None        # 阈值图像

        # 校准值
        self.xoff = 1  # x 方向的偏移量
        self.yoff = 1  # y 方向的偏移量
        self.calibration_frame_counter = None  # 校准帧计数器
        self.eyeoffx = 1  # 眼球 x 方向偏移量

        # 初始化最大最小值，用于更新图像中的眼睛位置
        self.xmax = -69420
        self.xmin = 69420
        self.ymax = -69420
        self.ymin = 69420
        self.cct = 300
        self.cccs = False
        self.ts = 10
        self.previous_rotation = self.config.rotation_angle  # 前一次图像的旋转角度

    '''
    

- **OneEuroFilter**:  
  OneEuroFilter 是一种用于平滑噪声数据的滤波器，特别适用于实时数据的处理。
  它能够根据数据的变化速率自动调整滤波强度，从而在平滑噪声的同时保留快速变化的信号。

- **`min_cutoff`（最小截止频率）**:  
  这个参数决定了滤波器的平滑程度。较低的 `min_cutoff` 会使得滤波器更平滑，
  但也会延迟响应。较高的 `min_cutoff` 则会使得滤波器更加敏感。

- **`beta`（速度系数）**: 
  `beta` 参数决定了滤波器对数据变化速率的响应速度。较高的 `beta` 值会让滤波器在数据变化迅速时更加敏感，从而减小延迟。

- **错误处理**:
  在从配置中获取 `min_cutoff` 和 `beta` 值时，代码使用了 `try`/`except` 结构来捕捉可能的错误
  。如果提供的值不能转换为浮点数（例如用户输入了非法字符），
  则会打印一条警告信息，并使用默认值 `min_cutoff=0.0004` 和 `beta=0.9`。

- **初始化滤波器**:
  `self.one_euro_filter` 是一个 OneEuroFilter 对象，
  它使用指定的 `noisy_point`、`min_cutoff` 和 `beta` 值来初始化。
  这个滤波器将用于平滑处理过程中获取的眼球数据，以消除噪声并获得更稳定的结果。

    
    '''
# 初始化 OneEuroFilter，用于平滑噪声数据
    try:
        # 从设置中获取最小截止频率（min_cutoff）和速度系数（beta）
        min_cutoff = float(self.settings.gui_min_cutoff)  # 最小截止频率
        beta = float(self.settings.gui_speed_coefficient) # 速度系数
    except:
     # 如果获取过程中出现错误，则使用默认值并发出警告
        print('[WARN] OneEuroFilter values must be a legal number.')
        min_cutoff = 0.0004  # 默认的最小截止频率
        beta = 0.9  # 默认的速度系数

    # 创建一个初始噪声点，通常用作滤波器的初始值
        noisy_point = np.array([1, 1])

    # 初始化 OneEuroFilter 滤波器对象，用于处理噪声数据
        self.one_euro_filter = OneEuroFilter(
        noisy_point,  # 传入初始噪声点
        min_cutoff=min_cutoff,  # 设置最小截止频率
        beta=beta  # 设置速度系数
        )


    '''
    •threshold_image:
        这是经过某种阈值处理后的图像，通常用于将图像二值化，即将图像中所有像素点根据某个阈值分为两类（例如黑白
        图像），便于后续的特征提取和分析。
	•output_information:
        这是一个 EyeInformation 对象，包含了与当前帧相关的处理信息，例如瞳孔扩张、是否眨眼等信息。
	•	cv2.cvtColor:
        OpenCV 的 cvtColor 函数用于将图像从一种颜色空间转换到另一种颜色空间。
        在这里，它将灰度图像转换为 BGR 图像。BGR 图像有三个通道（蓝色、绿色、红色），
        而灰度图像只有一个通道。这个转换是为了后续的图像拼接处理。
	•np.concatenate:
        这个函数用于沿指定的轴将两个或多个数组拼接在一起。
        在这段代码中,沿水平方向(axis=1)拼接当前帧的灰度图像和阈值图像。最终的结果是一个双宽度的 BGR 图像，
        其中左侧是原始灰度图像，右侧是阈值图像。
	•self.image_queue_outgoing.put:
        这个方法将拼接后的图像堆栈和与之相关的输出信息作为一个元组放入 image_queue_outgoing 队列中。
        这通常用于跨线程通信或异步处理，使得其他线程或进程可以从队列中读取数据并进行进一步处理或显示。
	•更新图像和旋转角度:
        self.previous_image 和 self.previous_rotation 存储当前帧的图像和旋转角度，
        以便在后续帧处理中可以参考和比较这些信息。这在连续帧处理或跟踪过程中是常见的做法，
        用于确保算法的一致性和准确性。
    '''

    def output_images_and_update(self, threshold_image, output_information: EyeInformation):
        # 将当前图像和阈值图像拼接在一起，形成一个图像堆栈
        image_stack = np.concatenate(
            (
                cv2.cvtColor(self.current_image_gray, cv2.COLOR_GRAY2BGR),  # 灰度图像转换为 BGR
                cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR),          # 阈值图像转换为 BGR
            ),
            axis=1,
        )
        # 将图像堆栈和输出信息放入输出队列中
        self.image_queue_outgoing.put((image_stack, output_information))
        # 更新前一帧图像和旋转角度
        self.previous_image = self.current_image
        self.previous_rotation = self.config.rotation_angle

    '''
    代码功能概述：

	1.	捕获当前帧图像:从数据源中获取当前图像帧,并裁剪到感兴趣区域(ROI)。ROI 是一个矩形区域，
        通常用于集中处理特定区域（如眼睛所在区域）。
	2.	错误处理：如果获取图像或裁剪时发生错误，则使用前一帧图像，并输出错误信息。
	3.	图像旋转：对裁剪后的图像应用旋转操作。旋转操作以图像中心为原点，按照配置中的角度进行旋转。
        对于旋转后图像之外的区域，使用白色填充。
	4.	返回值：函数返回 True,表明操作成功。
    '''

    def capture_crop_rotate_image(self):
        # 获取当前帧图像
        try:
            # 从捕获源中获取图像帧，并裁剪到感兴趣区域（ROI）
            self.current_image = self.current_image[
                int(self.config.roi_window_y): int(
                    self.config.roi_window_y + self.config.roi_window_h
                ),
                int(self.config.roi_window_x): int(
                    self.config.roi_window_x + self.config.roi_window_w
                ),
            ]
        except:
            # 如果图像处理失败，则使用前一帧图像
            self.current_image = self.previous_image
            print("[ERROR] Frame capture issue detected.")

        # 对裁剪区域应用旋转。对于图像边界之外的区域，填充白色。
        rows, cols, _ = self.current_image.shape
        img_center = (cols / 2, rows / 2)
        rotation_matrix = cv2.getRotationMatrix2D(
            img_center, self.config.rotation_angle, 1
        )
        self.current_image = cv2.warpAffine(
            self.current_image,
            rotation_matrix,
            (cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return True

    '''
    1.	圆形裁剪处理：
	•	如果启用了圆形裁剪(gui_circular_crop),并且 cct 为 0,则程序会根据之前计算的眼球半径,
        在图像上绘制一个圆形遮罩。遮罩会保留眼球区域，其他区域填充为白色，从而突出显示感兴趣的部分。
	•	cct 用作一个计数器，可以控制圆形裁剪的频率。如果 cct 不为 0,程序会递减该值,而不进行裁剪操作。
	2.	阈值处理：
	•	程序将当前的灰度图像进行阈值处理，将像素值超过设定阈值的部分设为白色(即 255),低于阈值的部分设为黑色(即 0)。
        这一步骤有助于突出显示图像中的目标对象（如眼球），便于后续的跟踪算法处理。
	3.	Blob 跟踪前提条件：
	•	如果当前没有获取到眼球的投影球体信息（即 lkg_projected_sphere 为空），程序无法进行 Blob 跟踪。
        因此，程序会将当前处理的图像和跟踪失败的信息发送到输出队列中，并终止函数执行。   

	1.	轮廓检测：
	    •	使用 cv2.findContours 方法在处理过的阈值图像 larger_threshold 中查找轮廓，
            并按照轮廓面积从大到小进行排序。如果未找到任何轮廓，则抛出运行时异常，并输出失败信息。
	2.	轮廓遍历与跟踪：
	    •	程序遍历找到的轮廓，通过计算每个轮廓的边界框，判断其尺寸是否在预设的范围内。如果不符合条件，则跳过该轮廓。
	    •	如果符合条件,则计算轮廓的中心点(cx, cy)并将其与先前计算的眼球投影球体信息进行对比,
            生成相应的跟踪坐标(xrlb, eyeyb)。
	3.	绘制跟踪信息：
	    •	在图像上绘制跟踪线条、轮廓和边界框，以便可视化眼动追踪过程。
	4.	校准与重定位：
	    •	如果正在进行校准(calibration_frame_counter 为非零)，则更新校准数据，如眼睛的偏移量和最大最小边界值。
            校准完成后，播放音效并重置校准计数器。
	    •	如果启用了眼睛重定位功能，程序会自动调整偏移量，并在完成后播放音效。
	5.	坐标计算与滤波：
	    •	根据计算出的偏移量和边界值,生成归一化的坐标值(out_x, out_y),并根据配置文件中的设置决定是否翻转坐标轴。
	    •	使用 One Euro Filter 对生成的坐标值进行滤波，以平滑数据，减少噪声对结果的影响。
	6.	输出与更新：
	    •	最终，将处理后的图像和生成的眼动信息通过 output_images_and_update 方法发送到输出队列中。
        如果未能成功检测到眼睛,则输出一个失败信息,并记录“BLINK Detected”(检测到眨眼)。
    '''


    def blob_tracking_fallback(self):
        # 定义圆形裁剪
        if self.config.gui_circular_crop:
            if self.cct == 0:  # 当cct（某个计数器）为0时，执行以下操作
                try:
                    # 获取当前灰度图像的高度和宽度
                    ht, wd = self.current_image_gray.shape[:2]

                    # 获取眼球的半径值，并使用该值绘制一个圆形遮罩
                    radius = int(float(self.lkg_projected_sphere["axes"][0]))
                    
                    # 创建一个黑色的遮罩（全零数组）
                    mask = np.zeros((ht, wd), dtype=np.uint8)
                    # 在遮罩上绘制一个白色的圆形区域，圆心为当前眼球的中心，半径为前面获取的radius
                    mask = cv2.circle(mask, (self.xc, self.yc), radius, 255, -1)
                    
                    # 创建一个与当前灰度图像大小相同的白色背景
                    color = np.full_like(self.current_image_gray, (255))
                    # 将圆形遮罩应用到当前灰度图像上，只保留遮罩区域内的图像部分
                    masked_img = cv2.bitwise_and(self.current_image_gray, self.current_image_gray, mask=mask)
                    # 将反向遮罩（遮罩区域外的部分）应用到白色背景上，保留白色区域
                    masked_color = cv2.bitwise_and(color, color, mask=255 - mask)
                    
                    # 将处理后的图像和背景合并，得到最终的圆形裁剪图像
                    self.current_image_gray = cv2.add(masked_img, masked_color)
                except:
                # 如果上述操作过程中发生任何错误，则不进行处理
                    pass
            else:
                # 如果cct不为0，则递减cct的值
                    self.cct = self.cct - 1
        
        # 对当前灰度图像进行阈值处理，将像素值超过阈值的部分设为白色，低于阈值的部分设为黑色
        _, larger_threshold = cv2.threshold(self.current_image_gray, int(self.config.threshold + 12), 255, cv2.THRESH_BINARY)
        
        # Blob跟踪算法需要对当前眼球位置有一个初步估计
        # 如果尚未获得投影球体信息（lkg_projected_sphere为空），则无法进行跟踪
        if self.lkg_projected_sphere is None:
            # 输出当前处理后的二值化图像和跟踪失败的信息
            self.output_images_and_update(
                larger_threshold, EyeInformation(InformationOrigin.FAILURE, 0, 0, 0, False)
            )
        # 终止函数执行
        return

        '''

    代码功能概述：

	1.	轮廓检测：
	    •	使用 cv2.findContours 方法在处理过的阈值图像 larger_threshold 中查找轮廓，
            并按照轮廓面积从大到小进行排序。如果未找到任何轮廓，则抛出运行时异常，并输出失败信息。
	2.	轮廓遍历与跟踪：
	    •	程序遍历找到的轮廓，通过计算每个轮廓的边界框，判断其尺寸是否在预设的范围内。如果不符合条件，则跳过该轮廓。
	    •	如果符合条件,则计算轮廓的中心点(cx, cy)并将其与先前计算的眼球投影球体信息进行对比,
            生成相应的跟踪坐标(xrlb, eyeyb)。
	3.	绘制跟踪信息：
	    •	在图像上绘制跟踪线条、轮廓和边界框，以便可视化眼动追踪过程。
	4.	校准与重定位：
	    •	如果正在进行校准(calibration_frame_counter 为非零)，则更新校准数据，如眼睛的偏移量和最大最小边界值。
            校准完成后，播放音效并重置校准计数器。
	    •	如果启用了眼睛重定位功能，程序会自动调整偏移量，并在完成后播放音效。
	5.	坐标计算与滤波：
	    •	根据计算出的偏移量和边界值,生成归一化的坐标值(out_x, out_y),并根据配置文件中的设置决定是否翻转坐标轴。
	    •	使用 One Euro Filter 对生成的坐标值进行滤波，以平滑数据，减少噪声对结果的影响。
	6.	输出与更新：
	    •	最终，将处理后的图像和生成的眼动信息通过 output_images_and_update 方法发送到输出队列中。
        如果未能成功检测到眼睛,则输出一个失败信息,并记录“BLINK Detected”(检测到眨眼)。
        '''

        try:
            # 尝试查找图像中的轮廓
            contours, _ = cv2.findContours(
                larger_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            # 如果没有找到轮廓，抛出异常并输出失败信息
            if len(contours) == 0:
                raise RuntimeError("No contours found for image")
        except:
            self.output_images_and_update(
                larger_threshold, EyeInformation(InformationOrigin.FAILURE, 0, 0, 0, False)
            )
            return

        rows, cols = larger_threshold.shape
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)

            # 检查 blob 的宽度/高度是否在预设范围内
            if not self.settings.gui_blob_minsize <= h <= self.settings.gui_blob_maxsize or not self.settings.gui_blob_minsize <= w <= self.settings.gui_blob_maxsize:
                continue

            # 计算轮廓的中心点坐标
            cx = x + int(w / 2)
            cy = y + int(h / 2)

            # 计算归一化后的坐标值
            xrlb = (cx - self.lkg_projected_sphere["center"][0]) / self.lkg_projected_sphere["axes"][0]
            eyeyb = (cy - self.lkg_projected_sphere["center"][1]) / self.lkg_projected_sphere["axes"][1]

            # 在图像上绘制跟踪线条、轮廓和边界框
            cv2.line(
                self.current_image_gray,
                (x + int(w / 2), 0),
                (x + int(w / 2), rows),
                (255, 0, 0),
                1,
            )  # 在阈值图像上可视化眼动跟踪
            cv2.line(
                self.current_image_gray,
                (0, y + int(h / 2)),
                (cols, y + int(h / 2)),
                (255, 0, 0),
                1,
            )
            cv2.drawContours(self.current_image_gray, [cnt], -1, (255, 0, 0), 3)
            cv2.rectangle(
                self.current_image_gray, (x, y), (x + w, y + h), (255, 0, 0), 2
            )

            # 校准逻辑处理
            if self.calibration_frame_counter == 0:
                self.calibration_frame_counter = None
                self.xoff = cx
                self.yoff = cy
                if sys.platform.startswith("win"):
                    PlaySound('Audio/compleated.wav', SND_FILENAME | SND_ASYNC)
            elif self.calibration_frame_counter is not None:
                self.settings.gui_recenter_eyes = False
                # 更新最大最小边界值
                if cx > self.xmax:
                    self.xmax = cx
                if cx < self.xmin:
                    self.xmin = cx
                if cy > self.ymax:
                    self.ymax = cy
                if cy < self.ymin:
                    self.ymin = cy
                self.calibration_frame_counter -= 1

            # 处理眼睛重定位
            if self.settings.gui_recenter_eyes:
                self.xoff = cx
                self.yoff = cy
                if self.ts == 0:
                    self.settings.gui_recenter_eyes = False
                    if sys.platform.startswith("win"):
                        PlaySound('Audio/compleated.wav', SND_FILENAME | SND_ASYNC)
                else:
                    self.ts -= 1
            else:
                self.ts = 10

            # 计算归一化后的坐标值
            xl = float((cx - self.xoff) / (self.xmax - self.xoff))
            xr = float((cx - self.xoff) / (self.xmin - self.xoff))
            yu = float((cy - self.yoff) / (self.ymin - self.yoff))
            yd = float((cy - self.yoff) / (self.ymax - self.yoff))

            out_x = 0
            out_y = 0

            # 根据配置决定是否翻转 Y 轴，并更新输出坐标
            if self.settings.gui_flip_y_axis:
                if yd > 0:
                    out_y = max(0.0, min(1.0, yd))
                if yu > 0:
                    out_y = -abs(max(0.0, min(1.0, yu)))
            else:
                if yd > 0:
                    out_y = -abs(max(0.0, min(1.0, yd)))
                if yu > 0:
                    out_y = max(0.0, min(1.0, yu))

            # 根据配置决定是否翻转 X 轴，并更新输出坐标
            if self.settings.gui_flip_x_axis_right:
                if xr > 0:
                    out_x = -abs(max(0.0, min(1.0, xr)))
                if xl > 0:
                    out_x = max(0.0, min(1.0, xl))
            else:
                if xr > 0:
                    out_x = max(0.0, min(1.0, xr))
                if xl > 0:
                    out_x = -abs(max(0.0, min(1.0, xl)))

            try:
                # 使用 One Euro Filter 平滑输出坐标
                noisy_point = np.array([out_x, out_y])
                point_hat = self.one_euro_filter(noisy_point)
                out_x = point_hat[0]
                out_y = point_hat[1]
            except:
                pass

            # 输出更新后的图像和眼动信息
            self.output_images_and_update(
                larger_threshold,
                EyeInformation(InformationOrigin.BLOB, out_x, out_y, 0, False),
            )
            return

    # 如果未成功处理轮廓，输出眨眼检测信息
    self.output_images_and_update(
        larger_threshold, EyeInformation(InformationOrigin.BLOB, 0, 0, 0, True)
    )
    print("[INFO] BLINK Detected.")


    def run(self):
        # 初始化 camera_model 和 detector_3d 为 None， pupil_dialation 设为 1
        camera_model = None
        detector_3d = None
        out_pupil_dialation = 1

        # 根据 eye_id 决定是否翻转 x 轴
        if self.eye_id == "EyeId.RIGHT":
            flipx = self.settings.gui_flip_x_axis_right
        else:
            flipx = self.settings.gui_flip_x_axis_left

        # 进入主循环
        while True:
            # 如果取消事件被触发，退出线程
            if self.cancellation_event.is_set():
                print("Exiting RANSAC thread")
                return

            # 如果 ROI 窗口的宽度或高度无效，等待用户在 GUI 中设置 ROI 窗口
            if self.config.roi_window_w <= 0 or self.config.roi_window_h <= 0:
                # 等待 0.1 秒再检查取消事件
                if self.cancellation_event.wait(0.1):
                    return
                continue

            # 如果相机模型或探测器未初始化，或 ROI 配置已更改，则重新初始化
            if (camera_model is None
                or detector_3d is None
                or camera_model.resolution != (
                    self.config.roi_window_w,
                    self.config.roi_window_h,
                )
            ):
                # 创建新的相机模型
                camera_model = CameraModel(
                    focal_length=self.config.focal_length,
                    resolution=(self.config.roi_window_w, self.config.roi_window_h),
                )
                # 创建新的 3D 探测器
                detector_3d = Detector3D(
                    camera=camera_model, long_term_mode=DetectorMode.blocking
                )

            try:
                # 如果输入的捕获队列为空，则设置捕获事件
                if self.capture_queue_incoming.empty():
                    self.capture_event.set()
                # 从队列中获取当前图像、帧号和帧率
                (
                    self.current_image,
                    self.current_frame_number,
                    self.current_fps,
                ) = self.capture_queue_incoming.get(block=True, timeout=0.2)
            except queue.Empty:
                # 如果队列为空且超时，则继续下一次循环
                continue

            # 如果捕获、裁剪或旋转图像失败，则继续下一次循环
            if not self.capture_crop_rotate_image():
                continue

            # 将图像转换为灰度图，并设置阈值处理。阈值基本上是一个低通滤波器，
            # 它会将任何低于阈值的像素值设置为0。由于光照的变化、摄像头位置和镜头的不同，
            # 用户可以配置阈值以适应不同情况。阈值设置的目标是确保我们只能看到瞳孔。
            # 这也是我们之前裁剪图像的原因，以减少可能引起混淆的暗区。
            self.current_image_gray = cv2.cvtColor(
                self.current_image, cv2.COLOR_BGR2GRAY
            )

            # 如果启用了圆形裁剪功能
            if self.config.gui_circular_crop == True:
                if self.cct == 0:
                    try:
                        # 获取图像的高度和宽度
                        ht, wd = self.current_image_gray.shape[:2]
                        # 计算圆的半径和中心点
                        radius = int(float(self.lkg_projected_sphere["axes"][0]))
                        self.xc = int(float(self.lkg_projected_sphere["center"][0]))
                        self.yc = int(float(self.lkg_projected_sphere["center"][1]))
                        # 在黑色背景上绘制白色实心圆作为掩码
                        mask = np.zeros((ht, wd), dtype=np.uint8)
                        mask = cv2.circle(mask, (self.xc, self.yc), radius, 255, -1)
                        # 创建白色背景
                        color = np.full_like(self.current_image_gray, (255))
                        # 将掩码应用到图像
                        masked_img = cv2.bitwise_and(self.current_image_gray, self.current_image_gray, mask=mask)
                        # 将逆掩码应用到彩色图像
                        masked_color = cv2.bitwise_and(color, color, mask=255 - mask)
                        # 合并两个掩码图像
                        self.current_image_gray = cv2.add(masked_img, masked_color)
                    except:
                        pass
                else:
                    self.cct = self.cct - 1
            else:
                # 如果没有启用圆形裁剪，则重置计数器
                self.cct = 300

            # 应用二值化操作，将图像转换为黑白图像。阈值是用户配置的，目的是将瞳孔区域突出显示。
            _, thresh = cv2.threshold(
                self.current_image_gray,
                int(self.config.threshold),
                255,
                cv2.THRESH_BINARY,
            )

            # 设置形态学转换，用于平滑和清理经过阈值处理后的图像。我们希望最终得到一个
            # 中间有黑色斑点，周围是白色区域的图像。
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            image = 255 - closing

            # 现在图像已经相对干净了，运行轮廓查找以获取瞳孔边界的2D轮廓。
            # 理想情况下，我们只希望得到一个边界。
            contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # 根据每个轮廓找到凸形状，并按面积从小到大排序。
            convex_hulls = []
            for i in range(len(contours)):
                convex_hulls.append(cv2.convexHull(contours[i], False))

            # 如果没有找到凸形状，我们就找不到瞳孔，无法继续。这时需要回退到使用
            # 斑点跟踪算法。
            if len(convex_hulls) == 0:
                if self.settings.gui_blob_fallback:
                    self.blob_tracking_fallback()
                else:
                    print("[INFO] Blob fallback disabled. Assuming blink.")
                    self.output_images_and_update(thresh, EyeInformation(InformationOrigin.RANSAC, 0, 0, 0, True))
                continue

            # 找到最大的凸形状，我们期望它是表示瞳孔的椭圆区域，可以作为眼睛的搜索区域。
            largest_hull = sorted(convex_hulls, key=cv2.contourArea)[-1]

            # 由于眼睛是三维的，所以我们需要将这个椭圆转化为球面（眼睛本身）上的曲面补丁。
            # 如果眼睛不是球形的，可能是由于散光的问题。
            try:
                cx, cy, w, h, theta = fit_rotated_ellipse_ransac(
                    largest_hull.reshape(-1, 2)
                )
            except:
                if self.settings.gui_blob_fallback:
                    self.blob_tracking_fallback()
                else:
                    print("[INFO] Blob fallback disabled. Assuming blink.")
                    self.output_images_and_update(thresh, EyeInformation(InformationOrigin.RANSAC, 0, 0, 0, True))
                continue

                # 获取椭圆的轴和角度，使用 pupil labs 的2D算法。接下来的代码可能有些神奇，
                # 因为大部分操作都在底层库中完成（因此通过字典传递）。
                result_2d = {}
                result_2d_final = {}

                # 设置2D检测结果的中心、轴和角度
                result_2d["center"] = (cx, cy)
                result_2d["axes"] = (w, h)
                result_2d["angle"] = theta * 180.0 / np.pi  # 将弧度转换为角度
                result_2d_final["ellipse"] = result_2d
                result_2d_final["diameter"] = w
                result_2d_final["location"] = (cx, cy)
                result_2d_final["confidence"] = 0.99
                result_2d_final["timestamp"] = self.current_frame_number / self.current_fps

                # 这里发生了“黑魔法”，我们得到了重投影后的瞳孔/眼睛，只需要“卖掉灵魂”给C++。
                result_3d = detector_3d.update_and_detect(
                    result_2d_final, self.current_image_gray
                )

                # 现在我们得到了3D瞳孔的椭圆
                ellipse_3d = result_3d["ellipse"]
                # 以及瞳孔所在的眼球表面
                self.lkg_projected_sphere = result_3d["projected_sphere"]

                # 记录瞳孔中心
                exm = ellipse_3d["center"][0]
                eym = ellipse_3d["center"][1]

                # 获取3D直径
                d = result_3d["diameter_3d"]

                # 如果校准帧计数器为0，完成校准并播放完成音效
                if self.calibration_frame_counter == 0:
                    self.calibration_frame_counter = None
                    self.xoff = cx
                    self.yoff = cy
                    if sys.platform.startswith("win"):
                        PlaySound('Audio/compleated.wav', SND_FILENAME | SND_ASYNC)
                # 如果正在校准，更新最大最小值
                elif self.calibration_frame_counter != None:
                    if exm > self.xmax:
                        self.xmax = exm
                    if exm < self.xmin:
                        self.xmin = exm
                    if eym > self.ymax:
                        self.ymax = eym
                    if eym < self.ymin:
                        self.ymin = eym
                    self.calibration_frame_counter -= 1

                # 如果需要重新居中眼睛，更新偏移值并播放完成音效
                if self.settings.gui_recenter_eyes:
                    self.xoff = cx
                    self.yoff = cy
                    if self.ts == 0:
                        self.settings.gui_recenter_eyes = False
                        if sys.platform.startswith("win"):
                            PlaySound('Audio/compleated.wav', SND_FILENAME | SND_ASYNC)
                    else:
                        self.ts = self.ts - 1
                else:
                    self.ts = 20

                # 计算瞳孔在x轴和y轴上的位置
                xl = float(
                    (cx - self.xoff) / (self.xmax - self.xoff)
                )
                xr = float(
                    (cx - self.xoff) / (self.xmin - self.xoff)
                )
                yu = float(
                    (cy - self.yoff) / (self.ymin - self.yoff)
                )
                yd = float(
                    (cy - self.yoff) / (self.ymax - self.yoff)
                )

                out_x = 0
                out_y = 0

                # 根据配置翻转y轴
                if self.settings.gui_flip_y_axis:
                    if yd > 0:
                        out_y = max(0.0, min(1.0, yd))
                    if yu > 0:
                        out_y = -abs(max(0.0, min(1.0, yu)))
                else:
                    if yd > 0:
                        out_y = -abs(max(0.0, min(1.0, yd)))
                    if yu > 0:
                        out_y = max(0.0, min(1.0, yu))

                # 根据配置翻转x轴
                if flipx:
                    if xr > 0:
                        out_x = -abs(max(0.0, min(1.0, xr)))
                    if xl > 0:
                        out_x = max(0.0, min(1.0, xl))
                else:
                    if xr > 0:
                        out_x = max(0.0, min(1.0, xr))
                    if xl > 0:
                        out_x = -abs(max(0.0, min(1.0, xl)))

                # 使用One Euro滤波器对输出值进行去噪处理
                try:
                    noisy_point = np.array([out_x, out_y])
                    point_hat = self.one_euro_filter(noisy_point)
                    out_x = point_hat[0]
                    out_y = point_hat[1]
                except:
                    pass

                # 创建 EyeInformation 对象，记录瞳孔的位置信息
                output_info = EyeInformation(InformationOrigin.RANSAC, out_x, out_y, out_pupil_dialation, False)

                # 绘制轮廓和瞳孔中心，用于视觉输出
                try:
                    cv2.drawContours(self.current_image_gray, contours, -1, (255, 0, 0), 1)
                    cv2.circle(self.current_image_gray, (int(cx), int(cy)), 2, (0, 0, 255), -1)
                except:
                    pass

                # 尝试绘制椭圆
                try:
                    cv2.ellipse(
                        self.current_image_gray,
                        tuple(int(v) for v in ellipse_3d["center"]),
                        tuple(int(v) for v in ellipse_3d["axes"]),
                        ellipse_3d["angle"],
                        0,
                        360,  # 绘制的起始/结束角度
                        (0, 255, 0),  # 颜色 (BGR): 红色
                    )
                except Exception:
                    # 有时我们会得到无效的轴，绘制时会出错。理想情况下我们应该提前检查有效性，
                    # 但目前只是跳过。通常下一帧就会修复。
                    pass

                # 绘制眼球表面的椭圆
                try:
                    cv2.ellipse(
                        self.current_image_gray,
                        tuple(int(v) for v in self.lkg_projected_sphere["center"]),
                        tuple(int(v) for v in self.lkg_projected_sphere["axes"]),
                        self.lkg_projected_sphere["angle"],
                        0,
                        360,  # 绘制的起始/结束角度
                        (0, 255, 0),  # 颜色 (BGR): 红色
                    )
                except:
                    pass

                # 从眼球中心到瞳孔中心画线
                cv2.line(
                    self.current_image_gray,
                    tuple(int(v) for v in self.lkg_projected_sphere["center"]),
                    tuple(int(v) for v in ellipse_3d["center"]),
                    (0, 255, 0),  # 颜色 (BGR): 红色
                )

                # 将处理后的图像传递到主GUI线程进行渲染
                self.output_images_and_update(thresh, output_info)