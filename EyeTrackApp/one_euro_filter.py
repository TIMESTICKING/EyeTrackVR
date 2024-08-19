import numpy as np
from time import time

'''
注释说明：

	•	smoothing_factor 函数用于根据当前时间间隔和截止频率计算平滑因子。
	•	exponential_smoothing 函数通过线性组合当前值和先前的值进行平滑处理。
	•	OneEuroFilter 类是一个实时滤波器，能够对输入的多维信号进行平滑处理，同时调整其响应速度。
	•	代码中的每一步都附有注释，以便理解每个部分的作用。
'''


# 计算平滑因子，根据时间间隔和截止频率
def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)

# 指数平滑函数，用于平滑当前值和上一次的值
def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

# OneEuroFilter 类，实时平滑滤波器
class OneEuroFilter:
    def __init__(self, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """初始化 One Euro Filter."""
        # 输入数据的形状，用于确保输入数据维度一致
        self.data_shape = x0.shape
        
        # 滤波器参数
        self.min_cutoff = np.full(x0.shape, min_cutoff)  # 最小截止频率
        self.beta = np.full(x0.shape, beta)  # 调节因子，控制响应速度
        self.d_cutoff = np.full(x0.shape, d_cutoff)  # 导数的截止频率

        # 保存上一次的值
        self.x_prev = x0.astype(np.float)  # 上一次的输入信号
        self.dx_prev = np.full(x0.shape, dx0)  # 上一次的导数值
        self.t_prev = time()  # 上一次的时间戳

    def __call__(self, x):
        """对输入信号进行滤波处理。"""
        assert x.shape == self.data_shape  # 确保输入信号形状与初始化时一致

        # 当前时间
        t = time()
        # 时间间隔，计算从上一次到现在的时间差
        t_e = t - self.t_prev

        # 确保时间间隔不是零，避免除以零的错误
        if t_e != 0.0: 
            t_e = np.full(x.shape, t_e)

            # 平滑信号的导数
            a_d = smoothing_factor(t_e, self.d_cutoff)  # 计算导数的平滑因子
            dx = (x - self.x_prev) / t_e  # 计算导数
            dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)  # 对导数进行平滑

            # 根据平滑后的导数调整截止频率
            cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
            a = smoothing_factor(t_e, cutoff)  # 计算信号的平滑因子
            x_hat = exponential_smoothing(a, x, self.x_prev)  # 对输入信号进行平滑

            # 更新保存的值，为下次调用做准备
            self.x_prev = x_hat  # 更新上一次的平滑信号
            self.dx_prev = dx_hat  # 更新上一次的平滑导数
            self.t_prev = t  # 更新上一次的时间戳

            return x_hat  # 返回平滑后的信号