# 导入NumPy库，用于高效的数值计算
import numpy as np

class IMUCore:
    """
    IMU核心类，负责处理惯性测量单元（IMU）数据
    
    IMU通常包含两个传感器：
    1. 加速度计（Accelerometer）：测量线性加速度
    2. 陀螺仪（Gyroscope）：测量角速度
    
    这个类的主要功能：
    1. 数据预处理和滤波
    2. 传感器偏差校准
    3. 时间同步和积分
    4. 运动稳定性检测
    """
    
    def __init__(self):
        """
        初始化IMU核心
        
        设置初始状态变量：
        1. 陀螺仪零偏：传感器静止时的读数
        2. 时间戳：用于计算时间间隔
        3. 可能的滤波器状态
        
        在实际应用中，初始化时通常需要：
        1. 校准传感器
        2. 估计初始姿态
        3. 配置滤波器参数
        """
        # 陀螺仪漂移校准值（通常在初始化时测量）
        # 传感器即使静止时也会有非零输出，这就是零偏
        # 在IMU静止时，测量一段时间的数据，取平均值作为零偏
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        
        # 上一次的时间戳，用于计算时间间隔
        # 积分需要知道时间间隔：角速度积分得到角度，加速度积分得到速度
        self.last_timestamp = 0
        
        # 打印初始化信息
        print("[IMU] IMUCore 已启动，高频数据监听中...")
        
        # 注意：这里缺少加速度计的零偏校准
        # 加速度计通常也有零偏，特别是低成本的MEMS传感器

    def process_imu_data(self, raw_acc, raw_gyro, timestamp):
        """
        处理IMU原始数据
        
        参数:
        raw_acc: 加速度计原始数据，[x, y, z]，单位通常是g或m/s²
        raw_gyro: 陀螺仪原始数据，[x, y, z]，单位通常是度/秒或弧度/秒
        timestamp: 数据的时间戳，单位通常是秒或毫秒
        
        返回:
        字典，包含处理后的数据：
        - linear_acc: 处理后的加速度
        - angular_vel: 处理后的角速度
        - is_stable: 运动是否稳定
        
        这个函数实现了IMU数据处理的基本流程：
        1. 数据预处理（滤波）
        2. 零偏补偿
        3. 时间间隔计算
        4. 稳定性检测
        """
        # 1. 数据预处理：简单的低通滤波（防止传感器原生抖动）
        # 低通滤波器去除高频噪声，保留低频信号
        acc = self._low_pass_filter(raw_acc)
        
        # 陀螺仪数据去除零偏
        # 从原始数据中减去零偏，得到真实的角速度
        gyro = np.array(raw_gyro) - self.gyro_bias
        
        # 2. 计算角速度带来的姿态变化（积分运算）
        # 计算时间间隔，用于积分
        dt = timestamp - self.last_timestamp
        
        # 更新时间戳
        self.last_timestamp = timestamp
        
        # 3. 封装：此时数据已清洗，可供PoseCore或MapEngine调用
        return {
            "linear_acc": acc,      # 处理后的加速度
            "angular_vel": gyro,    # 处理后的角速度
            "is_stable": self._check_stability(acc, gyro)  # 稳定性状态
        }
        
        # 注意：这里没有实际进行积分计算
        # 在实际应用中，通常会在这里进行姿态更新（四元数、旋转矩阵等）

    def _low_pass_filter(self, data, alpha=0.2):
        """
        低通滤波器：平滑传感器输出
        
        参数:
        data: 输入数据
        alpha: 滤波系数，范围0-1
              alpha越小，滤波越强（更平滑但延迟更大）
              alpha越大，滤波越弱（响应更快但噪声更多）
        
        返回:
        滤波后的数据
        
        一阶低通滤波器公式：
        output = alpha * input + (1 - alpha) * previous_output
        
        注意：原代码有问题，没有保存上一次的输出
        简化示例中使用了0作为上一次的输出，这会导致滤波效果不佳
        """
        # 原代码：return alpha * np.array(data) + (1 - alpha) * 0
        
        # 应该保存上一次的滤波结果
        if not hasattr(self, '_last_filtered_acc'):
            # 第一次调用，没有历史数据
            self._last_filtered_acc = np.zeros(3)
        
        # 应用低通滤波
        filtered = alpha * np.array(data) + (1 - alpha) * self._last_filtered_acc
        self._last_filtered_acc = filtered
        
        return filtered

    def _check_stability(self, acc, gyro):
        """
        稳定性检测：如果变化剧烈，通知上层视觉可能已失效
        
        参数:
        acc: 加速度数据
        gyro: 角速度数据
        
        返回:
        bool: 如果运动稳定返回True，否则返回False
        
        稳定性检测的常用方法：
        1. 角速度幅值阈值
        2. 加速度方差检测
        3. 零速检测（Zero-Velocity Update, ZUPT）
        4. 运动强度检测
        
        这里使用简化的角速度幅值检测
        """
        # 计算角速度的L2范数（幅值）
        gyro_norm = np.linalg.norm(gyro)
        
        # 阈值0.5是经验值，需要根据具体应用调整
        # 如果角速度幅值小于0.5，认为是稳定状态
        return gyro_norm < 0.5