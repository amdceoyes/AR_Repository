# 导入必要的Python库
import multiprocessing  # 多进程库，用于并行运行三个独立组件
import time  # 时间库，用于延时控制

# 从自定义模块导入三个运行时组件
from cloud_ai.cloud_runtime import CloudRuntime  # 云端AI运行时
from computer_box.box_runtime import BoxRuntime   # 计算盒运行时
from glasses.glass_runtime import GlassRuntime    # 眼镜端运行时

def start_cloud():
    """
    启动云端服务器进程
    
    这个函数在独立的进程中运行，负责：
    1. 加载AI模型
    2. 启动云端服务
    3. 监听计算盒的连接请求
    4. 处理AI推理任务
    """
    # 创建CloudRuntime实例，传入ONNX模型路径
    # 参数说明：
    #   'models/mobilenet.onnx': ONNX模型文件路径
    runtime = CloudRuntime('models/mobilenet.onnx')
    
    # 启动云端服务（阻塞调用，会一直运行直到被终止）
    runtime.run()

def start_box():
    """
    启动计算盒进程
    
    这个函数在独立的进程中运行，负责：
    1. 初始化摄像头和传感器
    2. 启动视觉和SLAM处理
    3. 连接到云端服务器
    4. 与眼镜端通信
    """
    # 创建BoxRuntime实例
    # 参数说明：
    #   camera_matrix: 相机内参矩阵，这里传入None（应传入实际的相机内参）
    # 注意：camera_matrix=None会导致错误，应该传入实际的相机内参矩阵
    runtime = BoxRuntime(camera_matrix=None) 
    
    # 启动计算盒（阻塞调用，会一直运行直到被终止）
    runtime.run()

def start_glasses():
    """
    启动眼镜端进程
    
    这个函数在独立的进程中运行，负责：
    1. 连接到计算盒
    2. 接收位姿和AI结果
    3. 渲染AR内容
    4. 管理显示状态
    """
    # 创建GlassRuntime实例
    runtime = GlassRuntime()
    
    # 启动眼镜端（阻塞调用，会一直运行直到被终止）
    runtime.run()

# 主程序入口
# 当这个文件被直接运行时，执行以下代码
if __name__ == "__main__":
    """
    分布式AR系统主启动脚本
    
    这个脚本协调三个组件的启动：
    1. 云端服务器（AI服务）
    2. 计算盒（本地处理）
    3. 眼镜端（显示）
    
    启动顺序很重要：
    1. 先启动云端（服务器）
    2. 再启动计算盒（客户端，连接到云端）
    3. 最后启动眼镜端（客户端，连接到计算盒）
    
    这样可以避免客户端尝试连接不存在的服务器
    """
    # 打印系统启动信息
    print("[System] 正在启动分布式 AR 系统...")

    # 创建三个独立进程
    # multiprocessing.Process: 创建新进程
    # 参数说明：
    #   target: 进程要执行的函数
    # 每个组件在独立的进程中运行，实现真正的分布式计算
    p1 = multiprocessing.Process(target=start_cloud)  # 云端进程
    p2 = multiprocessing.Process(target=start_box)    # 计算盒进程
    p3 = multiprocessing.Process(target=start_glasses)  # 眼镜端进程

    # 启动顺序：先启云端(服务器)，再启计算盒，最后启眼镜(客户端)
    # 1. 启动云端服务器
    p1.start()
    
    # 等待2秒，确保云端服务器完全启动并开始监听端口
    # 这是重要的延时，给服务器足够的时间初始化
    time.sleep(2)
    
    # 2. 启动计算盒
    p2.start()
    
    # 3. 启动眼镜端
    p3.start()
    
    # 打印系统启动完成信息
    print("[System] 分布式 AR 链路已全部点火，系统运行中...")

    # 保持主进程活跃
    # 使用try-except捕获键盘中断，实现优雅关机
    try:
        # 等待所有进程结束
        # join()是阻塞调用，会等待进程结束
        p1.join()  # 等待云端进程
        p2.join()  # 等待计算盒进程
        p3.join()  # 等待眼镜端进程
        
    except KeyboardInterrupt:
        # 捕获Ctrl+C中断
        print("\n[System] 检测到终止信号，正在关闭系统...")
        
        # 终止所有进程
        # terminate()会发送终止信号给进程
        p1.terminate()  # 终止云端进程
        p2.terminate()  # 终止计算盒进程
        p3.terminate()  # 终止眼镜端进程
        
        # 注意：terminate()是强制终止，不会执行清理代码
        # 实际应用中可能需要更优雅的关机方式