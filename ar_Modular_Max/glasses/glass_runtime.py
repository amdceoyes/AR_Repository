# 导入必要的Python库
import time  # 时间库，用于帧率控制和延时

# 从自定义模块导入组件
from Transport_glasses import GlassesTransport  # 眼镜传输层，负责与计算盒通信
from FSM_glasses import GlassesFSM              # 眼镜状态机，管理显示状态
from render_core import RenderCore               # 渲染核心，负责绘制AR内容

class GlassRuntime:
    """
    眼镜运行时主类，负责协调眼镜端的显示和交互逻辑
    
    这个类是眼镜端的核心，它：
    1. 初始化传输层、状态机和渲染器
    2. 从计算盒接收位姿和AI结果
    3. 根据状态机决定渲染策略
    4. 控制显示帧率
    
    眼镜作为显示终端，主要负责渲染AR内容
    """
    
    def __init__(self):
        """
        初始化眼镜运行时系统
        
        创建传输层、状态机和渲染器的实例
        注意：192.168.1.100是计算盒的IP地址，需要根据实际情况配置
        """
        # 创建传输层实例，连接到计算盒
        # 参数说明：
        #   host='192.168.1.100': 计算盒的IP地址
        #   port=9999: 计算盒的端口号
        self.transport = GlassesTransport(host='192.168.1.100', port=9999)
        
        # 创建状态机实例，管理眼镜显示状态
        self.fsm = GlassesFSM()
        
        # 创建渲染核心实例，负责绘制AR内容
        self.renderer = RenderCore()
        
    def run(self):
        """
        运行眼镜主循环
        
        这个方法实现了眼镜的主要工作流程：
        1. 连接到计算盒
        2. 持续接收数据
        3. 根据状态机决定渲染策略
        4. 维持稳定的帧率
        
        这个循环会一直运行，直到程序被中断
        """
        # 打印启动信息
        print("[Glass] 眼镜显示系统已启动...")
        
        # 连接到计算盒
        self.transport.connect()
        
        # 主循环，持续运行直到程序终止
        while True:
            # 1. 从计算盒接收数据 (Pose + AI结果)
            # 接收来自计算盒的数据，包含位姿和AI识别结果
            data = self.transport.receive_data()
            
            # 如果接收到数据
            if data:
                # 2. 状态机同步
                # 通知状态机SLAM成功（跟踪正常）
                self.fsm.update_render('SLAM_SUCCESS')
                
                # 解析数据并绘制
                # 假设数据是字典格式，包含'label'和'pose'字段
                # 实际数据格式需要与计算盒协商一致
                self.renderer.draw_object(data['label'], data['pose'])
            else:
                # 3. 如果连接丢失或无数据，进入丢帧状态
                # 通知状态机SLAM失败（跟踪丢失）
                self.fsm.update_render('SLAM_FAILED')
                
                # 清空屏幕
                self.renderer.clear()
            
            # 维持帧率控制（例如 60fps）
            # 每帧休眠1/60秒，约16.7毫秒
            time.sleep(1/60)
            
            # 注意：这个帧率控制是固定的
            # 实际应用中，可能需要动态调整帧率
            # 或者使用更精确的定时器

# 主程序入口
# 当这个文件被直接运行时，执行以下代码
if __name__ == "__main__":
    # 创建GlassRuntime实例
    runtime = GlassRuntime()
    
    # 运行眼镜系统
    runtime.run()