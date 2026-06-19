# 根目录的 __init__.py 负责让整个工程变成一个可调用的整体
# 这样你在 launcher 中就可以直接 import 整个系统的核心组件

from cloud_ai.cloud_runtime import CloudRuntime
from computer_box.box_runtime import BoxRuntime
from glasses.glass_runtime import GlassRuntime

# 甚至可以在这里定义整个系统的全局调试开关
DEBUG_MODE = True
VERSION = "1.0.0"

print(f"[System] AR 分布式引擎 v{VERSION} 已加载.")