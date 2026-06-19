# 眼镜端配置
TARGET_HOST = '192.168.1.100' # 计算盒的局域网 IP
TARGET_PORT = 9999

from .render_core import RenderCore
from .FSM_glasses import GlassesFSM
from .Transport_glasses import GlassesTransport