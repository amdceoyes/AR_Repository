import cv2
import mediapipe as mp
import sys

# 强制检测引擎
try:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    print("✅ 引擎核心加载成功")
except AttributeError:
    print("❌ 严重错误：检测到环境污染。请尝试在终端输入：pip uninstall mediapipe 然后重新 pip install mediapipe")
    sys.exit()

# 强制开启摄像头（增加 DSHOW 协议，这是 Windows 系统的特效药）
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 如果 0 号不行，自动尝试 1 号
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 降低分辨率以保证流畅度，适合你的笔记本开发
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

print("🚀 正在激活 AR 窗口...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue # 没画面就死等，不报错
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("SportCross AR Lab", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()