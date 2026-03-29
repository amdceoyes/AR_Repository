import cv2
import mediapipe as mp
import sys

# 初始化 mp 解决方案
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 检查 OpenCV 是否真的装回来了
print(f"✅ OpenCV 版本: {cv2.__version__}")

# 调用摄像头 (DELL 笔记本建议加上 CAP_DSHOW)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ 错误：无法打开摄像头。")
    sys.exit()

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    print("🚀 AR 引擎已启动，正在捕获画面... (按 'q' 退出)")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        # 转换颜色空间并处理
        img_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # 绘图
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('SportCross REBUILT', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()