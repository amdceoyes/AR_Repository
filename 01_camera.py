import cv2

# 1. 打开摄像头 (0 通常是笔记本内置摄像头)
cap = cv2.VideoCapture(0)

print("AR 实验室启动！按下 'q' 键退出预览。")

while True:
    # 2. 逐帧读取画面
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 在画面中心画一个绿色的 AR 准星 (证明我们能操控像素)
    h, w, _ = frame.shape
    cv2.drawMarker(frame, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)

    # 4. 显示窗口
    cv2.imshow('My_First_AR_Window', frame)

    # 5. 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()