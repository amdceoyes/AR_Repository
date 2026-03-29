import cv2

# 启动摄像头
cap = cv2.VideoCapture(0)

print("AR 实验室启动中... 按下 'q' 键退出预览")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 在画面中心画一个准星，证明我们拿到了像素控制权
    h, w, _ = frame.shape
    cv2.drawMarker(frame, (w//2, h//2), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

    cv2.imshow('AR_Core_Test', frame)

    # 检测按键，按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()