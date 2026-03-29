import cv2

# 1. 加载 OpenCV 自带的人脸识别“说明书”（级联分类器）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 2. 将画面转为灰度图（“侦探”在黑白世界里找人脸更准更快）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. 寻找人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 4. 给找到的每一张脸画个框
    for (x, y, w, h) in faces:
        # 在脸部画一个蓝色方框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # 在框上方写个字
        cv2.putText(frame, "Target Acquired", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Face Tracking AR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()