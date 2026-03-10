import cv2
import mediapipe as mp
import numpy as np   # 각도 계산을 위해 numpy 사용

# mediapipe 준비
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ------------------------------
# 세 점을 이용해 각도를 계산하는 함수
# ------------------------------
def calculate_angle(a, b, c):

    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중간 점 (각도를 계산할 관절)
    c = np.array(c)  # 세 번째 점

    # 두 벡터 사이의 각도 계산
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians*180.0/np.pi)

    # 각도가 180도를 넘으면 보정
    if angle > 180.0:
        angle = 360-angle

    return angle


# 웹캠 열기
cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # 왼쪽 엉덩이 / 무릎 / 발목 좌표
            hip = [landmarks[23].x, landmarks[23].y]
            knee = [landmarks[25].x, landmarks[25].y]
            ankle = [landmarks[27].x, landmarks[27].y]

            # 무릎 각도 계산
            angle = calculate_angle(hip, knee, ankle)

            print("무릎 각도:", angle)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()