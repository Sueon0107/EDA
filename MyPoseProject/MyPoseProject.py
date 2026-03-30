import time  # 시간 간격을 계산하기 위한 모듈
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


# 웹캠 열기 0=기본, 1=외장 웹캠
cap = cv2.VideoCapture(0)
#스마트폰 droidcam 앱 사용
#cap = cv2.VideoCapture("http://192.168.0.20:4747/video")
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 영상 버퍼 최소화

with mp_pose.Pose() as pose:
    
    last_update_time = 0  # 마지막으로 각도를 업데이트한 시간
    update_interval = 0.1  # 업데이트 간격 (0.1초 = 1초에 10번)
    angle = 0  # 무릎 각도 저장 변수 
       
    counter = 0      # 횟수 저장
    stage = None     # 현재 상태 (up/down)

    last_detected_time = time.time()   # 마지막 사람 인식 시간
    reset_time = 5   # 사람이 5초 이상 안 보이면 초기화 (초 단위)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:

            # 사람이 인식되었으므로 현재 시간을 기록
            last_detected_time = time.time()

            landmarks = results.pose_landmarks.landmark
            cv2.putText(
            image,
            f"Knee Angle: {int(angle)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
            )
            
            # 왼쪽 엉덩이 / 무릎 / 발목 좌표
            hip = [landmarks[23].x, landmarks[23].y]
            knee = [landmarks[25].x, landmarks[25].y]
            ankle = [landmarks[27].x, landmarks[27].y]

           
            current_time = time.time()  # 현재 시간 가져오기


          # 마지막 업데이트 이후 0.1초 이상 지났을 때만 각도 업데이트
            if current_time - last_update_time >= update_interval:
                angle = calculate_angle(hip, knee, ankle)
                last_update_time = current_time  # 마지막 업데이트 시간 갱신

                # ------------------------------
                # 무릎 카운트 로직 ⭐ 핵심
                # ------------------------------

                # 무릎이 펴진 상태
                if angle > 160:
                    stage = "up"

                # 무릎이 굽혀지면 카운트 증가
                if angle < 90 and stage == "up":
                    stage = "down"
                    counter += 1

            print(angle) 
            # ------------------------------
            # 화면에 무릎 각도 표시
            # ------------------------------
            cv2.putText(
                image,
                f"Knee Angle: {int(angle)}",   # 화면에 표시할 텍스트
                (50, 50),                     # 화면 위치 (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,     # 글꼴
                1,                            # 글자 크기
                (0, 255, 0),                  # 글자 색 (초록)
                2                             # 글자 두께
            )
            # ------------------------------
            # 횟수 표시 (Count)
            # ------------------------------
            cv2.putText(
                image,
                f"Count: {counter}",
                (50, 100),                 # 각도 아래쪽 위치
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),               # 파란색
                2
            )

            # ------------------------------
            # 현재 상태 표시 (Stage)
            # ------------------------------
            cv2.putText(
                image,
                f"Stage: {stage}",
                (50, 150),                 # Count 아래쪽 위치
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),               # 빨간색
                2
            )
            # MediaPipe 포즈 랜드마크를 화면에 항상 그리기
            mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
            )

        else:
            # 현재 시간 가져오기
            current_time = time.time()

            # 마지막 인식 이후 시간이 5초 이상 지났으면
            if current_time - last_detected_time > reset_time:

                # ----------- 초기화 -----------

                counter = 0
                stage = None

                print("5초 이상 사람 미인식 → 카운트 초기화")

        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()