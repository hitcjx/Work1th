import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    output_image = np.ones_like(image) * 255

    if results.pose_landmarks:
      landmarks = results.pose_landmarks.landmark

      # 计算头部中心点（两个耳朵关键点的中点）
      left_ear_x = landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * image.shape[1]
      left_ear_y = landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * image.shape[0]
      right_ear_x = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * image.shape[1]
      right_ear_y = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y * image.shape[0]
      head_center_x = int((left_ear_x + right_ear_x) / 2)
      head_center_y = int((left_ear_y + right_ear_y) / 2)

      # 计算头部大小（根据肩膀宽度进行缩放）
      shoulder_distance = np.linalg.norm(np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]) -
                                        np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]))
      head_size = int(shoulder_distance * image.shape[1] * 0.3)

      # 绘制头部
      cv2.circle(output_image, (head_center_x, head_center_y), head_size, (0, 0, 0), -1)

      # 躯干连接
      connections = [
          (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
          (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
          (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
          (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
      ]

      # 绘制躯干 (加粗)
      for connection in connections:
        start_idx = connection[0].value
        end_idx = connection[1].value
        start_point = (int(landmarks[start_idx].x * image.shape[1]),
                      int(landmarks[start_idx].y * image.shape[0]))
        end_point = (int(landmarks[end_idx].x * image.shape[1]),
                    int(landmarks[end_idx].y * image.shape[0]))
        cv2.line(output_image, start_point, end_point, (0, 0, 0), 12)

      # 四肢和手脚连接
      limb_connections = [
          (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
          (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
          (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
          (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
          (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
          (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
          (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
          (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
          (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY), # 手
          (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY),  # 手
          (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX), # 脚
          (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),  # 脚
      ]

      # 绘制四肢和手脚 (加粗)
      for connection in limb_connections:
        start_idx = connection[0].value
        end_idx = connection[1].value
        start_point = (int(landmarks[start_idx].x * image.shape[1]),
                      int(landmarks[start_idx].y * image.shape[0]))
        end_point = (int(landmarks[end_idx].x * image.shape[1]),
                    int(landmarks[end_idx].y * image.shape[0]))
        cv2.line(output_image, start_point, end_point, (0, 0, 0), 4)

    cv2.imshow('MediaPipe Stick Figure', cv2.flip(output_image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()