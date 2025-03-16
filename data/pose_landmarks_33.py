# 행동 : 이미지데이터의 좌표값 추출한 후 csv파일로 만들기

import cv2
import mediapipe as mp
import pandas as pd
import os
from google.colab import files

train_folder = '/content/drive/MyDrive/extracted_images/3000/train'
test_folder = '/content/drive/MyDrive/extracted_images/3000/test'

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,  # 이미지 처리 모드 설정
                    model_complexity=1,      # 모델 복잡도 설정
                    enable_segmentation=False, # 분할 활성화 여부
                    min_detection_confidence=0.5, # 검출 신뢰도 최소값 설정
                    min_tracking_confidence=0.5)  # 추적 신뢰도 최소값 설정

# Mediapipe에서 제공하는 33개 관절 이름
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel"
]

# 랜드마크 데이터를 저장할 리스트
landmarks_data = []

# train과 test 폴더 순회
for folder in [train_folder, test_folder]:
    dataset_type = os.path.basename(folder)  # train/test 구분
    label_folders = os.listdir(folder)

    for label in label_folders:
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            image_files = [f for f in os.listdir(label_folder) if f.endswith(('.jpg', '.png'))]

            for image_file in image_files:
                image_path = os.path.join(label_folder, image_file)
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    landmark_coords = {}

                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        if i in landmark_names:
                            landmark_coords[f'{landmark_names[i]}_x'] = landmark.x
                            landmark_coords[f'{landmark_names[i]}_y'] = landmark.y
                            landmark_coords[f'{landmark_names[i]}_z'] = landmark.z

                    landmark_coords['image'] = image_file
                    landmark_coords['label'] = label
                    landmark_coords['dataset'] = dataset_type  # train/test 구분 추가
                    landmarks_data.append(landmark_coords)

# 데이터프레임 생성 및 CSV 저장
landmarks_df = pd.DataFrame(landmarks_data)
landmarks_df.to_csv('/content/pose_landmarks.csv', index=False)  # 파일명 변경

print("CSV 파일 저장 완료!")
