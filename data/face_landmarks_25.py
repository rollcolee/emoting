# 표정 : 이미지데이터의 좌표값 추출한 후 csv파일로 만들기

import cv2
import os
import pandas as pd
import mediapipe as mp

train_folder = '/content/extracted_images/train'
test_folder = '/content/extracted_images/test'

# Mediapipe Face 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Mediapipe에서 제공하는 landmark 중 25개 추출
landmark_names = {
    33: 'Left Eye Bottom',
    133: 'Right Eye Bottom',
    61: 'Left Cheek',
    291: 'Right Cheek',
    152: 'Chin',
    70: 'Left Eyebrow Outer',
    300: 'Right Eyebrow Outer',
    362: 'Right Mouth Corner',
    287: 'Lower Lip',
    127: 'Right Eyebrow Inner',
    356: 'Right Eye Inner Corner',
    21: 'Left Eyebrow Inner',
    234: 'Left Mouth Corner',
    107: 'Right Eye Top',
    10: 'Left Eye Top',
    55: 'Left Eye Inner Corner',
    282: 'Right Lip Corner',
    105: 'Right Eye Outer Corner',
    334: 'Left Lip Corner',
    93: 'Upper Lip',
    23: 'Left Eyebrow Outer',
    454: 'Right Eyebrow Inner',
    226: 'Left Eyebrow Bottom',
    276: 'Nose Tip',
    446: 'Right Eyebrow Bottom'
}
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

                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        landmark_coords = {}

                        for i, landmark in enumerate(landmarks.landmark):
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
landmarks_df.to_csv('/content/facial.csv', index=False)

print("CSV 파일 저장 완료!")
