import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import xgboost as xgb
import json
from collections import Counter
from google.colab.patches import cv2_imshow  # Colab에서 이미지 출력용

# ✅ Google Drive에서 모델 로드
pose_model = xgb.XGBClassifier()
pose_model.load_model("/content/drive/MyDrive/2차프로젝트/자세모델/pose_classification_model.json")

# ✅ JSON 파일 로드
with open("/content/drive/MyDrive/2차프로젝트/표정모델/face_expression_model.json", "r", encoding="utf-8") as f:
    expression_model_data = json.load(f)

with open("/content/drive/MyDrive/2차프로젝트/자세모델/label_encoder.json", "r", encoding="utf-8") as f:
    pose_label_map = json.load(f)

with open("/content/drive/MyDrive/2차프로젝트/표정모델/face_label_encoder.json", "r", encoding="utf-8") as f:
    expression_label_map = json.load(f)

# ✅ MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()

# ✅ 자세 및 표정 카운트
pose_counter = Counter()
expression_counter = Counter()

def extract_pose_landmarks(frame):
    """ 프레임에서 자세의 랜드마크를 추출하여 모델 입력 형식으로 변환 """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark]).flatten()
        landmarks = landmarks[:93]  # 모델 입력 크기 맞춤
        return landmarks
    return None

def extract_face_landmarks(frame):
    """ 프레임에서 얼굴의 랜드마크를 추출하여 모델 입력 형식으로 변환 """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in face.landmark][:25]).flatten()
    return None

def classify_pose(landmarks):
    """ 자세를 분류하고 카운트 """
    if landmarks is not None:
        pred = pose_model.predict([landmarks])[0]
        label = pose_label_map.get(str(pred), "Unknown")
        pose_counter[label] += 1
        return label
    return "Unknown"

def classify_expression(landmarks):
    """ 표정을 분류하고 카운트 """
    if landmarks is not None:
        if "learner" in expression_model_data:
            pred = str(int(expression_model_data["learner"].get(str(landmarks.tolist()), -1)))
            label = expression_label_map.get(pred, "Unknown")
            expression_counter[label] += 1
            return label
        else:
            return "Unknown"
    return "Unknown"

def process_video(video_path, output_path):
    """ 비디오에서 자세 및 표정을 분석하고 결과를 저장 """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"⚠️ 비디오 파일을 열 수 없습니다: {video_path}")
        return

    # ✅ 동영상 저장 설정 (MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 🔄 프레임 180도 회전 (Colab에서 거꾸로 출력 방지)
        frame = cv2.flip(frame, 0)

        pose_landmarks = extract_pose_landmarks(frame)
        face_landmarks = extract_face_landmarks(frame)

        pose_label = classify_pose(pose_landmarks)
        expression_label = classify_expression(face_landmarks)

        # ✅ 비디오에 텍스트 추가 (자세 & 표정)
        cv2.putText(frame, f"Pose: {pose_label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Expression: {expression_label}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)  # ✅ 동영상 저장

    cap.release()
    out.release()

    # ✅ CSV 저장 (데이터 길이 맞춤)
    max_length = max(len(pose_counter), len(expression_counter))
    pose_labels = list(pose_counter.keys()) + ["Unknown"] * (max_length - len(pose_counter))
    pose_counts = list(pose_counter.values()) + [0] * (max_length - len(pose_counter))
    expression_labels = list(expression_counter.keys()) + ["Unknown"] * (max_length - len(expression_counter))
    expression_counts = list(expression_counter.values()) + [0] * (max_length - len(expression_counter))

    result_df = pd.DataFrame({"Pose": pose_labels, "Pose Count": pose_counts,
                              "Expression": expression_labels, "Expression Count": expression_counts})
    result_df.to_csv("/content/pose_expression_count.csv", index=False)

    print("✅ 결과 동영상과 CSV 저장 완료!")

# ✅ 사용 예시 (Google Drive에서 비디오 읽고 Colab에 저장)
process_video("/content/drive/MyDrive/2차프로젝트/IMG_1493 (1).mov", "/content/processed_output.mp4")

# ✅ Colab에서 다운로드 (선택 사항)
from google.colab import files
files.download('/content/processed_output.mp4')  # 동영상 다운로드
files.download('/content/pose_expression_count.csv')  # CSV 다운로드
