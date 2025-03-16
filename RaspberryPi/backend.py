# 동작 : 좌표 학습모델 사용
# 표정 : CNN 이미지 학습모델 사용

from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
import json
import joblib
from sklearn.preprocessing import LabelEncoder
import time
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# 카운트 변수 초기화
pose_counts = {
    'crossed_arm': 0,
    'crossed_leg': 0,
    'smile_cover': 0,
    'touching_chin': 0,
    'hair_comb': 0,
    'drinking': 0,
    'eating': 0,
}

expression_counts = {
    'sad': 0,
    'happy': 0,
    'neutral': 0,
    'disgust': 0,
    'surprise': 0,
    'angry': 0,
}

pose_weights = {
    'crossed_arm': -3,
    'crossed_leg': -1.5,
    'smile_cover': +2,
    'touching_chin': +1,
    'hair_comb': 1.5,
    'drinking': 0,
    'eating': 0
}

face_weights = {
    'angry': -3,
    'disgust': -2,
    'fear': -1,
    'happy': +3,
    'neutral': 0,
    'sad': -1,
    'surprise': 0
}

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.4, min_tracking_confidence=0.4)

# pose_model
pose_model = xgb.XGBClassifier()
pose_model.load_model("pose_classification_model.json")

# face_model
face_model_path = "/content/norm100_CNN_face_expression_model.keras"  # 2번 코드에서 사용하는 모델 경로로 변경
try:
    face_model = keras.models.load_model(face_model_path)
    print("얼굴 표정 모델 로드 성공")
except Exception as e:
    print(f"얼굴 표정 모델 로드 실패: {e}")
    exit()

# pose_label_mapping
with open("label_encoder.json", "r") as f:
    pose_label_mapping = json.load(f)

# pose_label_encoder
pose_label_encoder = LabelEncoder()
pose_label_encoder.classes_ = np.array(list(pose_label_mapping.values()))

# face_label_encoder
face_label_encoder_path = "/content/norm100_CNN_face_label_encoder.json"  # 2번 코드에서 사용하는 라벨 인코더 경로로 변경
try:
    with open(face_label_encoder_path, "r") as f:
        face_label_mapping = json.load(f)
    face_label_mapping = {value: key for key, value in face_label_mapping.items()}
    print(f"얼굴 표정 레이블 매핑: {face_label_mapping}")
except Exception as e:
    print(f"얼굴 표정 레이블 매핑 로드 실패: {e}")
    exit()

scaler = joblib.load("half_scaler_face_mesh.pkl")

pose_selected_indices = [i for i in range(31)]

face_landmark_indices = [
    33, 263, 50, 280, 152, 46, 276, 133, 362, 243, 466, 159, 386,
    145, 374, 70, 300, 105, 334, 61, 291, 0, 17, 78, 308
]

# 최종 점수 계산을 위한 변수
final_score = 100

# 카운트 증가 간격 설정 (단위: 초)
pose_update_interval = 2
face_update_interval = 2

# 마지막 업데이트 시간 추적
last_pose_update_time = time.time()
last_face_update_time = time.time()

# 2번 코드에서 추가된 함수들 (표정 예측 부분에 사용)
def preprocess_face(face_image, target_size=(96, 96)):
    """얼굴 영역을 전처리하고 모델 입력 형태로 변환하는 함수."""
    resized_face = cv2.resize(face_image, target_size)
    normalized_face = resized_face / 255.0
    normalized_face = np.expand_dims(normalized_face, axis=0)
    return normalized_face

def predict_expression(face_image, label_mapping):
    processed_data = preprocess_face(face_image)
    predictions = face_model.predict(processed_data)
    predicted_label = np.argmax(predictions)
    expression = label_mapping.get(predicted_label, "unknown")
    return expression

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x-25, y-25), (x+w+25, y+h+25), (0, 255, 0), 2)

def generate_frames():
    global final_score, last_pose_update_time, last_face_update_time
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 포즈 예측
        pose_results = pose.process(image)
        pose_predicted_text = "Pose Untracked"
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            pose_keypoints = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in pose_selected_indices]).flatten()

            pose_probs = pose_model.predict_proba([pose_keypoints])[0]
            sorted_indices = np.argsort(pose_probs)[::-1]
            pose_top_label_1 = pose_label_encoder.inverse_transform([sorted_indices[0]])[0]
            pose_top_conf_1 = pose_probs[sorted_indices[0]]

            pose_top_label_2 = pose_label_encoder.inverse_transform([sorted_indices[1]])[0]
            pose_top_conf_2 = pose_probs[sorted_indices[1]]

            if pose_top_conf_1 >= 0.6:
                pose_predicted_text = f"{pose_top_label_1} ({pose_top_conf_1:.2f})"
                if pose_top_conf_2 >= 0.4:
                    pose_predicted_text += f" {pose_top_label_2} ({pose_top_conf_2:.2f})"

                current_time = time.time()
                if current_time - last_pose_update_time >= pose_update_interval:
                    if pose_top_label_1 == 'crossed_arm' and pose_top_conf_1 >= 0.6:
                        pose_counts['crossed_arm'] += 1
                        final_score += pose_weights['crossed_arm']
                    if pose_top_label_1 == 'crossed_leg' and pose_top_conf_1 >= 0.6:
                        pose_counts['crossed_leg'] += 1
                        final_score += pose_weights['crossed_leg']
                    if pose_top_label_1 == 'smile_cover' and pose_top_conf_1 >= 0.6:
                        pose_counts['smile_cover'] += 1
                        final_score += pose_weights['smile_cover']
                    if pose_top_label_1 == 'touching_chin' and pose_top_conf_1 >= 0.6:
                        pose_counts['touching_chin'] += 1
                        final_score += pose_weights['touching_chin']
                    if pose_top_label_1 == 'hair_comb' and pose_top_conf_1 >= 0.6:
                        pose_counts['hair_comb'] += 1
                        final_score += pose_weights['hair_comb']
                    if pose_top_label_1 == 'drinking' and pose_top_conf_1 >= 0.6:
                        pose_counts['drinking'] += 1
                        final_score += pose_weights['drinking']
                    if pose_top_label_1 == 'eating' and pose_top_conf_1 >= 0.6:
                        pose_counts['eating'] += 1
                        final_score += pose_weights['eating']

                    last_pose_update_time = current_time

            else:
                pose_predicted_text = "Pose Untracked"

            for landmark in pose_results.pose_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # 얼굴 예측 (2번 코드 로직으로 대체)
        faces = detect_faces(image)  # 얼굴 감지

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_image = image[y - 25:y + h + 25, x - 25:x + w + 25]  # 얼굴 영역 추출

                try:
                    expression = predict_expression(face_image, face_label_mapping)  # 표정 예측
                    face_predicted_text = expression
                    current_time = time.time()
                    if current_time - last_face_update_time >= face_update_interval:
                      if expression == 'happy':
                          expression_counts['happy'] += 1
                          final_score += face_weights['happy']
                      if expression == 'sad':
                          expression_counts['sad'] += 1
                          final_score += face_weights['sad']
                      if expression == 'neutral':
                          expression_counts['neutral'] += 1
                          final_score += face_weights['neutral']
                      if expression == 'disgust':
                          expression_counts['disgust'] += 1
                          final_score += face_weights['disgust']
                      if expression == 'surprise':
                          expression_counts['surprise'] += 1
                          final_score += face_weights['surprise']
                      if expression == 'angry':
                          expression_counts['angry'] += 1
                          final_score += face_weights['angry']
                      
                      last_face_update_time = current_time
                    cv2.rectangle(frame, (x-25, y-25), (x+w+25, y+h+25), (0, 0, 255), 2)  # 얼굴 사각형 표시

                except Exception as e:
                    print(f"얼굴 예측 오류: {e}")
                    face_predicted_text = "Face Untracked"
        else:
            face_predicted_text = "No Face Detected"

        # 예측 결과 화면에 표시
        cv2.putText(frame, pose_predicted_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, face_predicted_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # JPEG 형식으로 프레임을 인코딩하여
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/update_counts')
def update_counts():
    return jsonify({'pose_counts': pose_counts, 'expression_counts': expression_counts, 'final_score': final_score})

@app.route('/get_score')
def get_score():
    return jsonify({'final_score': final_score})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emoting')
def emoting():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555)
