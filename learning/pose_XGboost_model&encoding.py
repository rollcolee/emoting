# 행동 : 좌표학습 모델 저장, 라벨 인코딩 정보 저장

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import json

# 파일 로드 (딕셔너리 형태로 수정)
file_paths = {
    "crossed_leg": "/content/다리꼬기_좌표값.csv",
    "hair_comb": "/content/머리넘기기_좌표값.csv",
    "touching_chin": "/content/touching_chin.csv",
    "folded_arm": "/content/folded_arm.csv",
    "smile_cover": "/content/입막고웃기_좌표값.csv",
    "drinking" : "/content/drinking_좌표값.csv",
    "eating" : "/content/밥먹기_좌표값.csv"
}

dataframes = []
for label, path in file_paths.items():
    df = pd.read_csv(path)
    df["label"] = label  # 각 데이터에 해당 자세 이름을 라벨로 추가
    dataframes.append(df)

# 데이터 통합
df = pd.concat(dataframes, ignore_index=True)

# 특성과 라벨 분리
X = df.drop(columns=["image_name", "label"], errors='ignore')  # 이미지명, 라벨 제외
y = df["label"]

print(X, y)

# 라벨 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 학습
model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(label_encoder.classes_), eval_metric="mlogloss")
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Test Accuracy: {accuracy:.4f}")

# 모델 저장
model.save_model("/content/pose_classification_model.json")

# 라벨 인코딩 정보 저장
with open("/content/label_encoder.json", "w") as f:
    json.dump({i: label for i, label in enumerate(label_encoder.classes_)}, f)
