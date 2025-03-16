표정에 대한 좌표학습 모델 저장, 라벨 인코딩 정보 저장

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import json
import numpy as np

# 1. 학습 준비 : 데이터 불러오기
df_face=pd.read_csv('/content/face_25_4000.csv')
df_face

# 'your_data.csv'를 실제 파일 경로로 변경하세요.
train_df = df_face[df_face['dataset'] == 'train']
test_df = df_face[df_face['dataset'] == 'test']

# 라벨 분리
X_train = train_df.drop(columns=["image", "label","dataset"], axis=1)
y_train = train_df["label"]
X_test = test_df.drop(columns=["image", "label","dataset"], axis=1)
y_test = test_df["label"]

print(X_train, y_train)
print(X_test, y_test)

# 라벨 인코딩
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# ✅ 변환된 y 값 확인
print("🔹 변환된 y_train:", np.unique(y_train_encoded))
print("🔹 변환된 y_test:", np.unique(y_test_encoded))

# 2. XGBoost 모델 학습
model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(label_encoder.classes_), random_state=42)
model.fit(X_train, y_train_encoded)

# 3. 예측 및 평가
y_pred_encoded = model.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"XG Test Accuracy: {accuracy:.4f}")

# 4. 모델 저장
model.save_model("/content/face_expression_model.json")

# 5. 라벨 인코딩 정보 저장
with open("/content/face_label_encoder.json", "w") as f:
    json.dump({i: label for i, label in enumerate(label_encoder.classes_)}, f)
