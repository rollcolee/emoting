# 다양한 모델 학습 및 평가

import random
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier 임포트
from sklearn.svm import SVC  # SVC 임포트
import lightgbm as lgb  # lightgbm 임포트
from catboost import CatBoostClassifier  # CatBoostClassifier 임포트
import xgboost as xgb  # XGBoost 임포트


# 모델 정의
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "XGBoost": xgb.XGBClassifier(random_state=42) # XGBoost 추가
}

# 모델 학습 및 평가
for model_name, model in models.items():
    model.fit(X_train, y_train_encoded)
    y_pred_encoded = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"{model_name} Test Accuracy: {accuracy:.4f}")
