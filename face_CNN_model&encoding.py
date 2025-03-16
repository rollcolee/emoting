import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision
import os
import shutil
import random
import json

# Mixed Precision 적용 (연산 속도 향상)
mixed_precision.set_global_policy('mixed_float16')

# 사용자 정의: 원본 데이터 경로 (구글 드라이브 혹은 로컬 경로)
base_dir = "/content/drive/MyDrive/extracted_images/3000" # 실제 사용자 폴더 경로에 맞게 변경
train_dir = os.path.join(base_dir, "train") #실제 폴더 경로에 맞게 변경
test_dir = os.path.join(base_dir, "test") #실제 폴더 경로에 맞게 변경

batch_size = 128  # 배치 크기 증가
image_size = (96, 96)  # 이미지 크기 유지

# 이미지 데이터 전처리 (train 데이터를 train/validation으로 분리)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20%를 검증 데이터로 사용
)

test_datagen = ImageDataGenerator(rescale=1./255)

# train, test, validation 데이터 학습 준비
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# CNN 모델 생성
def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax', dtype='float32')  # Mixed Precision 적용
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 모델 생성 및 학습
num_classes = len(train_generator.class_indices)
print(num_classes)
cnn_model = create_cnn_model((96, 96, 3), num_classes)

# Early Stopping 적용
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = cnn_model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator  # use_multiprocessing 제거
)

# 검증 정확도 확인
val_accuracy = history.history['val_accuracy']
print(val_accuracy)

# 모델 평가 (test 데이터는 학습 후 최종 평가만!)
loss, accuracy = cnn_model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy:.4f}")

# 모델 저장
cnn_model.save("/content/norm_CNN_face_expression_model.keras")  # 확장자를 .keras로 변경
print("모델 저장 완료!")

# 라벨 인코딩 정보 저장
label_map = train_generator.class_indices
with open("/content/norm_CNN_face_label_encoder.json", "w") as f: # json 저장 경로 지정
  json.dump(label_map, f)
print("라벨 인코딩 정보 저장 완료!")
