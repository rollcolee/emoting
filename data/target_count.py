# 학습 전 데이터셋 개수 조정

import os
import shutil
import random
from PIL import Image, ImageEnhance  # 데이터 증강을 위한 라이브러리 (필요 시 설치: pip install Pillow)

def adjust_folder_image_count(folder_path, target_count, output_folder):
    """주어진 폴더 경로 내의 각 하위 폴더 이미지 수를 target_count로 맞춥니다."""
    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(subfolder_path):
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            current_count = len(image_files)

            if current_count < target_count:
                # 이미지 개수가 부족한 경우 이미지 복사 또는 증강
                diff = target_count - current_count
                if current_count > 0: # 이미지가 하나라도 있는경우에만 증강을 진행
                  for i in range(diff):
                      src_image_path = os.path.join(subfolder_path, random.choice(image_files))
                      dst_image_path = os.path.join(subfolder_path, f"aug_{i}_{os.path.basename(src_image_path)}")
                      augment_image(src_image_path, dst_image_path)  # 이미지 증강 함수 호출
                else: # 이미지가 없는경우는 오류메세지를 출력한다.
                   print (f"{subfolder_path}에 이미지가 존재하지 않습니다.")

            elif current_count > target_count:
                # 이미지 개수가 초과하는 경우 이미지 삭제
                diff = current_count - target_count
                files_to_remove = random.sample(image_files, diff)
                for file_name in files_to_remove:
                    os.remove(os.path.join(subfolder_path, file_name))

def augment_image(src_path, dst_path):
    """이미지를 증강합니다."""
    image = Image.open(src_path)

    # 간단한 증강 예시: 밝기 조절
    enhancer = ImageEnhance.Brightness(image)
    augmented_image = enhancer.enhance(random.uniform(0.8, 1.2))  # 밝기 범위: 0.8 ~ 1.2

    augmented_image.save(dst_path)

# 학습 데이터 폴더 경로 설정
train_folder = '/content/extracted_images/train'
test_folder = '/content/extracted_images/test'

# 출력 폴더 경로 설정
output_train_folder = '/content/adjusted_images/train'
output_test_folder = '/content/adjusted_images/test'

# 출력 폴더 생성
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

# 이미지 개수 조정 및 파일 저장
adjust_folder_image_count(train_folder, 3000, output_train_folder)
adjust_folder_image_count(test_folder, 600, output_test_folder)

print("학습 데이터 폴더 이미지 개수 조정 및 파일 저장 완료!")
