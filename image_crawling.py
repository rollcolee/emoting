from selenium import webdriver
from time import sleep
import random
from selenium.webdriver.common.by import By
import urllib.request
import os

query = input("검색어 : ")

save_dir = query
os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)

driver = webdriver.Chrome()

url = f'https://www.google.com/search?tbm=isch&q={query}'
driver.get(url)

drag = driver.find_element(By.XPATH, 'html')

for i in range(10):
    driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', drag )
    sleep(random.randint(1, 2)) # 2~3초 쉬었다가 반복

# 웹 페이지에서 이미지 요소를 찾음
img_elements = driver.find_elements(By.CSS_SELECTOR, 'g-img.mNsIhb > img')
len(img_elements)

# 이미지 요소들의 'src' 속성을 추출하여 리스트에 저장
img_links = [elem.get_attribute('src') for elem in img_elements]

# 이미지 URL을 가져와 이미지 다운로드
for i, link in enumerate(img_links):
    try:
        # 이미지 링크(link)를 사용하여 이미지를 다운로드하고, 지정된 경로에 저장
        urllib.request.urlretrieve(link, f'{query}/img_{i}.jpg')
    except Exception as e:
         # 예외가 발생한 경우 해당 이미지의 인덱스와 에러 메시지를 출력
        print(f'{i}번째 사진 에러: {e}')
