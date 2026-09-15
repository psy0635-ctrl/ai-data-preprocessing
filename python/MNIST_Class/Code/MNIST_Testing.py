import os
from pathlib import Path

from PIL import Image
import numpy as np


# =========================================
# 경로 설정
# =========================================

IMAGE_ROOT = Path("/workspace/MNIST/images")
TEST_ROOT = IMAGE_ROOT / "testing"


# =========================================
# 테스트 파일 목록 가져오기
# =========================================

eval_files = []

for i in range(0, 10):

    path_dir = TEST_ROOT / str(i)

    file_list = os.listdir(path_dir)
    file_list.sort()

    eval_files.append(file_list)

    print(
        "테스트 숫자 {0} : {1}개".format(
            i,
            len(file_list)
        )
    )


# =========================================
# 테스트 데이터 생성
# =========================================

x_test_datas = []
y_test_datas = []


for num in range(0, 10):

    for numbers in eval_files[num]:

        img_path = TEST_ROOT / str(num) / numbers

        # 이미지 열기
        img = Image.open(img_path)

        # 이미지 → NumPy 배열
        # 0~255 → 0~1 정규화
        imgarr = np.array(
            img,
            dtype=np.float32
        ) / 255.0

        # 28 × 28 → 784 × 1
        x_test_datas.append(
            np.reshape(
                imgarr,
                (784, 1)
            )
        )

        # One-hot Vector
        y_tmp = np.zeros(
            shape=(10),
            dtype=np.float32
        )

        y_tmp[num] = 1

        y_test_datas.append(y_tmp)

    print(
        "테스트 숫자 {0} 처리 완료".format(num)
    )


# =========================================
# NumPy 배열로 변환
# =========================================

x_test_datas = np.array(
    x_test_datas,
    dtype=np.float32
)

y_test_datas = np.array(
    y_test_datas,
    dtype=np.float32
)


# =========================================
# 모델 입력 형태로 변환
# =========================================

x_test_datas = np.reshape(
    x_test_datas,
    (-1, 784)
)

y_test_datas = np.reshape(
    y_test_datas,
    (-1, 10)
)


# =========================================
# 확인
# =========================================

print()
print("========== 테스트 데이터 ==========")

print(
    "테스트 데이터 개수 :",
    len(x_test_datas)
)

print(
    "x_test_datas 형태 :",
    x_test_datas.shape
)

print(
    "y_test_datas 형태 :",
    y_test_datas.shape
)

print("====================================")