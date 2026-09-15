import os
from pathlib import Path

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


# =========================================
# Testing 데이터 가져오기
# =========================================

from MNIST_Testing import x_test_datas, y_test_datas


# =========================================
# 경로 설정
# =========================================

IMAGE_ROOT = Path("/workspace/MNIST/images")
TRAIN_ROOT = IMAGE_ROOT / "training"

# 현재 파일:
# /workspace/python/MNIST_Class/Code/MNIST_Training.py

CODE_DIR = Path(__file__).resolve().parent

# Code의 상위 폴더 = MNIST_Class
PROJECT_DIR = CODE_DIR.parent

# 결과 이미지 저장 폴더
RESULT_DIR = PROJECT_DIR / "Result_image"

# 폴더가 없으면 자동 생성
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================
# Training 이미지 목록
# =========================================

all_files = []

for i in range(0, 10):

    path_dir = TRAIN_ROOT / str(i)

    file_list = os.listdir(path_dir)
    file_list.sort()

    all_files.append(file_list)

    print(
        "학습 숫자 {0} : {1}개".format(
            i,
            len(file_list)
        )
    )


# =========================================
# 학습 데이터 생성
# =========================================

x_train_datas = []
y_train_datas = []


for num in range(0, 10):

    for numbers in all_files[num]:

        img_path = TRAIN_ROOT / str(num) / numbers

        # 이미지 읽기
        img = Image.open(img_path)

        # 이미지 → NumPy 배열
        # 0~255 → 0~1
        imgarr = np.array(
            img,
            dtype=np.float32
        )

        # 28 × 28 → 784 × 1
        x_train_datas.append(
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

        y_train_datas.append(y_tmp)

    print(
        "학습 숫자 {0} 처리 완료".format(num)
    )


# =========================================
# NumPy 배열로 변환
# =========================================

x_train_datas = np.array(
    x_train_datas,
    dtype=np.float32
)

y_train_datas = np.array(
    y_train_datas,
    dtype=np.float32
)


# =========================================
# 모델 입력 형태
# =========================================

x_train_datas = np.reshape(
    x_train_datas,
    (-1, 784)
)

y_train_datas = np.reshape(
    y_train_datas,
    (-1, 10)
)


print()
print("========== 데이터 형태 ==========")

print(
    "학습 데이터 개수 :",
    len(x_train_datas)
)

print(
    "x_train_datas 형태 :",
    x_train_datas.shape
)

print(
    "y_train_datas 형태 :",
    y_train_datas.shape
)

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

print("==================================")


# =========================================
# 딥러닝 모델 생성
# =========================================

input = tf.keras.Input(
    shape=(784,),
    name="Input"
)


# 은닉층
hidden = layers.Dense(
    512,
    activation="relu",
    name="Hidden1"
)(input)


# 출력층
output = layers.Dense(
    10,
    activation="softmax",
    name="Output"
)(hidden)


# 모델
model = tf.keras.Model(
    inputs=[input],
    outputs=[output]
)


# =========================================
# Optimizer
# =========================================

opt = keras.optimizers.Adam(
    learning_rate=0.001
)


# =========================================
# Compile
# =========================================

model.compile(

    loss="categorical_crossentropy",

    optimizer=opt,

    metrics=["accuracy"]
)


# 모델 구조
model.summary()


# =========================================
# 모델 학습
# =========================================

history = model.fit(

    x_train_datas,

    y_train_datas,

    epochs=5,

    shuffle=True,

    validation_data=(
        x_test_datas,
        y_test_datas
    )
)


# =========================================
# 모델 평가
# =========================================

test_loss, test_acc = model.evaluate(

    x_test_datas,

    y_test_datas
)


print()
print("========== 최종 테스트 ==========")

print(
    "테스트 손실 :",
    test_loss
)

print(
    "테스트 정확도 :",
    test_acc
)

print("=================================")


# =========================================
# 학습 결과 그래프
# =========================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    "b",
    label="loss"
)

plt.plot(
    history.history["val_accuracy"],
    "r",
    label="val_accuracy"
)

plt.xlabel("Epoch")

plt.legend()

plt.tight_layout()


# =========================================
# 결과 이미지 자동 저장
# =========================================

graph_path = (
    RESULT_DIR
    / "MNIST_60000개_학습결과_그래프.png"
)

plt.savefig(
    graph_path,
    dpi=150
)


print()
print(
    "그래프 저장 위치 :",
    graph_path
)


# 그래프 화면 출력
plt.show()