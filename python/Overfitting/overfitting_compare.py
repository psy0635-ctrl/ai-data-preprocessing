# ============================================================
# Chapter 6 - 과적합 / 과소적합
# IMDB 데이터를 이용한
# 모델 크기 / L2 규제 / Dropout 비교
# ============================================================


# ============================================================
# 1. TensorFlow 환경 설정
# ============================================================

import os

# TensorFlow 로그 줄이기
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# oneDNN 연산 차이 최소화
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# ============================================================
# 2. 라이브러리 불러오기
# ============================================================

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 3. 난수 고정
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ============================================================
# 4. IMDB 데이터 준비
# ============================================================

NUM_WORDS = 1000
EPOCHS = 100
BATCH_SIZE = 512

print("IMDB 데이터를 불러오는 중...")

(train_data, train_labels), (test_data, test_labels) = \
    keras.datasets.imdb.load_data(
        num_words=NUM_WORDS
    )


# ============================================================
# 5. Multi-Hot Encoding 함수
# ============================================================

def multi_hot_sequences(sequences, dimension):

    # 리뷰 개수 × 단어 개수 크기의 0 배열 생성
    results = np.zeros(
        (len(sequences), dimension),
        dtype=np.float32
    )

    # 리뷰에 등장한 단어 번호의 위치를 1로 변경
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0

    return results


print("데이터를 Multi-Hot 형식으로 변환하는 중...")


train_data = multi_hot_sequences(
    train_data,
    NUM_WORDS
)

test_data = multi_hot_sequences(
    test_data,
    NUM_WORDS
)


# ============================================================
# 6. Label 모양 수정
# ============================================================

# 모델 출력:
# (None, 1)
#
# Label도 같은 차원으로 변경:
# (25000,) → (25000, 1)

train_labels = np.asarray(
    train_labels,
    dtype=np.float32
).reshape(-1, 1)


test_labels = np.asarray(
    test_labels,
    dtype=np.float32
).reshape(-1, 1)


# 데이터 크기 확인
print()
print("========== 데이터 크기 ==========")

print("훈련 데이터 :", train_data.shape)
print("훈련 정답   :", train_labels.shape)

print("테스트 데이터 :", test_data.shape)
print("테스트 정답   :", test_labels.shape)


# ============================================================
# 7. Baseline Model
#
# 1000 → 16 → 16 → 1
# ============================================================

baseline_model = keras.Sequential([

    keras.Input(
        shape=(NUM_WORDS,)
    ),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


baseline_model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        "binary_crossentropy"
    ]
)


print()
print("===================================")
print("Baseline Model")
print("===================================")

baseline_model.summary()


# ============================================================
# 8. Smaller Model
#
# 1000 → 4 → 4 → 1
# ============================================================

smaller_model = keras.Sequential([

    keras.Input(
        shape=(NUM_WORDS,)
    ),

    keras.layers.Dense(
        4,
        activation="relu"
    ),

    keras.layers.Dense(
        4,
        activation="relu"
    ),

    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


smaller_model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        "binary_crossentropy"
    ]
)


print()
print("===================================")
print("Smaller Model")
print("===================================")

smaller_model.summary()


# ============================================================
# 9. Bigger Model
#
# 1000 → 512 → 512 → 1
# ============================================================

bigger_model = keras.Sequential([

    keras.Input(
        shape=(NUM_WORDS,)
    ),

    keras.layers.Dense(
        512,
        activation="relu"
    ),

    keras.layers.Dense(
        512,
        activation="relu"
    ),

    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


bigger_model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        "binary_crossentropy"
    ]
)


print()
print("===================================")
print("Bigger Model")
print("===================================")

bigger_model.summary()


# ============================================================
# 10. L2 규제 적용 모델
#
# 1000 → 16(L2) → 16(L2) → 1
# L2 = 0.001
# ============================================================

l2_model = keras.Sequential([

    keras.Input(
        shape=(NUM_WORDS,)
    ),

    keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.001)
    ),

    keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.001)
    ),

    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


l2_model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        "binary_crossentropy"
    ]
)


print()
print("===================================")
print("L2 Model")
print("===================================")

l2_model.summary()


# ============================================================
# 11. Dropout 적용 모델
#
# 1000
#   ↓
# Dense 16
#   ↓
# Dropout 0.5
#   ↓
# Dense 16
#   ↓
# Dropout 0.5
#   ↓
# Dense 1
# ============================================================

dpt_model = keras.Sequential([

    keras.Input(
        shape=(NUM_WORDS,)
    ),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dropout(
        0.5
    ),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dropout(
        0.5
    ),

    keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


dpt_model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        "binary_crossentropy"
    ]
)


print()
print("===================================")
print("Dropout Model")
print("===================================")

dpt_model.summary()


# ============================================================
# 12. Baseline Model 학습
# ============================================================

print()
print("===================================")
print("Baseline Model 학습 시작")
print("===================================")


baseline_history = baseline_model.fit(

    train_data,
    train_labels,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_data=(
        test_data,
        test_labels
    ),

    verbose=2
)


# ============================================================
# 13. Smaller Model 학습
# ============================================================

print()
print("===================================")
print("Smaller Model 학습 시작")
print("===================================")


smaller_history = smaller_model.fit(

    train_data,
    train_labels,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_data=(
        test_data,
        test_labels
    ),

    verbose=2
)


# ============================================================
# 14. Bigger Model 학습
# ============================================================

print()
print("===================================")
print("Bigger Model 학습 시작")
print("===================================")


bigger_history = bigger_model.fit(

    train_data,
    train_labels,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_data=(
        test_data,
        test_labels
    ),

    verbose=2
)


# ============================================================
# 15. L2 Model 학습
# ============================================================

print()
print("===================================")
print("L2 Model 학습 시작")
print("===================================")


l2_model_history = l2_model.fit(

    train_data,
    train_labels,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_data=(
        test_data,
        test_labels
    ),

    verbose=2
)


# ============================================================
# 16. Dropout Model 학습
# ============================================================

print()
print("===================================")
print("Dropout Model 학습 시작")
print("===================================")


dpt_model_history = dpt_model.fit(

    train_data,
    train_labels,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_data=(
        test_data,
        test_labels
    ),

    verbose=2
)


# ============================================================
# 17. 모델 학습 결과 그래프 함수
# ============================================================

def plot_history(
    histories,
    key="binary_crossentropy"
):

    plt.figure(
        figsize=(16, 10)
    )


    for name, history in histories:

        # ====================================================
        # Validation
        # 점선
        # ====================================================

        val = plt.plot(

            history.epoch,

            history.history[
                "val_" + key
            ],

            "--",

            label=name.title() + " Val"
        )


        # ====================================================
        # Train
        # 실선
        # ====================================================

        plt.plot(

            history.epoch,

            history.history[key],

            color=val[0].get_color(),

            label=name.title() + " Train"
        )


    # X축 이름
    plt.xlabel(
        "Epochs"
    )


    # Y축 이름
    plt.ylabel(
        key.replace(
            "_",
            " "
        ).title()
    )


    # 범례
    plt.legend()


    # 가장 긴 Epoch 기준
    max_epoch = max(
        max(history.epoch)
        for name, history in histories
    )


    plt.xlim([
        0,
        max_epoch
    ])


    # 격자 제거
    plt.grid(False)


    # 그래프 정리
    plt.tight_layout()


    # 그래프 출력
    plt.show()


# ============================================================
# 18. 다섯 모델 최종 비교
# ============================================================

plot_history([

    (
        "baseline",
        baseline_history
    ),

    (
        "smaller",
        smaller_history
    ),

    (
        "bigger",
        bigger_history
    ),

    (
        "l2",
        l2_model_history
    ),

    (
        "dropout",
        dpt_model_history
    )

])