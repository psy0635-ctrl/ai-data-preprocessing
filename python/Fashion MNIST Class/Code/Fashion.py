import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# =========================================
# Fashion MNIST 데이터 불러오기
# =========================================

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = (
    fashion_mnist.load_data()
)


# =========================================
# 클래스 이름
# =========================================

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


# =========================================
# 데이터 정규화
# 0 ~ 255 → 0 ~ 1
# =========================================

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0


print("학습 데이터 :", train_images.shape)
print("테스트 데이터 :", test_images.shape)

print(
    "정규화 범위 :",
    train_images.min(),
    "~",
    train_images.max()
)


# =========================================
# 모델 생성
# =========================================

model = tf.keras.Sequential([

    tf.keras.Input(
        shape=(28, 28)
    ),

    # 28 × 28 이미지를 784개의 값으로 변환
    tf.keras.layers.Flatten(),

    # 은닉층
    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    # 출력층
    tf.keras.layers.Dense(
        10
    )
])


# =========================================
# 모델 설정
# =========================================

model.compile(

    optimizer='adam',

    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),

    metrics=[
        'accuracy'
    ]
)


# =========================================
# 모델 구조 확인
# =========================================

model.summary()


# =========================================
# 모델 학습
# =========================================

model.fit(
    train_images,
    train_labels,
    epochs=10
)


# =========================================
# 모델 평가
# =========================================

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels,
    verbose=2
)


print()
print("Test accuracy :", test_acc)


# =========================================
# 예측값을 확률로 변환
# =========================================

probability_model = tf.keras.Sequential([

    model,

    tf.keras.layers.Softmax()

])


# =========================================
# 테스트 이미지 예측
# =========================================

predictions = probability_model.predict(
    test_images
)


# =========================================
# 이미지 출력 함수
# =========================================

def plot_image(
    i,
    predictions_array,
    true_label,
    img
):

    true_label = true_label[i]
    img = img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # 흑백 이미지
    plt.imshow(
        img,
        cmap=plt.cm.binary
    )

    # 가장 높은 확률의 클래스 번호
    predicted_label = np.argmax(
        predictions_array
    )

    # 정답이면 파란색
    if predicted_label == true_label:

        color = 'blue'

    # 틀렸으면 빨간색
    else:

        color = 'red'


    # 이미지 밑에
    # 예측 이름 / 확률 / 실제 정답 출력
    plt.xlabel(

        "{} {:2.0f}% ({})".format(

            class_names[predicted_label],

            100 * np.max(
                predictions_array
            ),

            class_names[true_label]
        ),

        color=color
    )


# =========================================
# 예측 확률 막대그래프 함수
# =========================================

def plot_value_array(
    i,
    predictions_array,
    true_label
):

    true_label = true_label[i]

    plt.grid(False)

    plt.xticks(
        range(10)
    )

    plt.yticks([])


    # 0~9 클래스의 예측 확률
    thisplot = plt.bar(

        range(10),

        predictions_array,

        color="#777777"
    )


    # 확률 범위 0 ~ 1
    plt.ylim(
        [0, 1]
    )


    # 가장 확률이 높은 클래스
    predicted_label = np.argmax(
        predictions_array
    )


    # 예측한 클래스 → 빨간색
    thisplot[
        predicted_label
    ].set_color(
        'red'
    )


    # 실제 정답 → 파란색
    thisplot[
        true_label
    ].set_color(
        'blue'
    )


# =========================================
# 15개 이미지 + 예측 결과 출력
# =========================================

num_rows = 5
num_cols = 3

num_images = (
    num_rows
    * num_cols
)


# 그래프 전체 크기
plt.figure(
    figsize=(
        2 * 2 * num_cols,
        2 * num_rows
    )
)


# 15개 이미지 반복
for i in range(num_images):

    # =====================================
    # 왼쪽 : 이미지
    # =====================================

    plt.subplot(
        num_rows,
        2 * num_cols,
        2 * i + 1
    )

    plot_image(
        i,
        predictions[i],
        test_labels,
        test_images
    )


    # =====================================
    # 오른쪽 : 예측 확률 그래프
    # =====================================

    plt.subplot(
        num_rows,
        2 * num_cols,
        2 * i + 2
    )

    plot_value_array(
        i,
        predictions[i],
        test_labels
    )


# 그래프 간격 자동 정리
plt.tight_layout()


# =========================================
# 화면 출력
# =========================================

plt.show()