from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


tf.keras.utils.set_random_seed(42)

RESULT_DIR = (
    Path(__file__).resolve().parent / 'Result_image'
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================
# 1. Fashion MNIST 불러오기
# =========================================

(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

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
# 2. 정규화 및 흑백 채널 추가
# (개수, 28, 28) → (개수, 28, 28, 1)
# =========================================

train_images = (
    train_images.astype(np.float32) / 255.0
)[..., np.newaxis]

test_images = (
    test_images.astype(np.float32) / 255.0
)[..., np.newaxis]

print('훈련 데이터:', train_images.shape)
print('평가 데이터:', test_images.shape)


# =========================================
# 3. CNN 모델 구성
# =========================================

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        64,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )
])


# =========================================
# 4. 모델 설정
# =========================================

# 정답이 정수 0~9이므로 sparse 손실 함수 사용
# 출력이 softmax 확률이므로 from_logits=True 사용 안 함
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =========================================
# 5. 모델 학습
# 훈련 데이터의 10%를 검증용으로 분리
# =========================================

history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    shuffle=True
)


# =========================================
# 6. 최종 평가
# 테스트 데이터는 모델 학습에 사용하지 않음
# =========================================

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels,
    verbose=2
)

print(f'평가 Loss: {test_loss:.4f}')
print(f'평가 정확도: {test_acc:.2%}')


# =========================================
# 7. Loss와 정확도 그래프
# =========================================

epochs = range(
    1,
    len(history.history['loss']) + 1
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

for ax, metric, title in zip(
    axes,
    ['loss', 'accuracy'],
    ['Loss', 'Accuracy']
):
    ax.plot(
        epochs,
        history.history[metric],
        label='Training'
    )

    ax.plot(
        epochs,
        history.history['val_' + metric],
        label='Validation'
    )

    ax.set(
        title=title,
        xlabel='Epoch',
        ylabel=title
    )

    ax.set_xticks(list(epochs))
    ax.legend()
    ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    RESULT_DIR / 'fashion_cnn_history.png',
    dpi=150
)


# =========================================
# 8. 테스트 이미지 예측
# =========================================

# 출력층이 이미 softmax이므로 추가 Softmax 불필요
predictions = model.predict(
    test_images,
    verbose=0
)


# =========================================
# 9. 이미지 15개와 클래스별 예측 확률 출력
# =========================================

fig, axes = plt.subplots(
    5,
    6,
    figsize=(18, 15)
)

for i in range(15):
    row, column = divmod(i, 3)

    image_ax = axes[row, column * 2]
    bar_ax = axes[row, column * 2 + 1]

    predicted = int(np.argmax(predictions[i]))
    actual = int(test_labels[i])

    # 정답이면 파란색, 오답이면 빨간색
    color = 'blue' if predicted == actual else 'red'

    # 흑백 채널을 제외하고 28×28 이미지 표시
    image_ax.imshow(
        test_images[i, :, :, 0],
        cmap='binary',
        vmin=0,
        vmax=1
    )

    image_ax.set_xticks([])
    image_ax.set_yticks([])

    image_ax.set_title(
        f'Pred: {class_names[predicted]}\n'
        f'{predictions[i, predicted]:.1%}'
        f' / True: {class_names[actual]}',
        color=color,
        fontsize=9
    )

    # 클래스 번호 0~9의 예측 확률
    bars = bar_ax.bar(
        range(10),
        predictions[i],
        color='#777777'
    )

    # 예측 클래스는 빨간색, 실제 정답은 파란색
    # 정답을 맞힌 경우 해당 막대는 파란색
    bars[predicted].set_color('red')
    bars[actual].set_color('blue')

    bar_ax.set_xticks(range(10))
    bar_ax.set_ylim(0, 1)

    bar_ax.set_title(
        'Class probabilities',
        fontsize=9
    )

fig.tight_layout()

fig.savefig(
    RESULT_DIR / 'fashion_cnn_predictions.png',
    dpi=150
)

print('결과 저장 위치:', RESULT_DIR)

plt.show()