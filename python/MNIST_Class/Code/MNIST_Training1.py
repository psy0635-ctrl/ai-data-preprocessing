from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from MNIST_Testing1 import IMAGE_ROOT, load_images


tf.keras.utils.set_random_seed(42)

RESULT_DIR = (
    Path(__file__).resolve().parent / 'Result_image'
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================
# 1. 훈련·평가 데이터 준비
# =========================================

x_train_datas, y_train_datas = load_images(
    IMAGE_ROOT / 'training'
)

x_test_datas, y_test_datas = load_images(
    IMAGE_ROOT / 'testing'
)

print(
    '훈련 데이터:',
    x_train_datas.shape,
    y_train_datas.shape
)

print(
    '평가 데이터:',
    x_test_datas.shape,
    y_test_datas.shape
)


# =========================================
# 2. 기존 MNIST 모델 구성
# 입력 784개 → 은닉층 512개 → 출력 10개
# =========================================

model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),

    tf.keras.layers.Dense(
        512,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )
])

# 정답이 One-hot 배열이므로 categorical 손실 사용
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =========================================
# 3. 모델 학습
# Epoch마다 평가 데이터의 Loss도 기록
# =========================================

history = model.fit(
    x_train_datas,
    y_train_datas,
    epochs=5,
    batch_size=32,
    shuffle=True,
    validation_data=(
        x_test_datas,
        y_test_datas
    )
)


# =========================================
# 4. 최종 평가
# =========================================

test_loss, test_acc = model.evaluate(
    x_test_datas,
    y_test_datas,
    verbose=2
)

print(f'평가 Loss: {test_loss:.4f}')
print(f'평가 정확도: {test_acc:.2%}')


# =========================================
# 5. 훈련 Loss와 평가 Loss 비교
# =========================================

epochs = range(
    1,
    len(history.history['loss']) + 1
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    epochs,
    history.history['loss'],
    'bo-',
    label='Training loss'
)

ax.plot(
    epochs,
    history.history['val_loss'],
    'ro-',
    label='Evaluation loss'
)

ax.set(
    title='MNIST loss by epoch',
    xlabel='Epoch',
    ylabel='Loss'
)

ax.set_xticks(list(epochs))
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    RESULT_DIR / 'mnist_loss.png',
    dpi=150
)


# =========================================
# 6. 오분류 이미지 찾기
# =========================================

predictions = model.predict(
    x_test_datas,
    verbose=0
)

# 가장 높은 확률의 숫자를 예측 결과로 선택
predicted_labels = np.argmax(
    predictions,
    axis=1
)

# One-hot 정답을 숫자로 변환
true_labels = np.argmax(
    y_test_datas,
    axis=1
)

# 예측과 정답이 다른 이미지의 위치
wrong_indices = np.flatnonzero(
    predicted_labels != true_labels
)

print(
    f'오분류: {len(wrong_indices)}'
    f' / {len(true_labels)}개'
)


# =========================================
# 7. 오분류 이미지 최대 15개 출력
# =========================================

if len(wrong_indices) > 0:
    selected = wrong_indices[:15]

    # 한 줄에 5개씩 표시
    rows = (len(selected) + 4) // 5

    fig, axes = plt.subplots(
        rows,
        5,
        figsize=(15, 3 * rows),
        squeeze=False
    )

    # 이미지가 없는 칸까지 축 표시 제거
    for ax in axes.flat:
        ax.axis('off')

    for ax, index in zip(axes.flat, selected):
        # -1~1을 화면 표시용 0~1로 복원
        image = (
            x_test_datas[index].reshape(28, 28) + 1.0
        ) / 2.0

        ax.imshow(
            image,
            cmap='gray',
            vmin=0,
            vmax=1
        )

        ax.set_title(
            f'True: {true_labels[index]}'
            f' / Pred: {predicted_labels[index]}\n'
            f'Probability: {predictions[index].max():.1%}',
            color='red',
            fontsize=10
        )

    fig.suptitle('MNIST misclassified images')

    fig.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    fig.savefig(
        RESULT_DIR / 'mnist_wrong_images.png',
        dpi=150
    )

else:
    print('오분류 이미지가 없습니다.')


print('결과 저장 위치:', RESULT_DIR)

# 생성한 그래프와 오분류 이미지 화면 출력
plt.show()