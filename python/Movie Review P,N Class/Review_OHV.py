import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# IMDB 영화 리뷰 데이터셋
imdb = keras.datasets.imdb

# 빈도 상위 10,000개 단어 사용
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000
)

print("훈련 샘플: {}, 레이블: {}".format(
    len(train_data), len(train_labels)
))

# 레이블을 One-Hot Vector로 변환
train_labels = keras.utils.to_categorical(train_labels, num_classes=2)
test_labels = keras.utils.to_categorical(test_labels, num_classes=2)

print("One-Hot Vector 예시:")
print(train_labels[0])

# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()

# 특수 인덱스 추가
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# 숫자 인덱스를 다시 단어로 변환
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)


def decode_review(text):
    return ' '.join(
        [reverse_word_index.get(i, '?') for i in text]
    )


# 첫 번째 영화 리뷰 출력
print(decode_review(train_data[0]))

# 모든 리뷰 길이를 256으로 맞춤
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

print(len(train_data[0]), len(train_data[1]))

# 사용할 단어 수
vocab_size = 10000

# 모델 생성
model = keras.Sequential()

model.add(
    keras.layers.Embedding(
        vocab_size,
        16,
        input_shape=(None,)
    )
)

model.add(
    keras.layers.GlobalAveragePooling1D()
)

model.add(
    keras.layers.Dense(
        16,
        activation='relu'
    )
)

# 부정 / 긍정 2개 출력
model.add(
    keras.layers.Dense(
        2,
        activation='softmax'
    )
)

model.summary()

# One-Hot Vector이므로 categorical_crossentropy 사용
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 검증 데이터 분리
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 모델 학습
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)

# 테스트 데이터 평가
results = model.evaluate(
    test_data,
    test_labels,
    verbose=2
)

print("테스트 결과:", results)

# 첫 번째 영화 리뷰 예측
prediction = model.predict(test_data[:1])

print("예측 확률:", prediction[0])

# 가장 높은 확률을 One-Hot Vector로 변환
predicted_class = np.argmax(prediction[0])

one_hot_result = keras.utils.to_categorical(
    predicted_class,
    num_classes=2
)

print("One-Hot Vector 결과:", one_hot_result)

# 실제 정답
print("실제 정답:", test_labels[0])

# 학습 결과 저장
history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# Loss 그래프
plt.plot(
    epochs,
    loss,
    'bo',
    label='Training loss'
)

plt.plot(
    epochs,
    val_loss,
    'b',
    label='Validation loss'
)

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 그래프 초기화
plt.clf()

# Accuracy 그래프
plt.plot(
    epochs,
    acc,
    'bo',
    label='Training acc'
)

plt.plot(
    epochs,
    val_acc,
    'b',
    label='Validation acc'
)

plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()