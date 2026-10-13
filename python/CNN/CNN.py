import tensorflow as tf
from tensorflow import keras

# MNIST 데이터셋을 로드합니다.
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# 데이터셋의 차원을 변경하여 CNN 모델에 맞게 조정합니다.
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

# CNN 모델을 구성합니다.
model = keras.Sequential()

# CNN 모델에 합성곱 층과 풀링 층을 추가합니다.
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

# 모델에 완전 연결 층을 추가합니다.
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# 모델을 컴파일합니다.
model.summary()

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)

