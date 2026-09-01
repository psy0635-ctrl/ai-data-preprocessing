import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Model
from keras.layers import Dense
import matplotlib.pyplot as plt
 
def test_system(x):
    return 0.4*x+8.0

def get_model():
    tf.random.set_seed(1000)
    input = Input(shape=(1,), name="Input")
    output = Dense(1, activation='linear', name="Output")(input)
    model = Model(inputs=[input], outputs=[output])
    opt = keras.optimizers.Adam(learning_rate=0.0025)
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
    model.summary()
    return model
 

if __name__ == '__main__':
    x_datas = np.array(range(-50, 51, 10))
    y_datas = []
    for x in x_datas:
        y_datas.append(test_system(x))
    y_datas = np.array(y_datas)

    # plt.scatter(x_datas, y_datas)
    # plt.show()

    model = get_model()
    history = model.fit(x_datas, y_datas, epochs=1000, shuffle=True)
    plt.plot(history.history['loss'], 'b', label='loss')
    plt.show()
 
    x_test = np.array(range(-45, 45, 10))
    result = model.predict(x_test)
 
    plt.scatter(x_datas, y_datas, c='blue', s=5)
    plt.scatter(x_test, result, c='red', s=5)
    plt.show()
 
    weights = model.get_weights()
    print("W : ", weights[0], "   b : ", weights[1])
