
import keras
import numpy as np

def build_detector(model):
    layers = [layer for layer in model.layers]
    num_layers = len(layers)
    output = keras.layers.Dense(2)(layers[num_layers-2].output)
    detector = keras.Model(inputs=layers[0].input, outputs=output)
    # detector.summary()
    return detector

def train(model, x_train, x_train_adv):
    nb_train = x_train.shape[0]
    x_train_detector = np.concatenate((x_train, x_train_adv), axis=0)
    y_train_detector = np.concatenate((np.array([[1,0]]*nb_train), np.array([[0,1]]*nb_train)), axis=0)

    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(x_train_detector, y_train_detector, epochs=20, batch_size=200)
    return 0

def result(model, x_test, x_test_adv):
    flag_adv = np.sum(np.argmax(model.predict(x_test_adv), axis=1) == 1)

    print("Adversarial test data (first 100 images):")
    print("Flagged: {}".format(flag_adv))
    print("Not flagged: {}".format(100 - flag_adv))

    flag_original = np.sum(np.argmax(model.predict(x_test[:100]), axis=1) == 1)

    print("Original test data (first 100 images):")
    print("Flagged: {}".format(flag_original))
    print("Not flagged: {}".format(100 - flag_original))
    return 0
