import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model

from art.config import ART_DATA_PATH
from art.utils import load_dataset, get_file
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod
from fast_gradient_sign_method import FastGradientSignMethod
from art.attacks.evasion import BasicIterativeMethod
# from art.defences.trainer import AdversarialTrainer

import numpy as np

# %matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print(tf.__version__)

# (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')
# path = get_file('mnist_cnn_original.h5', extract=False, path=ART_DATA_PATH,
#                 url='https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1')
# classifier_model = load_model(path)


# (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')
# path = get_file('mnist_cnn_original.h5', extract=False, path=ART_DATA_PATH,
#                 url='https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1')
# classifier_model = load_model(path)

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')
path = get_file('cifar_resnet.h5',extract=False, path=ART_DATA_PATH,
                url='https://www.dropbox.com/s/ta75pl4krya5djj/cifar_resnet.h5?dl=1')
classifier_model = load_model(path)


classifier = KerasClassifier(clip_values=(min_, max_), model=classifier_model, use_logits=False)

num = 100
classifier_model.summary()
x_test_pred = np.argmax(classifier.predict(x_test[:num]), axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test[:num], axis=1))

print(f"Original test data (first {num} images):")
print("Correctly classified: {}".format(nb_correct_pred))
print("Incorrectly classified: {}".format(num-nb_correct_pred))

# loss_object = tf.keras.losses.CategoricalCrossentropy()
# def create_adversarial_pattern(input_image, input_label):
#     with tf.GradientTape() as tape:
#         tape.watch(input_image)
#         prediction = pretrained_model(input_image)
#         loss = loss_object(input_label, prediction)
#
#     # Get the gradients of the loss w.r.t to the input image.
#     gradient = tape.gradient(loss, input_image)
#     # Get the sign of the gradients to create the perturbation
#     signed_grad = tf.sign(gradient)
#     return signed_grad


# x_0_arr = np.array([x_test[0] * num])
# attacker = FastGradientMethod(classifier, eps=0.1)
attacker = FastGradientSignMethod(classifier, eps=0.5, batch_size = 8)
# x_test_adv = attacker.generate(x_test[:num]) # non-targeted
# x_test_adv = attacker.generate_targeted(x_test[:num], x_test[0]) #targeted
# x_test_adv = attacker.generate_iterative(x_test[:num]) #iterative non-targeted
x_test_adv = attacker.generate_targeted_iterative(x_test[:num], x_test[0]) #iterative targeted



x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test[:num], axis=1))


print(f"Adversarial test data (first {num} images):")
print("Correctly classified: {}".format(nb_correct_adv_pred))
print("Incorrectly classified: {}".format(num-nb_correct_adv_pred))
nb_preb_as_tar = np.count_nonzero(x_test_adv_pred == np.argmax(y_test[:num], axis=1)[0])
print(f"Classified as targeted label: {nb_preb_as_tar}\n\n")

print(x_test_adv_pred)
print(np.argmax(y_test[:num], axis=1))
# print(x_test_pred)
# print(x_test[0])

# first_image = x_test[0]
# first_image = np.array(first_image, dtype='float')
# first_image_pixels = first_image.reshape((28, 28))
#
# first_adv_image = x_test_adv[0]
# first_adv_image = np.array(first_adv_image, dtype='float')
# first_adv_image_pixels = first_adv_image.reshape((28, 28))
# plt.imshow(first_adv_image_pixels, cmap='gray')
#
# plt.show()