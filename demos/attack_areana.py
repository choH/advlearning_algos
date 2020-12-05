import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model

from art.config import ART_DATA_PATH
from art.utils import load_dataset, get_file
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import BasicIterativeMethod

import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import name_to_label, label_to_name
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image


from fast_gradient_sign_method import FastGradientSignMethod
from hop_skip_jump import HopSkipJump
from deepfool import DeepFool
from dynamic_deepfool import DynamicDeepFool


import numpy as np
import timeit
# %matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print(tf.__version__)




(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')
path = get_file('mnist_cnn_original.h5', extract=False, path=ART_DATA_PATH,
                url='https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1')
classifier_model = load_model(path)

# (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')
# path = get_file('cifar_resnet.h5',extract=False, path=ART_DATA_PATH,
#                 url='https://www.dropbox.com/s/ta75pl4krya5djj/cifar_resnet.h5?dl=1')
# classifier_model = load_model(path)


# # Discarded iris and stl10 dataset
# (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('stl10')
# (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('iris')
# # path = get_file('cifar_resnet.h5',extract=False, path=ART_DATA_PATH,
# #                 url='https://www.dropbox.com/s/ta75pl4krya5djj/cifar_resnet.h5?dl=1')
# # classifier_model = load_model(path)
# classifier_model = KerasClassifier()
# classifier_model.fit(x = x_train, y = y_train)

test_num = adv_num = len(x_test)
# test_num = adv_num = 500
# test_num = 10
# adv_num = 10


classifier = KerasClassifier(clip_values=(min_, max_), model=classifier_model, use_logits=False)

# classifier_model.summary()
x_test_pred = np.argmax(classifier.predict(x_test[:test_num]), axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test[:test_num], axis=1))
print(f"Original test data (first {test_num} images):")
print("Correctly classified: {}".format(nb_correct_pred))
print("Incorrectly classified: {}".format(test_num-nb_correct_pred))

start = timeit.default_timer()
# FGSM with extensions
attacker = FastGradientSignMethod(classifier, eps=5, batch_size = 32)
# x_test_adv = attacker.generate(x_test[:adv_num]) # non-targeted
# x_test_adv = attacker.generate_targeted(x_test[:adv_num], aimed_target = x_test[0]) #targeted
x_test_adv = attacker.generate_iterative(x_test[:adv_num], eps_step = 0.05) #iterative non-targeted
# x_test_adv = attacker.generate_targeted_iterative(x_test[:adv_num], eps_step = 0.05, aimed_target =x_test[0]) #iterative targeted

# # Hop Skip Jump: Paper uses max_iter=64, max_eval=10000, init_eval=100 but thats ultra-mega slow.
# attacker = HopSkipJump(classifier=classifier, targeted=False, norm=np.inf, max_iter=32, max_eval=100, init_eval=10)
# attacker = HopSkipJump(classifier=classifier, targeted=False, norm=np.inf, max_iter=64, max_eval=1000, init_eval=100)
# x_test_adv = attacker.generate(x_test[:adv_num])


# # DeepFool
# attacker = DeepFool(classifier)
# x_test_adv = attacker.generate(x_test[:adv_num])

# # DeepFool extension
# attacker = DynamicDeepFool(classifier)
# x_test_adv = attacker.generate(x_test[:adv_num])


stop = timeit.default_timer()

x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test[:adv_num], axis=1))

print('#'*30)
print(f"Adversarial test data (first {adv_num} images):")
print("Correctly classified: {}".format(nb_correct_adv_pred))
print("Incorrectly classified: {}".format(adv_num-nb_correct_adv_pred))
nb_preb_as_tar = np.count_nonzero(x_test_adv_pred == np.argmax(y_test[:adv_num], axis=1)[0])
print(f"Classified as targeted label: {nb_preb_as_tar}")
print('#'*30)

# first_adv = x_test_adv[0]
# print(first_adv)


def pertub_budget(org_list, adv_list, norm_dist):
    norm_accum = 0
    for org, adv in zip(org_list, adv_list):
        for layer in range(org.shape[-1]):
            norm_accum += np.linalg.norm(adv[:, :, layer] - org[:, :, layer], ord = norm_dist)

    return norm_accum

print(f'test_num: {test_num};  adv_num: {adv_num}')
print('L_2: ', pertub_budget(x_test[:test_num], x_test_adv, norm_dist = 2))
print('L_inf: ', pertub_budget(x_test[:test_num], x_test_adv, norm_dist = np.inf))
print('Runtime: ', stop - start)
print('#'*30)
print('#'*30)

# DEFENCE
import defence.detector as detector
x_train_adv = attacker.generate_targeted_iterative(x_train[:test_num], x_test[0])
dmodel = detector.build_detector(classifier_model)
# detector.train(dmodel, x_test[:test_num], x_test_adv)
detector.train(dmodel,x_train[:test_num], x_train_adv)
detector.result(dmodel, x_test[:test_num], x_test_adv)

# print(x_test_adv_pred)
# print(np.argmax(y_test[:num], axis=1))
