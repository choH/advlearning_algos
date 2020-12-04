import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

from sklearn import datasets
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import BasicIterativeMethod

from art.config import ART_DATA_PATH
from art.utils import load_dataset, get_file
from art.estimators.classification import SklearnClassifier
# from art.attacks.evasion import HopSkipJump
from hop_skip_jump import HopSkipJump

import warnings
warnings.filterwarnings('ignore')



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

attacker = HopSkipJump(classifier=classifier, targeted=False, norm=np.inf, max_iter=30, max_eval=10, init_eval=10, init_size=1)
# Paper uses max_iter=64, max_eval=10000, init_eval=100 but thats ultra-mega slow.
# To make it a fair comparision with FGSM and stuff, should limit init_eval.

adv_num = 3 # Make it bigger for actually benchmarking
x_test_adv = attacker.generate(x_test[:adv_num])

x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test[:adv_num], axis=1))


print(f"Adversarial test data (first {adv_num} images):")
print("Correctly classified: {}".format(nb_correct_adv_pred))
print("Incorrectly classified: {}".format(adv_num-nb_correct_adv_pred))



# for i in range(adv_num):
#     plt.matshow(x_test_adv[i].reshape((28, 28))); # Inspect if too pertub and lost sematic.


print('Predicted label:', x_test_adv_pred)
print('True label:', np.argmax(y_test[:adv_num], axis=1))
