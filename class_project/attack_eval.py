import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model

from art.config import ART_DATA_PATH
from art.utils import load_dataset, get_file
from art.estimators.classification import KerasClassifier

from art.attacks.evasion import BasicIterativeMethod
# from art.defences.trainer import AdversarialTrainer


from art.attacks.evasion import FastGradientMethod
from fast_gradient_sign_method import FastGradientSignMethod
from hop_skip_jump import HopSkipJump
from deepfool import DeepFool
from dynamic_deepfool import DynamicDeepFool


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

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')
path = get_file('cifar_resnet.h5',extract=False, path=ART_DATA_PATH,
                url='https://www.dropbox.com/s/ta75pl4krya5djj/cifar_resnet.h5?dl=1')
classifier_model = load_model(path)


test_num = 30
adv_num = 30

classifier = KerasClassifier(clip_values=(min_, max_), model=classifier_model, use_logits=False)

classifier_model.summary()
x_test_pred = np.argmax(classifier.predict(x_test[:test_num]), axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test[:test_num], axis=1))
print(f"Original test data (first {test_num} images):")
print("Correctly classified: {}".format(nb_correct_pred))
print("Incorrectly classified: {}".format(test_num-nb_correct_pred))

# # FGSM with extensions
# attacker = FastGradientSignMethod(classifier, eps=0.5, batch_size = 8)
# x_test_adv = attacker.generate(x_test[:adv_num]) # non-targeted
# x_test_adv = attacker.generate_targeted(x_test[:adv_num], x_test[0]) #targeted
# x_test_adv = attacker.generate_iterative(x_test[:adv_num]) #iterative non-targeted
# x_test_adv = attacker.generate_targeted_iterative(x_test[:adv_num], x_test[0]) #iterative targeted

# # Hop Skip Jump: Paper uses max_iter=64, max_eval=10000, init_eval=100 but thats ultra-mega slow.
# attacker = HopSkipJump(classifier=classifier, targeted=False, norm=np.inf, max_iter=30, max_eval=10, init_eval=10, init_size=1)
# x_test_adv = attacker.generate(x_test[:adv_num])


# # DeepFool
# attacker = DeepFool(classifier)
# x_test_adv = attacker.generate(x_test[:adv_num])

# # DeepFool extension
attacker = DynamicDeepFool(classifier)
x_test_adv = attacker.generate(x_test[:adv_num])


x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test[:adv_num], axis=1))

print(f"Adversarial test data (first {adv_num} images):")
print("Correctly classified: {}".format(nb_correct_adv_pred))
print("Incorrectly classified: {}".format(adv_num-nb_correct_adv_pred))
# nb_preb_as_tar = np.count_nonzero(x_test_adv_pred == np.argmax(y_test[:adv_num], axis=1)[0])
# print(f"Classified as targeted label: {nb_preb_as_tar}\n\n")

print('Predicted label:', x_test_adv_pred)
print('True label:', np.argmax(y_test[:adv_num], axis=1))


# print(x_test_adv_pred)
# print(np.argmax(y_test[:num], axis=1))
