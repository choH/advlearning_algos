import warnings
warnings.filterwarnings('ignore')

import os, sys
from os.path import abspath

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import warnings
warnings.filterwarnings('ignore')

from keras.models import load_model

from art import config
from art.utils import load_dataset, get_file
from art.estimators.classification import KerasClassifier
# from art.attacks.poisoning import FeatureCollisionAttack
from feature_collision import FeatureCollisionAttack
# from feature_collision import FeatureCollision

import numpy as np

# %matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(301)

print(tf.__version__)

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')

num_samples_train = 1000
num_samples_test = 1000
x_train = x_train[0:num_samples_train]
y_train = y_train[0:num_samples_train]
x_test = x_test[0:num_samples_test]
y_test = y_test[0:num_samples_test]

class_descr = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

path = get_file('cifar_alexnet.h5',extract=False, path=config.ART_DATA_PATH,
                url='https://www.dropbox.com/s/ta75pl4krya5djj/cifar_alexnet.h5?dl=1')
classifier_model = load_model(path)
classifier = KerasClassifier(clip_values=(min_, max_), model=classifier_model, use_logits=False, preprocessing=(0.5, 1))



target_class = "bird" # one of ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
target_label = np.zeros(len(class_descr))
target_label[class_descr.index(target_class)] = 1
target_instance = np.expand_dims(x_test[np.argmax(y_test, axis=1) == class_descr.index(target_class)][3], axis=0)

# fig = plt.imshow(target_instance[0])
print('true_class: ' + target_class)
print('predicted_class: ' + class_descr[np.argmax(classifier.predict(target_instance), axis=1)[0]])

feature_layer = classifier.layer_names[-2]


base_class = "frog" # one of ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
base_idxs = np.argmax(y_test, axis=1) == class_descr.index(base_class)
base_instances = np.copy(x_test[base_idxs][:10])
base_labels = y_test[base_idxs][:10]

x_test_pred = np.argmax(classifier.predict(base_instances), axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(base_labels, axis=1))

print("New test data to be poisoned (10 images):")
print("Correctly classified: {}".format(nb_correct_pred))
print("Incorrectly classified: {}".format(10-nb_correct_pred))


# plt.figure(figsize=(10,10))
# for i in range(0, 9):
#     pred_label, true_label = class_descr[x_test_pred[i]], class_descr[np.argmax(base_labels[i])]
#     plt.subplot(330 + 1 + i)
#     fig=plt.imshow(base_instances[i])
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     fig.axes.text(0.5, -0.1, pred_label + " (" + true_label + ")", fontsize=12, transform=fig.axes.transAxes,
#                   horizontalalignment='center')


attack = FeatureCollisionAttack(classifier, target_instance, feature_layer, max_iter=10, similarity_coeff=256, watermark=0.3)
poison, poison_labels = attack.poison(base_instances)


poison_pred = np.argmax(classifier.predict(poison), axis=1)
# plt.figure(figsize=(10,10))
# for i in range(0, 9):
#     pred_label, true_label = class_descr[poison_pred[i]], class_descr[np.argmax(poison_labels[i])]
#     plt.subplot(330 + 1 + i)
#     fig=plt.imshow(poison[i])
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     fig.axes.text(0.5, -0.1, pred_label + " (" + true_label + ")", fontsize=12, transform=fig.axes.transAxes,
#                   horizontalalignment='center')


classifier.set_learning_phase(True)
print(x_train.shape)
print(base_instances.shape)
adv_train = np.vstack([x_train, poison])
adv_labels = np.vstack([y_train, poison_labels])
classifier.fit(adv_train, adv_labels, nb_epochs=5, batch_size=4)

fig = plt.imshow(target_instance[0])

print('true_class: ' + target_class)
print('predicted_class: ' + class_descr[np.argmax(classifier.predict(target_instance), axis=1)[0]])