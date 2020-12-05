# Defensive Distillation implemenation for adversarial learning group
# Author: Austin Keppers

import tensorflow as tf
import numpy as np
import json
from sys import argv
from tensorflow.python.framework.ops import disable_eager_execution
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

class Dataset:
    """
    Class to represent a data set
    Used as a way to combine the images and labels into one class
    """

    def __init__(self, images, labels, divide_images=True):

        # The images need to be converted to floats in order to work with the classifier
        if divide_images:
            self.images = images / 255.0
        else:
            self.images = images

        # An extra dimension needs to be added for greyscale images
        if self.images[0].ndim < 3:
            self.images = self.images.reshape(self.images.shape[0], self.images.shape[1], self.images.shape[2], 1)
        self.labels = labels

class SoftmaxTemperature(tf.keras.layers.Layer):
    """
    Custom layer to have a softmax layer with a temperature perameter
    Temperature determines how close the probabilities are to each
    Custom temperature value is not used when distilled is true and the classifier is not being trained.
    """
    
    def __init__(self, temperature, distilled):
        super(SoftmaxTemperature, self).__init__()
        self.temperature = temperature
        self.distilled = distilled

    def call(self, inputs, training=None):
        if training or not self.distilled:
            temp = self.temperature
        else:
            temp = 5
        output = tf.math.exp((inputs / self.temperature))
        normalized_output = output / tf.math.reduce_sum(output, axis=1, keepdims=True)
        return normalized_output

def get_dataset(dataset_name):
    """
    Loads a dataset from keras by name
    Returns tuple of images and labels in numpy arrays
    """
    dataset = getattr(tf.keras.datasets, dataset_name.lower())
    return dataset.load_data()

def get_model(parameters, dataset, distilled=False):
    """
    Create a convolutional neural network model according to parameters passed
    parameters is a dictionary of names and values. Look at mnist_dd.json and cifar10_dd.json for field examples
    distilled will determine whether the temperature is changed when not training
    """

    model = tf.keras.models.Sequential()

    # Create first set of convolutional layers and a pooling layer
    num_conv1_layers = parameters['conv1_layers']
    conv1_filters = parameters['conv1_num_filters']
    model.add(tf.keras.layers.Conv2D(conv1_filters, (3, 3), activation='relu', input_shape=dataset.images[0].shape))
    for i in range(num_conv1_layers - 1):
        model.add(tf.keras.layers.Conv2D(conv1_filters, (3, 3), activation='relu'))
        
    # Create second send of convolutional layers and a pooling layer
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    num_conv2_layers = parameters['conv2_layers']
    conv2_filters = parameters['conv2_num_filters']
    for i in range(num_conv2_layers):
        model.add(tf.keras.layers.Conv2D(conv2_filters, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Create dense and dropoutlayers
    model.add(tf.keras.layers.Flatten())
    num_dense_units = parameters['num_units']
    model.add(tf.keras.layers.Dropout(parameters['dropout_rate']))
    model.add(tf.keras.layers.Dense(num_dense_units, activation='relu'))
    model.add(tf.keras.layers.Dropout(parameters['dropout_rate']))
    model.add(tf.keras.layers.Dense(num_dense_units, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    if not distilled:
        model.add(SoftmaxTemperature(parameters['temperature'], False))
    else:
        model.add(SoftmaxTemperature(parameters['temperature'], True))
    return model

def train_model(model, training_dataset, testing_dataset, parameters, distilled=False):

    momentum=  parameters['momentum']

    # Use decay in the learning rate if it is not null in the parameters file
    if parameters['learning_rate_decay'] is None:
        learning_rate = parameters['learning_rate']
    else:
        learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
            parameters['learning_rate'], parameters['learning_rate_decay'], parameters['decay_delay'], staircase=True
        )

    # Choose loss based on whether distilled or not (sparse for not distilled due to the hard labels)
    if not distilled:
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['accuracy'])
    else:
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    history = model.fit(training_dataset.images, training_dataset.labels, epochs=parameters['epochs'],
                    validation_data=(testing_dataset.images, testing_dataset.labels),
                    batch_size=parameters['batch_size'])

def test_model(model, testing_dataset):
    """
    Test the model and print the accuracy
    """
    
    test_loss, test_acc = model.evaluate(testing_dataset.images,  testing_dataset.labels, verbose=2)
    print("Testing accuracy: {}".format(test_acc))

def get_soft_labels(model, dataset):
    """
    Using model based on hard labels get an array of soft labels
    """

    return model.predict(dataset.images, verbose=1)

def filter_soft_data_by_result(soft_dataset, hard_labels):
    """
    Take predictions and filter out examples that were misclassified
    If add_wrong_predictions is set to True will add these back in with hard labels
    """

    predictions = np.argmax(soft_dataset.labels, axis=1)

    # Find indices of correct predictions
    correct_predictions = (hard_labels == predictions)
    wrong_predictions = (hard_labels != predictions)

    return Dataset(soft_dataset.images[correct_predictions], soft_dataset.labels[correct_predictions], False)


def convert_to_one_hot(labels, num_classes):
    """
    Convert data to a one hot encoding (needed for distilled network)
    """
    return np.eye(num_classes)[labels.reshape(-1)]

def get_defensive_distilled_classifier(parameters, raw_training_data, raw_test_data, remove_incorrectly_predicted):
    """
    Train a distilled classifier
    First trains a classifier on hard labels and then a classifier on soft labels
    A tuple containing both classifiers is returned
    parameters are the model parameters loaded form a file
    raw training data and raw test data are tuples of numpy arrays of images and then labels
    remove_incorrectly_predicted will remove examples the first classifier predicts incorrectly from the training set for the second
    """

    # Creat data sets
    training_dataset = Dataset(raw_training_data[0], raw_training_data[1].flatten())
    test_dataset = Dataset(raw_test_data[0], raw_test_data[1])

    # Train a model on hard labels
    model = get_model(parameters, training_dataset)
    train_model(model, training_dataset, test_dataset, parameters)

    # Train distilled model on soft labels
    distilled_model = get_model(parameters, training_dataset, True)
    soft_labels = get_soft_labels(model, training_dataset)
    distilled_training_set = Dataset(raw_training_data[0], soft_labels)
    if remove_incorrectly_predicted:    # Alter the incorrectly predicted classes if it is selected
        distilled_training_set = filter_soft_data_by_result(distilled_training_set, training_dataset.labels)
    distilled_test_set = Dataset(raw_test_data[0], convert_to_one_hot(raw_test_data[1], 10))
    train_model(distilled_model, distilled_training_set, distilled_test_set, parameters, True)

    return model, distilled_model

if __name__ == '__main__':
    if len(argv) >= 2:

        # Read command line arguments
        disable_eager_execution()
        parameters_path = argv[1]
        remove_incorrectly_predicted = False
        if len(argv) >= 3:
            remove_incorrectly_predicted = bool(argv[2])
        parameters = None

        # Read parameters from file
        with open(parameters_path) as parameters_file:
            parameters = json.load(parameters_file)

        # Get classifiers, wrap as ART classifiers
        raw_training_data, raw_test_data = get_dataset(parameters['dataset'])
        classifier, distilled_classifier = get_defensive_distilled_classifier(parameters, raw_training_data, raw_test_data, remove_incorrectly_predicted)
        test_dataset = Dataset(raw_test_data[0], raw_test_data[1])
        distilled_test_set = Dataset(raw_test_data[0], convert_to_one_hot(raw_test_data[1], 10))
        art_classifier = KerasClassifier(model=classifier, clip_values=(0, 1), use_logits=False)
        art_classifier_distilled = KerasClassifier(model=distilled_classifier, clip_values=(0, 1), use_logits=False)

        # Create adversarial data sets with low epsilon
        fast_gradient_method_low_epsil = FastGradientMethod(estimator=art_classifier, eps=0.1)
        fast_gradient_method_high_epsil = FastGradientMethod(estimator=art_classifier, eps=0.2)
        adversarial_low_epsil = fast_gradient_method_low_epsil.generate(x=test_dataset.images)
        adversarial_high_epsil = fast_gradient_method_high_epsil.generate(x=test_dataset.images)
        adversarial_low_dataset = Dataset(adversarial_low_epsil, raw_test_data[1], False)
        adversarial_high_dataset = Dataset(adversarial_high_epsil, raw_test_data[1], False)

        # Create adversarial data sets with high epsilon
        fast_gradient_method_low_epsil_distilled = FastGradientMethod(estimator=art_classifier_distilled, eps=0.1)
        fast_gradient_method_high_epsil_distilled = FastGradientMethod(estimator=art_classifier_distilled, eps=0.3)
        adversarial_low_epsil_distilled = fast_gradient_method_low_epsil_distilled.generate(x=test_dataset.images)
        adversarial_high_epsil_distilled = fast_gradient_method_high_epsil_distilled.generate(x=test_dataset.images)
        adversarial_low_dataset_distilled = Dataset(adversarial_low_epsil_distilled, convert_to_one_hot(raw_test_data[1], 10), False)
        adversarial_high_dataset_distilled = Dataset(adversarial_high_epsil_distilled, convert_to_one_hot(raw_test_data[1], 10), False)

        # Print averages for different classifiers and adversarial datasets
        print("Non-distilled normal examples")
        test_model(classifier, test_dataset)
        print("Distilled normal examples")
        test_model(distilled_classifier, distilled_test_set)
        print("Non-distilled adversarial 0.1 epsilon examples")
        test_model(classifier, adversarial_low_dataset)
        print("Non-distilled adversarial 0.3 epsilon examples")
        test_model(classifier, adversarial_high_dataset)
        print("Distilled adversarial 0.1 epsilon examples")
        test_model(distilled_classifier, adversarial_low_dataset_distilled)
        print("Distilled adversarial 0.3 epsilon examples")
        test_model(distilled_classifier, adversarial_high_dataset_distilled)
    else:
        print("ERROR: No path to parameters file provided")
