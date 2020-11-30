import tensorflow as tf
import numpy as np
import json
from sys import argv

class Dataset:

    def __init__(self, images, labels, divide_images=True):
        if divide_images:
            self.images = images / 255.0
        else:
            self.images = images
        if self.images[0].ndim < 3:
            self.images = self.images.reshape(self.images.shape[0], self.images.shape[1], self.images.shape[2], 1)
        self.labels = labels

class SoftmaxTemperature(tf.keras.layers.Layer):
    
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
    dataset = getattr(tf.keras.datasets, dataset_name.lower())
    return dataset.load_data()

def get_model(parameters, dataset, distilled=False):

    model = tf.keras.models.Sequential()
    num_conv1_layers = parameters['conv1_layers']
    conv1_filters = parameters['conv1_num_filters']
    model.add(tf.keras.layers.Conv2D(conv1_filters, (3, 3), activation='relu', input_shape=dataset.images[0].shape))
    for i in range(num_conv1_layers - 1):
        model.add(tf.keras.layers.Conv2D(conv1_filters, (3, 3), activation='relu'))
        
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    num_conv2_layers = parameters['conv2_layers']
    conv2_filters = parameters['conv2_num_filters']
    for i in range(num_conv2_layers):
        model.add(tf.keras.layers.Conv2D(conv2_filters, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

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

    #if parameters['momentum_decay'] is None:
    momentum=  parameters['momentum']
    #else:
        #momentum = tf.keras.optimizers.schedules.InverseTimeDecay(
        #    parameters['momentum'], parameters['momentum_decay'], parameters['decay_delay'], staircase=True
        #)

    if parameters['learning_rate_decay'] is None:
        learning_rate = parameters['learning_rate']
    else:
        learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
            parameters['learning_rate'], parameters['learning_rate_decay'], parameters['decay_delay'], staircase=True
        )

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
    test_loss, test_acc = model.evaluate(testing_dataset.images,  testing_dataset.labels, verbose=2)
    print("Testing accuracy: {}".format(test_acc))

def get_soft_labels(model, dataset):
    return model.predict(dataset.images, verbose=1)

def filter_soft_data_by_result(soft_dataset, hard_labels):
    predictions = np.argmax(soft_dataset.labels, axis=1)
    correct_predictions = (hard_labels == predictions)
    return Dataset(soft_dataset.images[correct_predictions], soft_dataset.labels[correct_predictions], False)

def convert_to_one_hot(labels, num_classes):
    return np.eye(num_classes)[labels.reshape(-1)]

def get_defensive_distilled_classifier(parameters, raw_training_data, raw_test_data, remove_incorrectly_predicted):
    training_dataset = Dataset(raw_training_data[0], raw_training_data[1].flatten())
    test_dataset = Dataset(raw_test_data[0], raw_test_data[1])
    model = get_model(parameters, training_dataset)
    train_model(model, training_dataset, test_dataset, parameters)
    #test_model(model, test_dataset)
    distilled_model = get_model(parameters, training_dataset, True)
    soft_labels = get_soft_labels(model, training_dataset)
    distilled_training_set = Dataset(raw_training_data[0], soft_labels)
    if remove_incorrectly_predicted:
        distilled_training_set = filter_soft_data_by_result(distilled_training_set, training_dataset.labels)
    distilled_test_set = Dataset(raw_test_data[0], convert_to_one_hot(raw_test_data[1], 10))
    train_model(distilled_model, distilled_training_set, distilled_test_set, parameters, True)
    return model, distilled_model
    #test_model(distilled_model, distilled_test_set)

if __name__ == '__main__':
    if len(argv) >= 2:
        parameters_path = argv[1]
        remove_incorrectly_predicted = False
        if len(argv) >= 3:
            remove_incorrectly_predicted = bool(argv[2])
        parameters = None
        with open(parameters_path) as parameters_file:
            parameters = json.load(parameters_file)
        raw_training_data, raw_test_data = get_dataset(parameters['dataset'])
        classifier, distilled_classifier = get_defensive_distilled_classifier(parameters, raw_training_data, raw_test_data, remove_incorrectly_predicted)
        #fast_gradient_method = FastGradientMethod(estimator=classifier, eps=0.2)
        test_dataset = Dataset(raw_test_data[0], raw_test_data[1])
        distilled_test_set = Dataset(raw_test_data[0], convert_to_one_hot(raw_test_data[1], 10))
        # adversarial_test_images = attack.generate(x=raw_test_data[0])
        # adversarial_dataset = Dataset(adversarial_test_images, raw_test_data[1])
        # test_model(classifier, adversarial_dataset)
        # test_model(distilled_classifier, adversarial_dataset)
        test_model(classifier, test_dataset)
        test_model(distilled_classifier, distilled_test_set)
    else:
        print("ERROR: No path to parameters file provided")
