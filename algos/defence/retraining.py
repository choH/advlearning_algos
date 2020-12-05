from art.defences.transformer.poisoning import NeuralCleanse
import numpy as np

def ganSample(model, clean_x_test, clean_y_test):
    clean_data = []
    backdoor_data = []
    backdoor_labels = []
    cleanse = NeuralCleanse(model)
    defence_cleanse = cleanse(model, steps=10, learning_rate=0.1)
    for backdoored_label, mask, pattern in defence_cleanse.outlier_detection(clean_x_test, clean_y_test):
        # find data based on the backdoored_label
        data_for_class = np.copy(clean_x_test[np.argmax(clean_y_test, axis=1) == backdoored_label])
        labels_for_class = np.copy(clean_y_test[np.argmax(clean_y_test, axis=1) == backdoored_label])
        clean_data.append(np.copy(data_for_class))
        data_for_class = (1 - mask) * data_for_class + mask * pattern
        backdoor_data.append(data_for_class)
        backdoor_labels.append(labels_for_class)
    if len(backdoor_data)!=0:
        clean_data = np.vstack(clean_data)
        backdoor_data = np.vstack(backdoor_data)
        backdoor_labels = np.vstack(backdoor_labels)
    return backdoor_data, backdoor_labels

def retrain(model, backdoor_data, backdoor_labels):
    model.fit(backdoor_data, backdoor_labels, batch_size=1, nb_epochs=1)
    return 0

def result(model, clean_x_test, clean_y_test, poison_x_test, poison_y_test):
    poison_preds = np.argmax(model.predict(poison_x_test), axis=1)
    poison_correct = np.sum(poison_preds == np.argmax(poison_y_test, axis=1))
    poison_total = poison_y_test.shape[0]
    new_poison_acc = poison_correct / poison_total
    print("\n Effectiveness of poison after unlearning: %.2f%%" % (new_poison_acc * 100))
    clean_preds = np.argmax(model.predict(clean_x_test), axis=1)
    clean_correct = np.sum(clean_preds == np.argmax(clean_y_test, axis=1))
    clean_total = clean_y_test.shape[0]
    new_clean_acc = clean_correct / clean_total
    print("\n Clean test set accuracy: %.2f%%" % (new_clean_acc * 100))

