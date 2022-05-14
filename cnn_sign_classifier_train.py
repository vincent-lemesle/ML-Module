import os
import sys
import random
import numpy as np
from csv import reader
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# ---------- DATA ----------

TEST_PATH = './dataset/sign_mnist_test/sign_mnist_test.csv'
TRAIN_PATH = './dataset/sign_mnist_train/sign_mnist_train.csv'


def load_data_from_file(file_name):
    data = []
    label = []
    with open(file_name, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for index, row in enumerate(csv_reader):
            if index == 0:
                continue
            label.append(int(row[0]))
            d = np.array(list(map(int, row[1:])))
            data.append(d.reshape(28, 28).tolist())
    return data, label


def load_data():
    x_train, y_train = load_data_from_file(TRAIN_PATH)
    x_test, y_test = load_data_from_file(TEST_PATH)
    return (x_train, y_train), (x_test, y_test)


print('LOADING DATA...')
(x_train, y_train), (x_test, y_test) = load_data()
data_shape = np.shape(x_train[0])
print('DATA LOADED')
print('data shape:', data_shape)
print('number of train data:', len(x_train))
print('number of test data:', len(x_test))


def get_model():
    # ---------- MODEL ----------
    model = models.Sequential()
    model.add(layers.Conv2D(24, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(48, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(48, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(26))

    # display current neural net architecture
    model.summary()

    print('MODEL COMPILE')
    # ---------- TRAIN   ----------
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def get_dataset_batch(batch_size, dataset):
    if dataset == "train":
        batch = [random.randint(0, len(x_train) - 1) for _ in range(batch_size)]
        return [x_train[i] for i in batch], [y_train[i] for i in batch]
    batch = [random.randint(0, len(x_test) - 1) for _ in range(batch_size)]
    return [x_test[i] for i in batch], [y_test[i] for i in batch]


def batch_training(model, batch_size, epoch):
    # batch training
    batch_x, batch_y = get_dataset_batch(batch_size, "train")
    batch_x_test, batch_y_test = get_dataset_batch(400, "test")
    print('Training model on:', len(batch_x), 'cases...')
    history = model.fit(batch_x, batch_y, batch_size=50, epochs=epoch, validation_data=(batch_x_test, batch_y_test))
    return history.history['accuracy'], history.history['loss'], history.history['val_accuracy']


def train():
    EPOCH = 5
    BATCH_SIZE = 50 * 10
    NUMBER_OF_BATCH = 5

    model = get_model()
    accuracy, loss, val_accuracy = [], [], []

    for i in range(NUMBER_OF_BATCH):
        print('\nBATCH', i + 1)
        acc, lo, val_acc = batch_training(model, BATCH_SIZE, EPOCH)
        loss += lo
        accuracy += acc
        val_accuracy += val_acc
    print('Training finished.')

    # ---------- METRICS ----------
    is_show_data = input('Do you want to see the accuracy graph [Y/n]: ')
    if is_show_data == 'Y':
        plt.plot(accuracy, label='accuracy')
        plt.plot(val_accuracy, label='val_accuracy')
        plt.plot(loss, label='loss')
        plt.xlabel('Epoch')
        plt.ylim([0, 5])
        plt.legend(loc='lower right')
        plt.show()

    # ---------- EVALUATING ----------
    is_evaluate = input('Do you want to evaluate the model on test data [Y/n]: ')
    if is_evaluate == 'Y':
        print('Evaluating model on:', len(x_test), 'test cases...')
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print('Test accuracy:', str(round(test_acc, 2)) + '%')

    is_save = input('Do you want to save the model [Y/n]: ')
    if is_save == 'Y':
        print('Saving model...')
        model.save('./model')
        print('Model saved')

if len(sys.argv) == 1:
    train()
