import numpy as np
from csv import reader

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
