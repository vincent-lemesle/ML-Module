import dataset
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

print('LOADING DATA...')
(x_train, y_train), (x_test, y_test) = dataset.load_data()
data_shape = np.shape(x_train[0])
print('DATA LOADED')
print('data shape:', data_shape)
print('number of train data:', len(x_train))
print('number of test data:', len(x_test))

train_format_data = [[item for sublist in x for item in sublist] for x in x_train]
test_format_data = [[item for sublist in x for item in sublist] for x in x_test]

print('\n---------  KNeighborsClassifier ---------\n')
model_knc = KNeighborsClassifier(n_neighbors=2)
print('evaluating model...')
model_knc.fit(train_format_data, y_train)
prediction = model_knc.predict(test_format_data)
correct_cases = 0
for pred, t in zip(prediction, y_test):
    if pred == t:
        correct_cases += 1
print('KNeighborsClassifier:', round(correct_cases / len(prediction) * 100, 2), '%')

print('\n---------  RandomForestClassifier ---------\n')
model_rfc = RandomForestClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
print('evaluating model...')
model_rfc.fit(train_format_data, y_train)
prediction = model_rfc.predict(test_format_data)
correct_cases = 0
for pred, t in zip(prediction, y_test):
    if pred == t:
        correct_cases += 1
print('KNeighborsClassifier:', round(correct_cases / len(prediction) * 100, 2), '%')
