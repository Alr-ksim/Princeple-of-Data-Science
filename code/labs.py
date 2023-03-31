import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from reduce import feature_select, feature_learn, feature_project


def load_dataset(features_file, labels_file):
    source = 'AwA2'
    features = np.loadtxt(os.path.join(source, features_file))
    labels = np.loadtxt(os.path.join(source, labels_file))
    return features, labels


def split_train_test(features, labels, test_size=0.4, random_state=233):
    x_train, x_test, y_train, y_test = \
        train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
    return x_train, x_test, y_train, y_test


def find_best_c(x_train, y_train):
    # Use K-fold cross-validation to find the best C parameter.
    param_grid = {'C': [0.1, 1, 10, 100, 200, 500]}
    grid_search = GridSearchCV(LinearSVC(), param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_c = grid_search.best_params_['C']
    print(f'The best C is {best_c}')
    return best_c


def svm_classify(x_train, x_test, y_train, y_test, mode='learn'):
    # Train the linear SVM with the best C parameter.
    svm = LinearSVC()
    if mode == 'select':
        x_train, x_test = feature_select(svm, x_train, x_test, y_train)
    elif mode == 'project':
        x_train, x_test = feature_project(x_train, x_test, y_train)
    elif mode == 'learn':
        x_train, x_test = feature_learn(x_train, x_test, y_train)

    best_c = find_best_c(x_train, y_train)
    print(f"Best C is {best_c}")
    # best_c = 1e-3
    svm = LinearSVC(C=best_c)

    print(f'Dimension reduction using {mode} successfully, the reduced dimensions is {x_train.shape[1]}')
    svm.fit(x_train, y_train)

    # Evaluate the SVM on the test set.
    y_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
