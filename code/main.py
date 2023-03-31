from labs import *

features_file = "AwA2-features.txt"
labels_file = "AwA2-labels.txt"
features, labels = load_dataset(features_file, labels_file)

X_train, X_test, y_train, y_test = split_train_test(features, labels, test_size=0.4)

acc = svm_classify(X_train, X_test, y_train, y_test)

print(f"Test accuracy: {acc:.2f}")
