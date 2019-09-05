from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

import csv

from load_svrt import load_svrt_parsing


for n in [10,100,1000]:
    print("------- SVRT by Adaboost -------")
    accuracies = []
    for i in range(23):
        problem = i + 1
        X, y = load_svrt_parsing(problem)
        print(X[0])
        print(y[0])
        X_train, y_train = X[:2000], y[:2000]
        X_test, y_test = X[9000:], y[9000:]
        clf = AdaBoostClassifier(n_estimators=n)
        print("Now fitting with Adaboost ...")
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print("Test accuracy:", accuracy)
        print("-------")

    with open('model/accuracy_n={}.csv'.format(n), 'a') as f:
        accuracies_writer = csv.writer(f)
        accuracies_writer.writerow(accuracies)
