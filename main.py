from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

import csv

from load_svrt import load_svrt_parsing


for n_e in [10,100,1000,10000]:
    print("------- SVRT by Adaboost -------")
    accuracies = []
    for i in range(23):
        problem = i + 1
        p, py, n, ny = load_svrt_parsing(problem)
        X_train, y_train = p[:40]+n[:40], py[:40]+ny[:40]
        X_test, y_test = p[40:]+n[40:], py[40:]+ny[40:]
        clf = AdaBoostClassifier(n_estimators=n_e)
        print("Now fitting with Adaboost ...")
        try:
            model = clf.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
        except Exception as e:
            accuracy = -1
        accuracies.append(accuracy)
        print("Test accuracy:", accuracy)
        print("-------")

    with open('model/50accuracy_sas.csv'.format(n), 'a') as f:
        accuracies_writer = csv.writer(f)
        accuracies_writer.writerow(accuracies)
