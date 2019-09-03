from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

from load_svrt import load_svrt_parsing

print("------- SVRT by Adaboost -------")

for i in range(23):
    problem = i + 1
    X, y = load_svrt_parsing(problem)
    X_train, y_train = X[:9000], y[:9000]
    X_test, y_test = X[9000:], y[9000:]
    clf = AdaBoostClassifier(n_estimators=10)
    print("Now fitting with Adaboost ...")
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("-------")
