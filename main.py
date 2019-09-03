from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

from load_svrt import load_svrt_parsing

X, y = load_svrt_parsing(1)
X_train, y_train = X, y
X_test, y_test = X, y
clf = AdaBoostClassifier(n_estimators=100)
model = clf.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
