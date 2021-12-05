from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics


class SVMClassifier:
    """text classfication using Bayes Classification"""

    def __init__(self, train, test, y_train, y_test) -> None:
        n_estimators = 10
        self.clf = OneVsRestClassifier(
            BaggingClassifier(
                SVC(C=1.0, kernel="linear", gamma="scale"),
                max_samples=1.0 / n_estimators,
                n_estimators=n_estimators,
                n_jobs=-1,
                verbose=True,
            )
        )
        self.train = train
        self.test = test
        self.y_train = y_train
        self.y_test = y_test

    def fit_classifier(self):
        self.clf.fit(self.train, self.y_train)

    def evaluate(self, output_dict: bool) -> None:
        """evaluate data"""
        classfication_report = metrics.classification_report(
            self.y_test,
            self.clf.predict(self.test),
            output_dict=output_dict,
        )
        return classfication_report
