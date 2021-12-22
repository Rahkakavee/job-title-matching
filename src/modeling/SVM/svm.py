from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold, cross_validate
from scipy import stats
import numpy as np


class SVMClassifier:
    """text classfication using Bayes Classification"""

    def __init__(self, train, test, y_train, y_test) -> None:
        n_estimators = 10
        self.clf = OneVsRestClassifier(
            BaggingClassifier(
                SVC(C=1.0, kernel="linear", gamma="scale"),
                max_samples=1.0,
                n_estimators=n_estimators,
                n_jobs=-1,
                verbose=True,
            )
        )

        # self.clf = OneVsRestClassifier(SVC(C=1.0, kernel="linear", gamma="scale"))

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

    def mean_cfi(self, result, metric):
        alpha = 0.05
        metric = result[metric]
        df = len(metric) - 1  # degree of freedom
        t_value = stats.t.ppf(1 - alpha / 2, df)
        std_ = np.std(metric, ddof=1)
        n = len(metric)

        lower = np.mean(metric) - (t_value * std_ / np.sqrt(n))
        upper = np.mean(metric) + (t_value * std_ / np.sqrt(n))

        return round(lower, 2), round(upper, 2)

    def cross_validate(self):
        n_estimators = 10
        self.clf = OneVsRestClassifier(
            BaggingClassifier(
                SVC(C=1.0, kernel="linear", gamma="scale"),
                max_samples=1.0,
                n_estimators=n_estimators,
                n_jobs=-1,
                verbose=True,
            )
        )

        scorings = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_micro",
            "recall_micro",
            "f1_micro",
        ]

        kfold = RepeatedKFold(n_splits=5, n_repeats=20)
        scores = cross_validate(
            estimator=self.clf, X=self.train, y=self.y_train, cv=kfold, scoring=scorings
        )

        results = {}
        for scoring in scorings:
            metric = "test_" + scoring
            lower, upper = self.mean_cfi(scores, metric)
            results.update({scoring: [np.mean(scores[metric]), f"[{lower}, {upper}]"]})
        return results
