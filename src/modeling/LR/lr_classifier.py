from numpy import result_type
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, cross_validate
from scipy import stats
import numpy as np


class LRClassifier:
    """text classfication using LR Classification"""

    def __init__(self, train, test, y_train, y_test) -> None:
        self.clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=10000,
            n_jobs=-1,
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

    def mean_cfi(self, result, metric):
        alpha = 0.05
        metric = result[metric]
        df = len(metric) - 1  # degree of freedom
        t_value = stats.t.ppf(1 - alpha / 2, df)
        std_ = np.std(metric, ddof=1)
        n = len(metric)

        lower = np.mean(metric) - (t_value * std_ / np.sqrt(n))
        upper = np.mean(metric) + (t_value * std_ / np.sqrt(n))

        return lower, upper

    def cross_validate(self):
        self.clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=10000,
            n_jobs=-1,
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
        results = []
        for scoring in scorings:
            metric = "test_" + scoring
            lower, upper = self.mean_cfi(scores, metric)
            results.append(
                {
                    "metric": scoring,
                    "mean": np.mean(scores[metric]),
                    "cfi": f"[{lower}, {upper}]",
                }
            )
        return results
