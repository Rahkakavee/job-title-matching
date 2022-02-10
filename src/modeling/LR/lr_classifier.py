from numpy import result_type
from sklearn.metrics import precision_score, recall_score, f1_score
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
        )

        self.train = train
        self.test = test
        self.y_train = y_train
        self.y_test = y_test

    def fit_classifier(self):
        self.clf.fit(self.train, self.y_train)

    def evaluate(self, output_dict: bool) -> dict:
        """evaluate data"""
        accuracy = self.clf.score(X=self.test, y=self.y_test)

        predictions = self.clf.predict(self.test)

        precision_score_macro = precision_score(
            y_pred=predictions, y_true=self.y_test, average="macro"
        )

        precision_score_micro = precision_score(
            y_pred=predictions, y_true=self.y_test, average="micro"
        )

        recall_score_macro = recall_score(
            y_pred=predictions, y_true=self.y_test, average="macro"
        )

        recall_score_micro = recall_score(
            y_pred=predictions, y_true=self.y_test, average="micro"
        )

        f1_score_macro = f1_score(
            y_pred=predictions, y_true=self.y_test, average="macro"
        )

        f1_score_micro = f1_score(
            y_pred=predictions,
            y_true=self.y_test,
            average="micro",
        )

        classification_report = {
            "accuracy": accuracy,
            "macro": {
                "precision": precision_score_macro,
                "recall": recall_score_macro,
                "f1-score": f1_score_macro,
            },
            "micro": {
                "precision": precision_score_micro,
                "recall": recall_score_micro,
                "f1-score": f1_score_micro,
            },
        }
        return classification_report

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
        self.clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=10000,
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
