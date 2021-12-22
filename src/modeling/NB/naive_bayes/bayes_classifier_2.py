from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB


class BayesClassifier:
    """text classfication using Bayes Classification"""

    def __init__(self, train, test, y_train, y_test) -> None:
        self.clf = MultinomialNB()
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
