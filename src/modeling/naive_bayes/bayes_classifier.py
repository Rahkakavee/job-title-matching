from typing import Dict, Union, List
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


class BayesClassifier:
    """text classfication using Bayes Classification"""

    def __init__(self, data: Union[List, Dict], vectorizer: str) -> None:
        """init

        Parameters
        ----------
        data : Union[List, Dict]
            data with text data and labels
        """
        self.dataset = data
        self.vectorizer = vectorizer

    def vectorize_data(self):
        """vectorize text data

        Returns
        -------
        [type]
            vectorized text data
        """
        self.df = pd.DataFrame(data=self.dataset)
        if self.vectorizer == "CountVectorizer":
            vectorizer = CountVectorizer()
            data = vectorizer.fit_transform(self.df["title"]).toarray()
        if self.vectorizer == "TfidfVectorizer":
            vectorizer = TfidfVectorizer()
            data = vectorizer.fit_transform(self.df["title"]).toarray()
        return data

    def extract_labels(self):
        """extract labels

        Returns
        -------
        [type]
            labels (classes)
        """
        labels = self.df.iloc[:, 0]
        return labels

    def split_data(self, data, labels):
        """split data into training and test data

        Parameters
        ----------
        data : [type]
            text data (vectorized)
        labels : [type]
            labels
        """
        (
            self.data_train,
            self.data_test,
            self.label_train,
            self.label_test,
        ) = train_test_split(data, labels)

    def train_classifier(self):
        """trains classfier"""
        data = self.vectorize_data()
        labels = self.extract_labels()
        self.split_data(data=data, labels=labels)
        self.bayes_classifier = GaussianNB()
        self.bayes_classifier.fit(self.data_train, self.label_train)

    def evaluate(self, output_dict: bool):
        """evaluate data"""
        self.accuracy = self.bayes_classifier.score(self.data_test, self.label_test)
        self.classfication_report = metrics.classification_report(
            self.label_test,
            self.bayes_classifier.predict(self.data_test),
            output_dict=output_dict,
        )
