from typing import Dict, Union, List
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
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

    def split_data(self):
        """split data into training and test data

        Parameters
        ----------
        data : [type]
            text data (vectorized)
        labels : [type]
            labels
        """
        self.df = pd.DataFrame(data=self.dataset)
        self.sentences = self.df["title"]
        self.labels = self.df["id"]
        (
            self.data_train,
            self.data_test,
            self.label_train,
            self.label_test,
        ) = train_test_split(self.sentences, self.labels)

    def vectorize_data(self):
        """vectorize text data

        Returns
        -------
        [type]
            vectorized text data
        """
        if self.vectorizer == "CountVectorizer":
            vectorizer = CountVectorizer()
            vectorizer.fit(self.data_train)
            self.data_train = vectorizer.transform(self.data_train).toarray()
            self.data_test = vectorizer.transform(self.data_test).toarray()
        if self.vectorizer == "TfidfVectorizer":
            vectorizer = TfidfVectorizer()
            self.data_train = vectorizer.fit_transform(self.data_train).toarray()
            self.data_test = vectorizer.fit_transform(self.data_test).toarray()

    def train_classifier(self):
        """trains classfier"""
        self.split_data()
        self.vectorize_data()
        self.bayes_classifier = MultinomialNB()
        self.bayes_classifier.fit(self.data_train, self.label_train)

    def evaluate(self, output_dict: bool):
        """evaluate data"""
        self.accuracy = self.bayes_classifier.score(self.data_test, self.label_test)
        self.classfication_report = metrics.classification_report(
            self.label_test,
            self.bayes_classifier.predict(self.data_test),
            output_dict=output_dict,
        )
