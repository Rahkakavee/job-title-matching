from typing import Dict, Union, List
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import logging
import sys
from sklearn.ensemble import BaggingClassifier

from src.logger import logger


class SVMClassifier:
    def __init__(self, data: Union[List, Dict], vectorizer: str) -> None:
        """init

        Parameters
        ----------
        data : Union[List, Dict]
            data with text data and labels

        vectorizer: feature selection method
            CountVectorizer or TfidfVectorizer
        """
        self.dataset = data
        self.vectorizer = vectorizer
        self.data_train = []
        self.data_test = []
        self.label_train = []
        self.label_test = []
        self.train = []
        self.test = []

    def split_data(self) -> None:
        """split data into training and test data

        Parameters
        ----------
        data : [type]
            text data (vectorized)
        labels : [type]
            labels
        """
        df = pd.DataFrame(data=self.dataset)
        sentences = df["title"]
        labels = df["id"]
        (
            self.data_train,
            self.data_test,
            self.label_train,
            self.label_test,
        ) = train_test_split(sentences, labels)

    def vectorize_data(self) -> None:
        """Feature selection"""
        if self.vectorizer == "CountVectorizer":
            vectorizer = CountVectorizer()
            vectorizer.fit(self.data_train)
            self.train = vectorizer.transform(self.data_train).toarray()
            self.test = vectorizer.transform(self.data_test).toarray()
        if self.vectorizer == "TfidfVectorizer":
            vectorizer = TfidfVectorizer()
            vectorizer.fit(self.data_train)
            self.train = vectorizer.transform(self.data_train).toarray()
            self.test = vectorizer.transform(self.data_test).toarray()

    def train_classifier(self):
        """trains classfier"""
        logger.debug("Split TrainingData")
        self.split_data()
        logger.debug("Vectorize Data")
        self.vectorize_data()
        logger.debug("Train the classfier")
        n_estimators = 10
        self.clf = OneVsRestClassifier(
            BaggingClassifier(
                SVC(C=1.0, kernel="linear", gamma="scale"),
                max_samples=1.0,
                n_estimators=n_estimators,
                n_jobs=-1,
            )
        )
        self.clf.fit(self.train, self.label_train)

    def evaluate(self, output_dict: bool):
        """evaluate data"""
        self.accuracy = self.clf.score(self.test, self.label_test)
        self.classfication_report = metrics.classification_report(
            self.label_test,
            self.clf.predict(self.test),
            output_dict=output_dict,
        )
