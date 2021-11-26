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
    def __init__(self, data: List[Dict], vectorizer: str) -> None:
        """init

        Parameters
        ----------
        data : Union[List, Dict]
            data with text data and labels
        """
        self.dataset = data
        self.vectorizer = vectorizer
        self.df = pd.DataFrame()

    # TODO: dimension CountVectorizer TF-IDF --> pruning?
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
            vectorizer.fit(self.df["title"])
            data = vectorizer.transform(self.df["title"])
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
        ) = train_test_split(data, labels, test_size=0.20, random_state=1000)

    def train_classifier(self):
        """trains classfier"""
        logger.debug("Vectorize Data")
        data = self.vectorize_data()
        logger.debug("Extract Labels")
        labels = self.extract_labels()
        logger.debug("Split TrainingData")
        self.split_data(data=data, labels=labels)
        logger.debug("Train the classfier")
        n_estimators = 10
        self.svm_classifier = OneVsRestClassifier(
            BaggingClassifier(
                SVC(C=1.0, kernel="linear", gamma="scale"),
                max_samples=1.0,
                n_estimators=n_estimators,
                n_jobs=-1,
            )
        )
        self.svm_classifier.fit(self.data_train, self.label_train)

    def evaluate(self, output_dict: bool):
        """evaluate data"""
        self.accuracy = self.svm_classifier.score(self.data_test, self.label_test)
        self.classfication_report = metrics.classification_report(
            self.label_test,
            self.svm_classifier.predict(self.data_test),
            output_dict=output_dict,
        )
