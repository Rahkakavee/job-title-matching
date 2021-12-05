import json
import pickle
from src.preprocessing.preprocessing_functions import *
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

with open(file="data/processed/training_data_long.json") as fp:
    training_data_long = json.load(fp=fp)

with open(file="data/processed/training_data_short.json") as fp:
    training_data_short = json.load(fp=fp)

with open(file="data/raw/dictionary_occupations_complete_update.json") as fp:
    kldbs = json.load(fp=fp)

sentences_short = [job["title"] for job in training_data_short]
labels_short = [job["id"] for job in training_data_short]
sentences_long = [job["title"] for job in training_data_long]
labels_long = [job["id"] for job in training_data_long]


prior_knowledge = []
for kldb in kldbs:
    if kldb["level"] == 5:
        for searchword in kldb["searchwords"]:
            prior_knowledge.append({"title": searchword["name"]})

with open(file="src/preprocessing/specialwords.tex", mode="rb") as fp:
    specialwords = pickle.load(fp)

prior_knowledge_cleaned = preprocess(
    data=prior_knowledge, special_words_ovr=specialwords
)

sbert_model = SentenceTransformer("bert-base-nli-mean-tokens")

train, test, y_train, y_test = train_test_split(sentences_short, labels_short)

train_vec = sbert_model.encode(train)
test_vec = sbert_model.encode(test)

clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")
clf.fit(train_vec, y_train)
metrics.classification_report(y_test, clf.predict(test_vec), output_dict=True)

# TODO: BERT mit prior knowledge

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

train, test, y_train, y_test = train_test_split(sentences_short, labels_short)

train_vec = model(train)
test_vec = model(test)

clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")
clf.fit(train_vec, y_train)
metrics.classification_report(y_test, clf.predict(test_vec), output_dict=True)

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model_2 = hub.load(module_url)

# TODO: Nachher weitermachen mit model von google
