"This file trains doc2vec model with and without prior knowledge"

import pickle
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import json
import gensim
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing_functions import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

with open(file="data/processed/training_data_long_l1.json") as fp:
    training_data_long = json.load(fp=fp)

with open(file="data/processed/training_data_short_l1.json") as fp:
    training_data_short = json.load(fp=fp)

with open(file="data/raw/dictionary_occupations_complete_update.json") as fp:
    kldbs = json.load(fp=fp)

# process data from BA data
sentences_long = [job["title"] for job in training_data_long]
labels_long = [job["id"] for job in training_data_long]
sentences_long_tokens = [word_tokenize(sentence) for sentence in sentences_long]

# create prior knowledge
prior_knowledge = []
for kldb in kldbs:
    if kldb["level"] == 5:
        for searchword in kldb["searchwords"]:
            prior_knowledge.append({"title": searchword["name"]})
with open(file="src/preprocessing/specialwords.txt", mode="rb") as fp:
    specialwords = pickle.load(fp)
prior_knowledge_cleaned = preprocess(
    data=prior_knowledge, special_words_ovr=specialwords
)
prior_knowledge_cleaned_sentences = [
    sentence["title"] for sentence in prior_knowledge_cleaned
]
prior_knowledge_cleaned_tokens = [
    word_tokenize(sentence) for sentence in prior_knowledge_cleaned_sentences
]

# create data for training
embedding_data = prior_knowledge_cleaned_tokens + sentences_long_tokens
embedding_data_tagged = [TaggedDocument(d, [i]) for i, d in enumerate(embedding_data)]
sentences_long_tokens_tagged = [
    TaggedDocument(d, [i]) for i, d in enumerate(sentences_long_tokens)
]

# create models
model_without_prior = Doc2Vec(
    sentences_long_tokens_tagged, vector_size=300, window=5, min_count=1, epochs=10
)
model_with_prior = Doc2Vec(
    embedding_data_tagged, vector_size=300, window=5, min_count=1, epochs=10
)

model_without_prior.save("src/vectorizer/Doc2vec/doc2_vec_without_prior")
model_with_prior.save("src/vectorizer/Doc2vec/doc2_vec_with_prior")

# TEST
sentences_short = [job["title"] for job in training_data_short]
labels_short = [job["id"] for job in training_data_short]
sentences_short_tokens = [word_tokenize(sentence) for sentence in sentences_short]


train, test, y_train, y_test = train_test_split(sentences_short_tokens, labels_short)


train_vecs = [model_without_prior.infer_vector(sent) for sent in train]
test_vecs = [model_without_prior.infer_vector(sent) for sent in test]
clf = LogisticRegression(max_iter=1000)
clf.fit(train_vecs, y_train)
score = clf.score(test_vecs, y_test)
print(score)

train_vecs = [model_with_prior.infer_vector(sent) for sent in train]
test_vecs = [model_with_prior.infer_vector(sent) for sent in test]
clf = LogisticRegression(max_iter=1000)
clf.fit(train_vecs, y_train)
score = clf.score(test_vecs, y_test)
print(score)
