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

sentences_short_tokens = [word_tokenize(sentence) for sentence in sentences_short]
sentences_long_tokens = [word_tokenize(sentence) for sentence in sentences_long]

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

prior_knowledge_cleaned_sentences = [
    sentence["title"] for sentence in prior_knowledge_cleaned
]

prior_knowledge_cleaned_tokens = [
    word_tokenize(sentence) for sentence in prior_knowledge_cleaned_sentences
]

embedding_data = prior_knowledge_cleaned_tokens + sentences_long_tokens

tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(embedding_data)]

model = Doc2Vec(tagged_data, vector_size=20, window=5, min_count=1, epochs=100)
