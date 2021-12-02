from src.preprocessing.preprocessing_functions import *
from src.preprocessing.training_data import TrainingData
from src.logger import logger
import pickle
import random
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logger.debug("#######TRAINING DATA#######")
# Training Data
data_level_1_old = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/processed/data_old_format.json",
    kldb_level=1,
    new_data=False,
)
data_level_1_new = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/processed/data_new_format.json",
    kldb_level=1,
    new_data=True,
)

data_level_1_old.create_training_data()
data_level_1_new.create_training_data()

training_data_level_1 = data_level_1_old.training_data + data_level_1_new.training_data

data = [
    dict(t) for t in {tuple(example.items()) for example in training_data_level_1}
]  # source: "https://stackoverflow.com/questions/9427163/remove-duplicate-dict-in-list-in-python"

with open("src/preprocessing/specialwords.tex", "rb") as fp:
    specialwords = pickle.load(fp)

logger.debug("#######Preprocessing#######")
# Preprocess
training_data = preprocess(
    data=data, lowercase_whitespace=False, special_words_ovr=specialwords
)

training_data_short = random.sample(training_data, 15000)

sentences = [job["title"] for job in training_data_short]
labels = [job["id"] for job in training_data_short]

sentences_tokenized = [word_tokenize(sentence) for sentence in sentences]

model = Word2Vec(sentences_tokenized, vector_size=100, window=5, min_count=1, workers=4)

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels)


train = []
for sent in sentences_train:
    sent_vec = np.zeros(100)
    cnt_words = 0
    for word in sent:
        if word in model.wv.key_to_index:
            vec = model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    train.append(sent_vec)


test = []
for sent in sentences_test:
    sent_vec = np.zeros(100)
    cnt_words = 0
    for word in sent:
        if word in model.wv.key_to_index:
            vec = model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    test.append(sent_vec)

tfidf_sent_vectors = []
row = 0
for sent in sentences_train:
    sent_vec = np.zeros(100)
    weight_sum = 0
    for word in sent:
        if word in model.wv.key_to_index and word in tfidf_feat:
            vec = model.wv[word]
            tf_idf = dictionary[word] * (sent.count(word) / len(sent))
            sent_vec += vec * tf_idf
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1

tfidf_sent_vectors_test = []
row = 0
for sent in sentences_test:
    sent_vec = np.zeros(100)
    weight_sum = 0
    for word in sent:
        if word in model.wv.key_to_index and word in tfidf_feat:
            vec = model.wv[word]
            tf_idf = dictionary[word] * (sent.count(word) / len(sent))
            sent_vec += vec * tf_idf
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors_test.append(sent_vec)
    row += 1


clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")
clf.fit(train, y_train)

metrics.classification_report(y_test, clf.predict(test), output_dict=True)
