from gensim.models import KeyedVectors
import json
from nltk import word_tokenize
from gensim.models import Word2Vec
import pickle
from src.preprocessing.preprocessing_functions import *
from src.vectorizer.word2vec.average_word_2_vec import AverageWord2Vec
import numpy as np


# load google model
model = KeyedVectors.load_word2vec_format(
    "model/GoogleNews-vectors-negative300.bin.gz", binary=True
)

# get data
with open(file="data/processed/training_data_long_l1.json") as fp:
    training_data_long = json.load(fp=fp)
with open(file="data/processed/training_data_short_l1.json") as fp:
    training_data_short = json.load(fp=fp)
sentences_long = [el["title"] for el in training_data_long]
sentences_long_tokenized = [word_tokenize(sent) for sent in sentences_long]

# get knowledgebase
with open(file="data/raw/dictionary_occupations_complete_update.json") as fp:
    kldbs = json.load(fp=fp)
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

embedding_sentences = prior_knowledge_cleaned_tokens + sentences_long_tokenized


# without prior knowledge
custom_model_without_prior_knowledge = Word2Vec(vector_size=300, min_count=1)
custom_model_without_prior_knowledge.build_vocab(sentences_long_tokenized)
training_example_count = custom_model_without_prior_knowledge.corpus_count
custom_model_without_prior_knowledge.build_vocab(
    [list(model.key_to_index.keys())], update=True
)

custom_model_without_prior_knowledge.wv.vectors_lockf = np.ones(
    len(custom_model_without_prior_knowledge.wv)
)
custom_model_without_prior_knowledge.wv.intersect_word2vec_format(
    "model/GoogleNews-vectors-negative300.bin.gz", binary=True, lockf=1.0
)
custom_model_without_prior_knowledge.train(
    sentences_long_tokenized, total_examples=training_example_count, epochs=10
)

custom_model_without_prior_knowledge.save(
    "src/vectorizer/word2vec/word2vec_without_prior"
)


sentences_short = [el["title"] for el in training_data_short]

avvec = AverageWord2Vec(custom_model_without_prior_knowledge.wv)
training_vecs = avvec.vectorize_all(sentences_short)
labels = [el["id"] for el in training_data_short]


full_training_vecs = []
full_labels = []
for i in range(0, len(training_vecs)):
    if training_vecs[i].shape == (300,):
        full_training_vecs.append(training_vecs[i])
        full_labels.append(labels[i])

# Test
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

train, test, y_train, y_test = train_test_split(full_training_vecs, full_labels)
clf = LogisticRegression(max_iter=1000)
clf.fit(train, y_train)
score = clf.score(test, y_test)
print(score)

# with prior knowldege
custom_model_with_prior_knowledge = Word2Vec(vector_size=300, min_count=1)
custom_model_with_prior_knowledge.build_vocab(embedding_sentences)
training_example_count = custom_model_with_prior_knowledge.corpus_count
custom_model_with_prior_knowledge.build_vocab(
    [list(model.key_to_index.keys())], update=True
)
import numpy as np

custom_model_with_prior_knowledge.wv.vectors_lockf = np.ones(
    len(custom_model_with_prior_knowledge.wv)
)
custom_model_with_prior_knowledge.wv.intersect_word2vec_format(
    "model/GoogleNews-vectors-negative300.bin.gz", binary=True, lockf=1.0
)
custom_model_with_prior_knowledge.train(
    embedding_sentences, total_examples=training_example_count, epochs=10
)

custom_model_with_prior_knowledge.save("src/vectorizer/word2vec/word2vec_with_prior")

sentences_short = [el["title"] for el in training_data_short]

avvec = AverageWord2Vec(custom_model_with_prior_knowledge.wv)
training_vecs = avvec.vectorize_all(sentences_short)
labels = [el["id"] for el in training_data_short]


full_training_vecs = []
full_labels = []
for i in range(0, len(training_vecs)):
    if training_vecs[i].shape == (300,):
        full_training_vecs.append(training_vecs[i])
        full_labels.append(labels[i])

# Test

train, test, y_train, y_test = train_test_split(full_training_vecs, full_labels)
clf = LogisticRegression(max_iter=1000)
clf.fit(train, y_train)
score = clf.score(test, y_test)
print(score)
