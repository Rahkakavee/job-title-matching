from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "model/GoogleNews-vectors-negative300.bin.gz", binary=True
)


import json
from nltk import word_tokenize

with open(file="data/processed/training_data_long.json") as fp:
    training_data_short = json.load(fp=fp)

with open(file="data/processed/training_data_short.json.json") as fp:
    training_data = json.load(fp=fp)


sentences = [el["title"].lower() for el in training_data_short]
sentences_short = [el["title"].lower() for el in training_data]
sentences_tokenized = [word_tokenize(sent) for sent in sentences]

from gensim.models import Word2Vec

custom_model = Word2Vec(vector_size=300, min_count=1)
custom_model.build_vocab(sentences_tokenized)
training_example_count = custom_model.corpus_count
custom_model.build_vocab([list(model.key_to_index.keys())], update=True)
import numpy as np

custom_model.wv.vectors_lockf = np.ones(len(custom_model.wv))
custom_model.wv.intersect_word2vec_format(
    "model/GoogleNews-vectors-negative300.bin.gz", binary=True, lockf=1.0
)
custom_model.train(sentences, total_examples=training_example_count, epochs=10)


class DocSim:
    def __init__(self, w2v_model, stopwords=None):
        self.w2v_model = w2v_model
        self.stopwords = stopwords if stopwords is not None else []

    def vectorize(self, doc: str) -> np.ndarray:
        """
        Identify the vector values for each word in the given document
        :param doc:
        :return:
        """
        words = [w for w in doc.split(" ")]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        vector = np.mean(word_vecs, axis=0)
        return vector

    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_target(self, targets):
        return [self.vectorize(target) for target in targets]

    def calculate_similarity(
        self, source_doc, target_docs=None, target_vecs=None, threshold=0
    ):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        # if not target_docs:
        #     return []

        # if isinstance(target_docs, str):
        #     target_docs = [target_docs]

        source_vec = self.vectorize(source_doc)
        results = []
        for i in tqdm(range(0, len(target_docs))):
            target_vec = target_vecs[i]
            sim_score = self._cosine_sim(source_vec, target_vec)
            if sim_score > threshold:
                results.append({"score": sim_score, "doc": target_docs[i]})
            # Sort results by score in desc order
            results.sort(key=lambda k: k["score"], reverse=True)
        return results


ds = DocSim(custom_model.wv)
training_vecs = ds.calculate_target(sentences_short)
labels = [el["id"] for el in training_data]


full_training_vecs = []
full_labels = []
for i in range(0, len(training_vecs)):
    if training_vecs[i].shape == (300,):
        full_training_vecs.append(training_vecs[i])
        full_labels.append(labels[i])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

train, test, y_train, y_test = train_test_split(full_training_vecs, full_labels)
clf = LogisticRegression(max_iter=1000)
clf.fit(train, y_train)
clf.score(test, y_test)
