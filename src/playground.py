import pickle
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import json
import gensim
from sklearn.model_selection import train_test_split

with open(file="data/processed/training_data_long.json") as fp:
    training_data_long = json.load(fp=fp)

with open(file="data/processed/training_data_short.json") as fp:
    training_data_short = json.load(fp=fp)


class Sequencer:
    def __init__(self, all_words, max_words, seq_len, embedding_matrix):

        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        """
        temp_vocab = Vocab which has all the unique words
        self.vocab = Our last vocab which has only most used N words.
    
        """
        temp_vocab = list(set(all_words))
        self.vocab = []
        self.word_cnts = {}
        """
        Now we'll create a hash map (dict) which includes words and their occurencies
        """
        for word in tqdm(temp_vocab):
            # 0 does not have a meaning, you can add the word to the list
            # or something different.
            count = len([0 for w in all_words if w == word])
            self.word_cnts[word] = count
            counts = list(self.word_cnts.values())
            indexes = list(range(len(counts)))

        # Now we'll sort counts and while sorting them also will sort indexes.
        # We'll use those indexes to find most used N word.
        cnt = 0
        while cnt + 1 != len(counts):
            cnt = 0
            for i in range(len(counts) - 1):
                if counts[i] < counts[i + 1]:
                    counts[i + 1], counts[i] = counts[i], counts[i + 1]
                    indexes[i], indexes[i + 1] = indexes[i + 1], indexes[i]
                else:
                    cnt += 1

        for ind in indexes[:max_words]:
            self.vocab.append(temp_vocab[ind])

    def textToVector(self, text):
        # First we need to split the text into its tokens and learn the length
        # If length is shorter than the max len we'll add some spaces (100D vectors which has only zero values)
        # If it's longer than the max len we'll trim from the end.
        tokens = text.split()
        len_v = len(tokens) - 1 if len(tokens) < self.seq_len else self.seq_len - 1
        vec = []
        for tok in tokens[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception as E:
                pass

        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
            vec.append(
                np.zeros(
                    100,
                )
            )

        return np.asarray(vec, dtype=object).flatten()


sentences_short = [job["title"] for job in training_data_short]
labels_short = [job["id"] for job in training_data_short]
sentences_long = [job["title"] for job in training_data_long]
labels_long = [job["id"] for job in training_data_long]

sentences_short_tokens = [word_tokenize(sentence) for sentence in sentences_short]
sentences_long_tokens = [word_tokenize(sentence) for sentence in sentences_long]

custom_model = gensim.models.Word2Vec(sentences_long_tokens, vector_size=300)

with open("src/modeling/vectorizer/word2vec/sequencer_all_sentences.pkl", "rb") as fp:
    sequencer = pickle.load(fp)

train, test, y_train, y_test = train_test_split(sentences_short_tokens, labels_short)

train_vecs = np.asarray([sequencer.textToVector(" ".join(seq)) for seq in train])
test_vecs = np.asarray([sequencer.textToVector(" ".join(seq)) for seq in test])

custom_model.wv.vectors_lockf = np.ones(len(custom_model.wv))

custom_model.wv.intersect_word2vec_format(
    "model/GoogleNews-vectors-negative300.bin.gz", binary=True, lockf=0.0
)

custom_model.train(sentences_long_tokens, total_examples=3, epochs=100)

# clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")
# clf.fit(train, y_train)

# metrics.classification_report(y_test, clf.predict(test), output_dict=True)


# from sentence_transformers import SentenceTransformer

# sbert_model = SentenceTransformer("bert-base-nli-mean-tokens")


# sentence_embeddings = sbert_model.encode(sentences)

# print(sentence_embeddings[0])
