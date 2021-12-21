from gensim.models.doc2vec import Word2Vec
from typing import List, Tuple, Any
from nltk import word_tokenize
from src.vectorizer.word2vec.average_word_2_vec import AverageWord2Vec


class Word2VecVectorizer:
    """This class embeds data with a Doc2Vec model"""

    def __init__(
        self,
        train_sentences: List,
        test_sentences: List,
        y_train: List,
        y_test: List,
        modelname: str,
    ) -> None:
        self.train_sentences = train_sentences
        self.test_sentences = test_sentences
        self.y_train = y_train
        self.y_test = y_test
        self.modelname = modelname

    def load_model(self) -> Word2Vec:
        """load the model

        Returns
        -------
        Doc2Vec
            Doc2vec model
        """
        return Word2Vec.load(self.modelname)

    def transform_data(self):
        """transforms text inputs to vectors

        Returns
        -------
        list
            list with embedded sentences
        """
        model = self.load_model()
        avgr = AverageWord2Vec(model.wv)

        if len(self.test_sentences) > 0:
            train_vecs = avgr.vectorize_all(self.train_sentences)
            test_vecs = avgr.vectorize_all(self.test_sentences)
            full_training_vecs_train = []
            full_labels_train = []
            for i in range(0, len(train_vecs)):
                if train_vecs[i].shape == (300,):
                    full_training_vecs_train.append(train_vecs[i])
                    full_labels_train.append(self.y_train[i])

            full_training_vecs_test = []
            full_labels_test = []
            for i in range(0, len(test_vecs)):
                if test_vecs[i].shape == (300,):
                    full_training_vecs_test.append(test_vecs[i])
                    full_labels_test.append(self.y_test[i])

            return (
                full_training_vecs_train,
                full_training_vecs_test,
                full_labels_train,
                full_labels_test,
            )

        if len(self.test_sentences) == 0:
            train_vecs = avgr.vectorize_all(self.train_sentences)
            full_training_vecs_train = []
            full_labels_train = []
            for i in range(0, len(train_vecs)):
                if train_vecs[i].shape == (300,):
                    full_training_vecs_train.append(train_vecs[i])
                    full_labels_train.append(self.y_train[i])
            return full_training_vecs_train, full_labels_train
