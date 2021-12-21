from gensim.models.doc2vec import Doc2Vec
from typing import List, Tuple, Any
from nltk import word_tokenize


class Doc2VecVectorizer:
    """This class embeds data with a Doc2Vec model"""

    def __init__(
        self, train_sentences: List, test_sentences: List, modelname: str
    ) -> None:
        self.train_sentences = [word_tokenize(sent) for sent in train_sentences]
        self.test_sentences = [word_tokenize(sent) for sent in test_sentences]
        self.modelname = modelname

    def load_model(self) -> Doc2Vec:
        """load the model

        Returns
        -------
        Doc2Vec
            Doc2vec model
        """
        return Doc2Vec.load(self.modelname)

    def transform_data(self):
        """transforms text inputs to vectors

        Returns
        -------
        list
            list with embedded sentences
        """
        model = self.load_model()
        if len(self.test_sentences) > 0:
            train_vecs = [model.infer_vector(sent) for sent in self.train_sentences]
            test_vecs = [model.infer_vector(sent) for sent in self.test_sentences]
            return train_vecs, test_vecs
        if len(self.test_sentences) == 0:
            train_vecs = [model.infer_vector(sent) for sent in self.train_sentences]
            return train_vecs
