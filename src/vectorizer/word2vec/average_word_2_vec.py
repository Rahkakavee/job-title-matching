import numpy as np


"""This file contains the helper class averageWord2Vec for vectorizing sentences with word2Vec"""


class AverageWord2Vec:
    """Helper class for averaging word embeddings"""

    def __init__(self, w2v_model):
        self.w2v_model = w2v_model

    def vectorize(self, doc: str) -> np.ndarray:
        """average word2vec embeddings and transform sentence to one vec

        Parameters
        ----------
        doc : str
            [description]

        Returns
        -------
        np.ndarray
            [description]
        """
        words = [w for w in doc.split(" ")]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                pass
        vector = np.mean(word_vecs, axis=0)
        return vector

    def vectorize_all(self, sentences):
        """vectorize all sentences

        Parameters
        ----------
        targets : [type]


        Returns
        -------
        [type]
            [description]
        """
        return [self.vectorize(sent) for sent in sentences]
