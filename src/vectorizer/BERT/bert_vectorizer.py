from typing import Tuple, List, Any
from src.preprocessing.preprocessing_functions import *
from sentence_transformers import SentenceTransformer


class BertVectorizer:
    """This class encode sentences according to BERT"""

    def __init__(
        self, train_sentences: List, test_sentences: List, modelname: str
    ) -> None:
        self.train_sentences = train_sentences
        self.test_sentences = test_sentences
        self.modelname = modelname

    def load_model(self) -> SentenceTransformer:
        """load the model

        Returns
        -------
        Doc2Vec
            Doc2vec model
        """
        return SentenceTransformer(self.modelname)

    def transform_data(self) -> Tuple[List[Any], List[Any]]:
        """transforms text inputs to vectors

        Returns
        -------
        list
            list with embedded sentences
        """
        model = self.load_model()
        model.save("src/vectorizer/BERT/roberta")
        train_vecs = model.encode(self.train_sentences)
        test_vecs = model.encode(self.test_sentences)
        return train_vecs, test_vecs
