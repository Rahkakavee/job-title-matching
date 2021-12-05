from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF:
    def __init__(self, train_sentences, test_sentences) -> None:
        self.train_sentences = train_sentences
        self.test_sentences = test_sentences
        self.vectorizer = TfidfVectorizer()

    def fit_vectorizer(self):
        self.vectorizer.fit(self.train_sentences)

    def transform_data(self):
        self.fit_vectorizer()
        train = self.vectorizer.transform(self.train_sentences).toarray()
        test = self.vectorizer.transform(self.test_sentences).toarray()
        return train, test
