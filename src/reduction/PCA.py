from sklearn.decomposition import PCA
from src.logger import logger


class dimension_reduction:
    def __init__(self, train_vecs, test_vecs, components: float) -> None:
        self.model = PCA(n_components=components)
        self.variance_ratio = 0
        self.train_vecs = train_vecs
        self.test_vecs = test_vecs

    def fit_model(self):
        self.model.fit(self.train_vecs)

    def evalute_reduction(self):
        self.variance_ratio = sum(self.model.explained_variance_ratio_)
        logger.debug(f"Explained variance ratio: {self.variance_ratio}")

    def transform_data(self):
        train = self.model.transform(self.train_vecs)
        test = self.model.transform(self.test_vecs)
        return train, test
