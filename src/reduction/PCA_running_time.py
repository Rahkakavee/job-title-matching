# import
from src.logger import logger
import json
from sklearn.model_selection import train_test_split
from src.vectorizer.countVectorizer.countvectorizer import CountVectorizer_
from src.vectorizer.TFIDF.tfidf import TFIDF
from src.vectorizer.word2vec.word2vec_vectorizer import Word2VecVectorizer
from src.vectorizer.Doc2vec.Doc2vec_vectorizer import Doc2VecVectorizer

# from src.vectorizer.BERT.bert_vectorizer import BertVectorizer
from src.reduction.PCA import dimension_reduction
from src.modeling.LR.lr_classifier import LRClassifier
from src.modeling.SVM.svm import SVMClassifier
from src.modeling.RF.randomforest import RFClassifier
import pandas as pd
import random
import time
import seaborn as sns

data_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
time_report = {"datsize": [], "running_time": [], "method": []}

for data_size in data_sizes:
    # LOAD TRAINING DATA
    logger.debug("LOAD TRAINING DATA")
    with open(file="data/processed/training_data_long_l1.json") as fp:
        training_data_long = json.load(fp=fp)

    training_data_short = random.sample(training_data_long, data_size)

    sentences_short = [job["title"] for job in training_data_short]
    labels_short = [job["id"] for job in training_data_short]

    train_sentences, test_sentences, y_train, y_test = train_test_split(
        sentences_short, labels_short
    )

    logger.debug("Word2Vec")
    word2vec_vectorizer_without = Word2VecVectorizer(
        train_sentences=train_sentences,
        test_sentences=test_sentences,
        y_train=y_train,
        y_test=y_test,
        modelname="src/vectorizer/word2vec/word2vec_without_prior",
    )
    (
        train_vecs_word2vec_without,
        test_vecs_word2vec_without,
        y_word2vec_without_train,
        y_word2vec_without_test,
    ) = word2vec_vectorizer_without.transform_data()

    logger.debug("Word2Vec")
    pca_word2vec_without = dimension_reduction(
        train_vecs=train_vecs_word2vec_without,
        test_vecs=test_vecs_word2vec_without,
        components=0.95,
    )
    pca_word2vec_without.fit_model()
    pca_word2vec_without.evalute_reduction()
    (
        train_word2vec_without,
        test_word2vec_without,
    ) = pca_word2vec_without.transform_data()

    start = time.time()
    count_clf_SVM = SVMClassifier(
        train=train_vecs_word2vec_without,
        test=test_vecs_word2vec_without,
        y_train=y_word2vec_without_train,
        y_test=y_word2vec_without_test,
    )
    count_clf_SVM.fit_classifier()
    end = time.time()
    print(end - start)
    time_report["datsize"].append(data_size)
    time_report["method"].append("SVM without PCA")
    time_report["running_time"].append(round(end - start, 2))

    start = time.time()
    count_clf_SVM = SVMClassifier(
        train=train_word2vec_without,
        test=test_word2vec_without,
        y_train=y_word2vec_without_train,
        y_test=y_word2vec_without_test,
    )
    count_clf_SVM.fit_classifier()
    end = time.time()
    print(end - start)
    time_report["datsize"].append(data_size)
    time_report["method"].append("SVM with PCA")
    time_report["running_time"].append(round(end - start, 2))

    start = time.time()
    word2vec_without_clf_LR = LRClassifier(
        train=train_vecs_word2vec_without,
        test=test_vecs_word2vec_without,
        y_train=y_word2vec_without_train,
        y_test=y_word2vec_without_test,
    )
    word2vec_without_clf_LR.fit_classifier()
    end = time.time()
    print(end - start)
    time_report["datsize"].append(data_size)
    time_report["method"].append("LR without PCA")
    time_report["running_time"].append(round(end - start, 2))

    start = time.time()
    word2vec_without_clf_LR = LRClassifier(
        train=train_word2vec_without,
        test=test_word2vec_without,
        y_train=y_word2vec_without_train,
        y_test=y_word2vec_without_test,
    )
    word2vec_without_clf_LR.fit_classifier()
    end = time.time()
    print(end - start)
    time_report["datsize"].append(data_size)
    time_report["method"].append("LR with PCA")
    time_report["running_time"].append(round(end - start, 2))

    start = time.time()
    word2vec_without_clf_RF = RFClassifier(
        train=train_vecs_word2vec_without,
        test=test_vecs_word2vec_without,
        y_train=y_word2vec_without_train,
        y_test=y_word2vec_without_test,
    )
    word2vec_without_clf_RF.fit_classifier()
    end = time.time()
    print(end - start)
    time_report["datsize"].append(data_size)
    time_report["method"].append("RF without PCA")
    time_report["running_time"].append(round(end - start, 2))

    start = time.time()
    word2vec_without_clf_RF = RFClassifier(
        train=train_word2vec_without,
        test=test_word2vec_without,
        y_train=y_word2vec_without_train,
        y_test=y_word2vec_without_test,
    )
    word2vec_without_clf_RF.fit_classifier()
    end = time.time()
    print(end - start)
    time_report["datsize"].append(data_size)
    time_report["method"].append("RF with PCA")
    time_report["running_time"].append(round(end - start, 2))


df = pd.DataFrame(time_report)


palette_custom = {
    "SVM without PCA": "darkblue",
    "SVM with PCA": "royalblue",
    "LR without PCA": "darkred",
    "LR with PCA": "indianred",
    "RF without PCA": "darkgreen",
    "RF with PCA": "lime",
}

plt = sns.lineplot(
    data=df, x="datsize", y="running_time", hue="method", palette=palette_custom
)
plt.set(ylabel="running time in sec")
fig = plt.get_figure()
fig.savefig("visualization/running_time_PCA.jpeg")
