# import
from src.logger import logger
import json
from sklearn.model_selection import train_test_split
from src.vectorizer.countVectorizer.countvectorizer import CountVectorizer_
from src.vectorizer.TFIDF.tfidf import TFIDF
from src.reduction.PCA import dimension_reduction
from src.modeling.LR.lr_classifier import LRClassifier
from src.modeling.SVM.svm import SVMClassifier
import pandas as pd

# LOAD TRAINING DATA
logger.debug("LOAD TRAINING DATA")
with open(file="data/processed/training_data_long.json") as fp:
    training_data_long = json.load(fp=fp)

with open(file="data/processed/training_data_short.json") as fp:
    training_data_short = json.load(fp=fp)

sentences_short = [job["title"] for job in training_data_short]
labels_short = [job["id"] for job in training_data_short]
sentences_long = [job["title"] for job in training_data_long]
labels_long = [job["id"] for job in training_data_long]

train_sentences, test_sentences, y_train, y_test = train_test_split(
    sentences_short, labels_short
)

# TRANSFORM I - VECTORIZING
logger.debug("TRANSFORM I - VECTORIZING")
countvectorizer = CountVectorizer_(
    train_sentences=train_sentences, test_sentences=test_sentences
)
train_vecs_count, test_vecs_count = countvectorizer.transform_data()

tfidfvectorizer = TFIDF(train_sentences=train_sentences, test_sentences=test_sentences)
train_vecs_tfidf, test_vecs_tfidf = tfidfvectorizer.transform_data()

# TRANSFORM II - REDUCTION"
logger.debug("TRANSFORM II - REDUCTION")
pca_count = dimension_reduction(
    train_vecs=train_vecs_count, test_vecs=test_vecs_count, components=0.95
)

pca_count.fit_model()
pca_count.evalute_reduction()
train_count, test_count = pca_count.transform_data()

pca_tfidvectoizer = dimension_reduction(
    train_vecs=train_vecs_tfidf, test_vecs=test_vecs_tfidf, components=0.95
)

pca_tfidvectoizer.fit_model()
pca_tfidvectoizer.evalute_reduction()
train_tfidf, test_tfidf = pca_tfidvectoizer.transform_data()

# TRAIN CLASSIFIERS
logger.debug("TRAIN CLASSIFIERS")
# LR
logger.debug("LR")

# CountVectorizer
count_clf_LR = LRClassifier(
    train=train_count, test=test_count, y_train=y_train, y_test=y_test
)
count_clf_LR.fit_classifier()

# TFIDF
tfidf_clf_LR = LRClassifier(
    train=train_tfidf, test=test_tfidf, y_train=y_train, y_test=y_test
)
tfidf_clf_LR.fit_classifier()

# SVM
logger.debug("SVM")
# CountVectorizer
count_clf_SVM = SVMClassifier(
    train=train_count, test=test_count, y_train=y_train, y_test=y_test
)
count_clf_SVM.fit_classifier()

# TFIDF
tfidf_clf_SVM = SVMClassifier(
    train=train_tfidf, test=test_tfidf, y_train=y_train, y_test=y_test
)
tfidf_clf_SVM.fit_classifier()

# EVALUATE CLASSIFIERS
logger.debug("EVALUATE CLASSIFIERS")
logger.debug("LR")
count_clf_LR_report = count_clf_LR.evaluate(output_dict=True)
tfidf_clf_LR_report = tfidf_clf_LR.evaluate(output_dict=True)

logger.debug("SVM")
count_clf_SVM_report = count_clf_SVM.evaluate(output_dict=True)
tfidf_clf_SVM_report = tfidf_clf_SVM.evaluate(output_dict=True)

# TO_LATEX
logger.debug("LATEX")
logger.debug("Accuracy")
df_accuracy = pd.DataFrame(
    {
        "LR": [
            round(count_clf_LR_report["accuracy"], 2),
            round(tfidf_clf_LR_report["accuracy"], 2),
        ],
        "SVM": [
            round(count_clf_SVM_report["accuracy"], 2),
            round(tfidf_clf_SVM_report["accuracy"], 2),
        ],
    },
    index=["CountVectorizer", "TFIDF"],
)

print(df_accuracy.to_latex())

logger.debug("Precision, Recall, F1-score")
df_prf = pd.DataFrame(
    {
        "LR": [
            f"p: {round(count_clf_LR_report['macro avg']['precision'], 2)}, r: {round(count_clf_LR_report['macro avg']['recall'], 2)}, F1: {round(count_clf_LR_report['macro avg']['f1-score'], 2)}",
            f"p: {round(tfidf_clf_LR_report['macro avg']['precision'], 2)}, r: {round(tfidf_clf_LR_report['macro avg']['recall'], 2)}, F1: {round(tfidf_clf_LR_report['macro avg']['f1-score'], 2)}",
        ],
        "SVM": [
            f"p: {round(count_clf_SVM_report['macro avg']['precision'], 2)}, r: {round(count_clf_SVM_report['macro avg']['recall'], 2)}, F1: {round(count_clf_SVM_report['macro avg']['f1-score'], 2)}",
            f"p: {round(tfidf_clf_SVM_report['macro avg']['precision'], 2)}, r: {round(tfidf_clf_SVM_report['macro avg']['recall'], 2)}, F1: {round(tfidf_clf_SVM_report['macro avg']['f1-score'], 2)}",
        ],
    },
    index=["CountVectorizer_", "TFIDF"],
)

print(df_prf.to_latex())
