# import
from src.logger import logger
import json
from sklearn.model_selection import train_test_split
from src.vectorizer.countVectorizer.countvectorizer import CountVectorizer_
from src.vectorizer.TFIDF.tfidf import TFIDF
from src.vectorizer.word2vec.word2vec_vectorizer import Word2VecVectorizer
from src.vectorizer.Doc2vec.Doc2vec_vectorizer import Doc2VecVectorizer
from src.vectorizer.BERT.bert_vectorizer import BertVectorizer
from src.reduction.PCA import dimension_reduction
from src.modeling.LR.lr_classifier import LRClassifier
from src.modeling.SVM.svm import SVMClassifier
from src.modeling.RF.randomforest import RFClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def savefig(ax, path):
    fig = ax.get_figure()
    fig.savefig(path)


def plot_cm(cm):
    f, ax = plt.subplots(figsize=(11, 9))
    ax = sns.heatmap(data=cm, yticklabels=False, xticklabels=False, cmap="gray_r")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.axhline(y=0, color="k", linewidth=2)
    ax.axhline(y=cm.shape[1], color="k", linewidth=2)
    ax.axvline(x=0, color="k", linewidth=2)
    ax.axvline(x=cm.shape[0], color="k", linewidth=2)
    return ax


# LOAD TRAINING DATA
logger.debug("LOAD TRAINING DATA")
with open(file="data/processed/training_data_short_l3.json") as fp:
    training_data_short = json.load(fp=fp)

sentences_short = [job["title"] for job in training_data_short]
labels_short = [job["id"] for job in training_data_short]

train_sentences, test_sentences, y_train, y_test = train_test_split(
    sentences_short, labels_short, random_state=0
)

# TRANSFORM I - VECTORIZING
logger.debug("TRANSFORM I - VECTORIZING")
# Count
logger.debug("CountVectorizer")
countvectorizer = CountVectorizer_(
    train_sentences=train_sentences, test_sentences=test_sentences
)
train_vecs_count, test_vecs_count = countvectorizer.transform_data()

# TFIDF
logger.debug("TFIDF")
tfidfvectorizer = TFIDF(train_sentences=train_sentences, test_sentences=test_sentences)
train_vecs_tfidf, test_vecs_tfidf = tfidfvectorizer.transform_data()

# Word2vec
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

word2vec_vectorizer_with = Word2VecVectorizer(
    train_sentences=train_sentences,
    test_sentences=test_sentences,
    y_train=y_train,
    y_test=y_test,
    modelname="src/vectorizer/word2vec/word2vec_with_prior",
)
(
    train_vecs_word2vec_with,
    test_vecs_word2vec_with,
    y_word2vec_with_train,
    y_word2vec_with_test,
) = word2vec_vectorizer_with.transform_data()

# Doc2vec
logger.debug("Doc2Vec")
doc2vec_vectorizer_without = Doc2VecVectorizer(
    train_sentences=train_sentences,
    test_sentences=test_sentences,
    modelname="src/vectorizer/Doc2vec/doc2_vec_without_prior",
)
(
    train_vecs_doc2vec_without,
    test_vecs_do2vec_without,
) = doc2vec_vectorizer_without.transform_data()

doc2vec_vectorizer_with = Doc2VecVectorizer(
    train_sentences=train_sentences,
    test_sentences=test_sentences,
    modelname="src/vectorizer/Doc2vec/doc2_vec_with_prior",
)
(
    train_vecs_doc2vec_with,
    test_vecs_do2vec_with,
) = doc2vec_vectorizer_with.transform_data()

# BERT
logger.debug("BERT")
bert_vectorizer = BertVectorizer(
    train_sentences=train_sentences,
    test_sentences=test_sentences,
    modelname="src/vectorizer/BERT/bert_fine_tuning_level3",
)

train_vecs_bert, test_vecs_bert = bert_vectorizer.transform_data()

# # TRANSFORM II - REDUCTION"
logger.debug("TRANSFORM II - REDUCTION")
# countvectorizer
logger.debug("CountVectorizer")
pca_count = dimension_reduction(
    train_vecs=train_vecs_count, test_vecs=test_vecs_count, components=0.95
)
pca_count.fit_model()
pca_count.evalute_reduction()
train_count, test_count = pca_count.transform_data()

# TFIDF
logger.debug("TFIDF")
pca_tfidvectoizer = dimension_reduction(
    train_vecs=train_vecs_tfidf, test_vecs=test_vecs_tfidf, components=0.95
)
pca_tfidvectoizer.fit_model()
pca_tfidvectoizer.evalute_reduction()
train_tfidf, test_tfidf = pca_tfidvectoizer.transform_data()


# Word2vec
logger.debug("Word2Vec")
pca_word2vec_without = dimension_reduction(
    train_vecs=train_vecs_word2vec_without,
    test_vecs=test_vecs_word2vec_without,
    components=0.95,
)
pca_word2vec_without.fit_model()
pca_word2vec_without.evalute_reduction()
train_word2vec_without, test_word2vec_without = pca_word2vec_without.transform_data()

pca_word2vec_with = dimension_reduction(
    train_vecs=train_vecs_word2vec_with,
    test_vecs=test_vecs_word2vec_with,
    components=0.95,
)
pca_word2vec_with.fit_model()
pca_word2vec_with.evalute_reduction()
train_word2vec_with, test_word2vec_with = pca_word2vec_with.transform_data()


# Doc2vec
logger.debug("Doc2Vec")
pca_doc2vec_without = dimension_reduction(
    train_vecs=train_vecs_doc2vec_without,
    test_vecs=test_vecs_do2vec_without,
    components=0.95,
)
pca_doc2vec_without.fit_model()
pca_doc2vec_without.evalute_reduction()
train_doc2vec_without, test_doc2vec_without = pca_doc2vec_without.transform_data()

pca_doc2vec_with = dimension_reduction(
    train_vecs=train_vecs_doc2vec_with, test_vecs=test_vecs_do2vec_with, components=0.95
)
pca_doc2vec_with.fit_model()
pca_doc2vec_with.evalute_reduction()
train_doc2vec_with, test_doc2vec_with = pca_doc2vec_with.transform_data()

# # BERT
logger.debug("BERT")
pca_bert = dimension_reduction(
    train_vecs=train_vecs_bert, test_vecs=test_vecs_bert, components=0.95
)
pca_bert.fit_model()
pca_bert.evalute_reduction()
train_bert, test_bert = pca_bert.transform_data()


# TRAIN CLASSIFIERS
logger.debug("TRAIN CLASSIFIERS")
# LR
logger.debug("LR")

# CountVectorizer
logger.debug("CountVectorizer")
count_clf_LR = LRClassifier(
    train=train_count, test=test_count, y_train=y_train, y_test=y_test
)
count_clf_LR.fit_classifier()

# TFIDF
logger.debug("TFIDF")
tfidf_clf_LR = LRClassifier(
    train=train_tfidf, test=test_tfidf, y_train=y_train, y_test=y_test
)
tfidf_clf_LR.fit_classifier()

# Word2vec
logger.debug("Word2Vec")
word2vec_without_clf_LR = LRClassifier(
    train=train_word2vec_without,
    test=test_word2vec_without,
    y_train=y_word2vec_without_train,
    y_test=y_word2vec_without_test,
)
word2vec_without_clf_LR.fit_classifier()

word2vec_with_clf_LR = LRClassifier(
    train=train_word2vec_with,
    test=test_word2vec_with,
    y_train=y_word2vec_with_train,
    y_test=y_word2vec_with_test,
)
word2vec_with_clf_LR.fit_classifier()

# Doc2vec
logger.debug("Doc2Vec")
doc2vec_without_clf_LR = LRClassifier(
    train=train_doc2vec_without,
    test=test_doc2vec_without,
    y_train=y_train,
    y_test=y_test,
)
doc2vec_without_clf_LR.fit_classifier()

doc2vec_with_clf_LR = LRClassifier(
    train=train_doc2vec_with, test=test_doc2vec_with, y_train=y_train, y_test=y_test
)
doc2vec_with_clf_LR.fit_classifier()

# BERT
logger.debug("BERT")
bert_clf_LR = LRClassifier(
    train=train_bert, test=test_bert, y_train=y_train, y_test=y_test
)

bert_clf_LR.fit_classifier()

# SVM
logger.debug("SVM")
# CountVectorizer
logger.debug("CountVectorizer")
count_clf_SVM = SVMClassifier(
    train=train_count, test=test_count, y_train=y_train, y_test=y_test
)
count_clf_SVM.fit_classifier()

# TFIDF
logger.debug("TFIDF")
tfidf_clf_SVM = SVMClassifier(
    train=train_tfidf, test=test_tfidf, y_train=y_train, y_test=y_test
)
tfidf_clf_SVM.fit_classifier()

# Word2vec
logger.debug("Word2Vec")
word2vec_without_clf_SVM = SVMClassifier(
    train=train_word2vec_without,
    test=test_word2vec_without,
    y_train=y_word2vec_without_train,
    y_test=y_word2vec_without_test,
)
word2vec_without_clf_SVM.fit_classifier()

word2vec_with_clf_SVM = SVMClassifier(
    train=train_word2vec_with,
    test=test_word2vec_with,
    y_train=y_word2vec_with_train,
    y_test=y_word2vec_with_test,
)
word2vec_with_clf_SVM.fit_classifier()

# Doc2vec
logger.debug("Doc2Vec")
doc2vec_without_clf_SVM = SVMClassifier(
    train=train_doc2vec_without,
    test=test_doc2vec_without,
    y_train=y_train,
    y_test=y_test,
)
doc2vec_without_clf_SVM.fit_classifier()


doc2vec_with_clf_SVM = SVMClassifier(
    train=train_doc2vec_with, test=test_doc2vec_with, y_train=y_train, y_test=y_test
)
doc2vec_with_clf_SVM.fit_classifier()

# BERT
logger.debug("BERT")
bert_clf_SVM = SVMClassifier(
    train=train_bert, test=test_bert, y_train=y_train, y_test=y_test
)

bert_clf_SVM.fit_classifier()

# RF
logger.debug("RF")
# CountVectorizer
logger.debug("CountVectorizer")
count_clf_RF = RFClassifier(
    train=train_count, test=test_count, y_train=y_train, y_test=y_test
)
count_clf_RF.fit_classifier()

# TFIDF
logger.debug("TFIDF")
tfidf_clf_RF = RFClassifier(
    train=train_tfidf, test=test_tfidf, y_train=y_train, y_test=y_test
)
tfidf_clf_RF.fit_classifier()

# Word2vec
logger.debug("Word2Vec")
word2vec_without_clf_RF = RFClassifier(
    train=train_word2vec_without,
    test=test_word2vec_without,
    y_train=y_word2vec_without_train,
    y_test=y_word2vec_without_test,
)
word2vec_without_clf_RF.fit_classifier()


word2vec_with_clf_RF = RFClassifier(
    train=train_word2vec_with,
    test=test_word2vec_with,
    y_train=y_word2vec_with_train,
    y_test=y_word2vec_with_test,
)
word2vec_with_clf_RF.fit_classifier()

# Doc2vec
logger.debug("Doc2Vec")
doc2vec_without_clf_RF = RFClassifier(
    train=train_doc2vec_without,
    test=test_doc2vec_without,
    y_train=y_train,
    y_test=y_test,
)
doc2vec_without_clf_RF.fit_classifier()


doc2vec_with_clf_RF = RFClassifier(
    train=train_doc2vec_with, test=test_doc2vec_with, y_train=y_train, y_test=y_test
)
doc2vec_with_clf_RF.fit_classifier()

# BERT
logger.debug("BERT")
bert_clf_RF = RFClassifier(
    train=train_bert, test=test_bert, y_train=y_train, y_test=y_test
)

bert_clf_RF.fit_classifier()

####Step 1: get predictions for a classifiers####
predictions_count_LR = count_clf_LR.clf.predict(test_count)
predictions_count_RF = count_clf_RF.clf.predict(test_count)
predictions_count_SVM = count_clf_SVM.clf.predict(test_count)

predictions_tfidf_LR = tfidf_clf_LR.clf.predict(test_tfidf)
predictions_tfidf_RF = tfidf_clf_RF.clf.predict(test_tfidf)
predictions_tfidf_SVM = tfidf_clf_SVM.clf.predict(test_tfidf)

predictions_word2vec_without_LR = word2vec_without_clf_LR.clf.predict(
    test_word2vec_without
)
predictions_word2vec_without_SVM = word2vec_without_clf_SVM.clf.predict(
    test_word2vec_without
)
predictions_word2vec_without_RF = word2vec_without_clf_RF.clf.predict(
    test_word2vec_without
)

predictions_word2vec_with_LR = word2vec_with_clf_LR.clf.predict(test_word2vec_with)
predictions_word2vec_with_RF = word2vec_with_clf_RF.clf.predict(test_word2vec_with)
predictions_word2vec_with_SVM = word2vec_with_clf_SVM.clf.predict(test_word2vec_with)

predictions_doc2vec_without_LR = doc2vec_without_clf_LR.clf.predict(
    test_doc2vec_without
)
predictions_doc2vec_without_RF = doc2vec_without_clf_RF.clf.predict(
    test_doc2vec_without
)
predictions_doc2vec_without_SVM = doc2vec_without_clf_SVM.clf.predict(
    test_doc2vec_without
)

predictions_doc2vec_with_LR = doc2vec_with_clf_LR.clf.predict(test_doc2vec_with)
predictions_doc2vec_with_SVM = doc2vec_with_clf_SVM.clf.predict(test_doc2vec_with)
predictions_doc2vec_with_RF = doc2vec_with_clf_RF.clf.predict(test_doc2vec_with)

predictions_bert_LR = bert_clf_LR.clf.predict(test_bert)
predictions_bert_RF = bert_clf_RF.clf.predict(test_bert)
predictions_bert_SVM = bert_clf_SVM.clf.predict(test_bert)

# Analysis II - covariance matrix
# LR
ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_count_LR))
savefig(ax=ax, path="visualization/results/level3/cm_count_LR.jpg")

ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_tfidf_LR))
savefig(ax=ax, path="visualization/results/level3/cm_tfidf_LR.jpg")

ax = plot_cm(
    confusion_matrix(
        y_true=y_word2vec_without_test, y_pred=predictions_word2vec_without_LR
    )
)
savefig(ax=ax, path="visualization/results/level3/cm_word2vec_without_LR.jpg")


ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_doc2vec_without_LR))
savefig(ax=ax, path="visualization/results/level3/cm_doc2vec_without_LR.jpg")

ax = plot_cm(
    confusion_matrix(y_true=y_word2vec_with_test, y_pred=predictions_word2vec_with_LR)
)
savefig(ax=ax, path="visualization/results/level3/cm_word2vec_with_LR.jpg")


ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_doc2vec_with_LR))
savefig(ax=ax, path="visualization/results/level3/cm_doc2vec_with_LR.jpg")


ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_bert_LR))
savefig(ax=ax, path="visualization/results/level3/cm_bert_LR.jpg")


# RF
ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_count_RF))
savefig(ax=ax, path="visualization/results/level3/cm_count_RF.jpg")

ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_tfidf_RF))
savefig(ax=ax, path="visualization/results/level3/cm_tfidf_RF.jpg")

ax = plot_cm(
    confusion_matrix(
        y_true=y_word2vec_without_test, y_pred=predictions_word2vec_without_RF
    )
)
savefig(ax=ax, path="visualization/results/level3/cm_word2vec_without_RF.jpg")

ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_doc2vec_without_RF))
savefig(ax=ax, path="visualization/results/level3/cm_doc2vec_without_RF.jpg")

ax = plot_cm(
    confusion_matrix(y_true=y_word2vec_with_test, y_pred=predictions_word2vec_with_RF)
)
savefig(ax=ax, path="visualization/results/level3/cm_word2vec_with_RF.jpg")

ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_doc2vec_with_RF))
savefig(ax=ax, path="visualization/results/level3/cm_doc2vec_with_RF.jpg")


ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_bert_RF))
savefig(ax=ax, path="visualization/results/level3/cm_bert_RF.jpg")

# SVM
ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_count_SVM))
savefig(ax=ax, path="visualization/results/level3/cm_count_SVM.jpg")

ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_tfidf_SVM))
savefig(ax=ax, path="visualization/results/level3/cm_tfidf_SVM.jpg")

ax = plot_cm(
    confusion_matrix(
        y_true=y_word2vec_without_test, y_pred=predictions_word2vec_without_SVM
    )
)
savefig(ax=ax, path="visualization/results/level3/cm_word2vec_without_SVM.jpg")

ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_doc2vec_without_SVM))
savefig(ax=ax, path="visualization/results/level3/cm_doc2vec_without_SVM.jpg")

ax = plot_cm(
    confusion_matrix(y_true=y_word2vec_with_test, y_pred=predictions_word2vec_with_SVM)
)
savefig(ax=ax, path="visualization/results/level3/cm_word2vec_with_SVM.jpg")

ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_doc2vec_with_SVM))
savefig(ax=ax, path="visualization/results/level3/cm_doc2vec_with_SVM.jpg")

ax = plot_cm(confusion_matrix(y_true=y_test, y_pred=predictions_bert_SVM))
savefig(ax=ax, path="visualization/results/level3/cm_bert_SVM.jpg")
