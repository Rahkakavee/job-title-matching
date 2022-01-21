import pandas
import collections
import seaborn as sns

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
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# LOAD TRAINING DATA
logger.debug("LOAD TRAINING DATA")
with open(file="data/processed/training_data_long_l1.json") as fp:
    training_data_long = json.load(fp=fp)

with open(file="data/processed/training_data_short_l3.json") as fp:
    training_data_short = json.load(fp=fp)

with open(file="data/processed/training_data_medium_l1.json") as fp:
    training_data_medium = json.load(fp=fp)

sentences_short = [job["title"] for job in training_data_short]
labels_short = [job["id"] for job in training_data_short]
sentences_long = [job["title"] for job in training_data_long]
labels_long = [job["id"] for job in training_data_long]
sentences_medium = [job["title"] for job in training_data_medium]
labels_medium = [job["id"] for job in training_data_medium]

train_sentences, test_sentences, y_train, y_test = train_test_split(
    sentences_short, labels_short
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
    modelname="src/vectorizer/BERT/fine_tuned_bert_IV",
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


def predictions_wrong(predictions_, labels_, sentences_list_):
    predictions_list_ = []
    for i in range(0, len(predictions_)):
        if predictions_[i] != labels_[i]:
            predictions_list_.append(
                {
                    "title": sentences_list_[i],
                    "pred": predictions_[i],
                    "label": labels_[i],
                }
            )
    return predictions_list_


def predictions_correct(predictions_, labels_, sentences_list_):
    predictions_list_ = []
    for i in range(0, len(predictions_)):
        if predictions_[i] == labels_[i]:
            predictions_list_.append(
                {
                    "title": sentences_list_[i],
                    "pred": predictions_[i],
                    "label": labels_[i],
                }
            )
    return predictions_list_


def create_predicition_data(
    predictions, predictions_list_, name, format_dict_, divider
):
    if format_dict_:
        for key, value in collections.Counter(
            [el["pred"] for el in predictions]
        ).items():
            predictions_list_.append(
                {
                    "kldb classes": key,
                    "percentage": value / divider,
                    "vectorization method": name,
                }
            )
    else:
        for key, value in collections.Counter(predictions).items():
            predictions_list_.append(
                {
                    "kldb classes": key,
                    "percentage": value / divider,
                    "vectorization method": name,
                }
            )


palette_01 = {
    "count": "skyblue",
    "tfidf": "darkblue",
    "word2vec_I": "green",
    "bert": "red",
    "doc2vec_I": "lightgreen",
}


palette_02 = {
    "word2vec_I": "green",
    "doc2vec_I": "lightgreen",
    "word2vec_II": "purple",
    "doc2vec_II": "plum",
}


def plot_countings(predictions_list_, palette, ylabel):
    df = pandas.DataFrame(predictions_list_)
    df["kldb classes"] = df["kldb classes"].astype("int")
    ax = sns.barplot(
        data=df,
        x="kldb classes",
        y="percentage",
        hue="vectorization method",
        palette=palette,
    )
    ax.set(ylabel=ylabel)
    return ax


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

# Analysis I
# Comparison of vectorization techniques - LR
predictions_counting_LR = []
create_predicition_data(
    predictions_count_LR,
    predictions_counting_LR,
    name="count",
    format_dict_=False,
    divider=len(predictions_count_LR),
)

create_predicition_data(
    predictions_tfidf_LR,
    predictions_counting_LR,
    name="tfidf",
    format_dict_=False,
    divider=len(predictions_tfidf_LR),
)

create_predicition_data(
    predictions_word2vec_without_LR,
    predictions_counting_LR,
    name="word2vec_I",
    format_dict_=False,
    divider=len(predictions_word2vec_without_LR),
)

create_predicition_data(
    predictions_doc2vec_without_LR,
    predictions_counting_LR,
    name="doc2vec_I",
    format_dict_=False,
    divider=len(predictions_doc2vec_without_LR),
)

create_predicition_data(
    predictions_bert_LR,
    predictions_counting_LR,
    name="bert",
    format_dict_=False,
    divider=len(predictions_bert_LR),
)

ax = plot_countings(predictions_counting_LR, palette_01, "frequency (%)")
fig = ax.get_figure()
fig.savefig("visualization/results/level1/predictions_LR.jpg")


# Comparison of vectorization techniques - RF
predictions_counting_RF = []
create_predicition_data(
    predictions_count_RF,
    predictions_counting_RF,
    name="count",
    format_dict_=False,
    divider=len(predictions_count_RF),
)

create_predicition_data(
    predictions_tfidf_RF,
    predictions_counting_RF,
    name="tfidf",
    format_dict_=False,
    divider=len(predictions_tfidf_RF),
)

create_predicition_data(
    predictions_word2vec_without_RF,
    predictions_counting_RF,
    name="word2vec_I",
    format_dict_=False,
    divider=len(predictions_word2vec_without_RF),
)

create_predicition_data(
    predictions_doc2vec_without_RF,
    predictions_counting_RF,
    name="doc2vec_I",
    format_dict_=False,
    divider=len(predictions_doc2vec_without_RF),
)

create_predicition_data(
    predictions_bert_RF,
    predictions_counting_RF,
    name="bert",
    format_dict_=False,
    divider=len(predictions_bert_RF),
)

ax = plot_countings(predictions_counting_RF, palette_01, "frequency (%)")
fig = ax.get_figure()
fig.savefig("visualization/results/level1/predictions_RF.jpg")


# Comparison of additional knowledge - LR
predictions_counting_LR_word_doc = []
create_predicition_data(
    predictions_word2vec_without_LR,
    predictions_counting_LR_word_doc,
    name="word2vec_I",
    format_dict_=False,
    divider=len(predictions_word2vec_without_LR),
)

create_predicition_data(
    predictions_doc2vec_without_LR,
    predictions_counting_LR_word_doc,
    name="doc2vec_I",
    format_dict_=False,
    divider=len(predictions_doc2vec_without_LR),
)

create_predicition_data(
    predictions_word2vec_with_LR,
    predictions_counting_LR_word_doc,
    name="word2vec_II",
    format_dict_=False,
    divider=len(predictions_word2vec_with_LR),
)

create_predicition_data(
    predictions_doc2vec_with_LR,
    predictions_counting_LR_word_doc,
    name="doc2vec_II",
    format_dict_=False,
    divider=len(predictions_doc2vec_with_LR),
)

ax = plot_countings(predictions_counting_LR_word_doc, palette_02, "frequency (%)")
fig = ax.get_figure()
fig.savefig("visualization/results/level1/predictions_LR_word_doc.jpg")

# Comparison of additional knowledge - RF
predictions_counting_RF_word_doc = []
create_predicition_data(
    predictions_word2vec_without_RF,
    predictions_counting_RF_word_doc,
    name="word2vec_I",
    format_dict_=False,
    divider=len(predictions_word2vec_without_RF),
)

create_predicition_data(
    predictions_doc2vec_without_RF,
    predictions_counting_RF_word_doc,
    name="doc2vec_I",
    format_dict_=False,
    divider=len(predictions_doc2vec_without_RF),
)

create_predicition_data(
    predictions_word2vec_with_RF,
    predictions_counting_RF_word_doc,
    name="word2vec_II",
    format_dict_=False,
    divider=len(predictions_word2vec_with_RF),
)

create_predicition_data(
    predictions_doc2vec_with_RF,
    predictions_counting_RF_word_doc,
    name="doc2vec_II",
    format_dict_=False,
    divider=len(predictions_doc2vec_with_RF),
)

ax = plot_countings(predictions_counting_RF_word_doc, palette_02, "frequency (%)")
fig = ax.get_figure()
fig.savefig("visualization/results/level1/predictions_RF_word_doc.jpg")


# Analysis II
# # Comparison of vectorization techniques wrong - LR
true_counting = []
for key, value in collections.Counter(y_test).items():
    true_counting.append({"kldb classes": key, "frequency": value})
df_labels = pandas.DataFrame(true_counting)
df_labels["kldb classes"] = df_labels["kldb classes"].astype(int)

predictions_counting_correct_LR = []
create_predicition_data(
    predictions_correct(predictions_count_LR, y_test, test_sentences),
    predictions_counting_correct_LR,
    name="count",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_tfidf_LR, y_test, test_sentences),
    predictions_counting_correct_LR,
    name="tfidf",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(
        predictions_word2vec_without_LR, y_word2vec_without_test, test_sentences
    ),
    predictions_counting_correct_LR,
    name="word2vec_I",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_doc2vec_without_LR, y_test, test_sentences),
    predictions_counting_correct_LR,
    name="doc2vec_I",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_bert_LR, y_test, test_sentences),
    predictions_counting_correct_LR,
    name="bert",
    format_dict_=True,
    divider=1,
)

s1 = plot_countings(predictions_counting_correct_LR, palette_01, "frequency")
s2 = sns.barplot(
    data=df_labels,
    x="kldb classes",
    y="frequency",
    linewidth=1.5,
    facecolor=(0, 0, 1, 0),
    errcolor=".1",
    edgecolor=".1",
)
fig = s2.get_figure()
fig.savefig("visualization/results/predictions_counting_wrong_LR.jpg")

# # Comparison of vectorization techniques - RF
predictions_counting_correct_RF = []
create_predicition_data(
    predictions_correct(predictions_count_RF, y_test, test_sentences),
    predictions_counting_correct_RF,
    name="count",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_tfidf_RF, y_test, test_sentences),
    predictions_counting_correct_RF,
    name="tfidf",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(
        predictions_word2vec_without_RF, y_word2vec_without_test, test_sentences
    ),
    predictions_counting_correct_RF,
    name="word2vec_I",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_doc2vec_without_RF, y_test, test_sentences),
    predictions_counting_correct_RF,
    name="doc2vec_I",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_bert_RF, y_test, test_sentences),
    predictions_counting_correct_RF,
    name="bert",
    format_dict_=True,
    divider=1,
)

s1 = plot_countings(predictions_counting_correct_RF, palette_01, "frequency")
s2 = sns.barplot(
    data=df_labels,
    x="kldb classes",
    y="frequency",
    linewidth=1.5,
    facecolor=(0, 0, 1, 0),
    errcolor=".1",
    edgecolor=".1",
)
fig = s2.get_figure()
fig.savefig("visualization/results/level1/predictions_correct_relative_RF.jpg")

# Comparison of additional knowledge - LR
predictions_counting_LR_word_doc_correct = []
create_predicition_data(
    predictions_correct(
        predictions_word2vec_without_LR, y_word2vec_without_test, test_sentences
    ),
    predictions_counting_LR_word_doc_correct,
    name="word2vec_I",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_doc2vec_without_LR, y_test, test_sentences),
    predictions_counting_LR_word_doc_correct,
    name="doc2vec_I",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(
        predictions_word2vec_with_LR, y_word2vec_with_test, test_sentences
    ),
    predictions_counting_LR_word_doc_correct,
    name="word2vec_II",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_doc2vec_with_LR, y_test, test_sentences),
    predictions_counting_LR_word_doc_correct,
    name="doc2vec_II",
    format_dict_=True,
    divider=1,
)

s1 = plot_countings(predictions_counting_LR_word_doc_correct, palette_02, "frequency")
s2 = sns.barplot(
    data=df_labels,
    x="kldb classes",
    y="frequency",
    linewidth=1.5,
    facecolor=(0, 0, 1, 0),
    errcolor=".1",
    edgecolor=".1",
)
fig = s2.get_figure()
fig.savefig("visualization/results/level1/predictions_correct_relative_word_doc_LR.jpg")


# Comparison of additional knowledge - RF
predictions_counting_RF_word_doc_correct = []
create_predicition_data(
    predictions_correct(
        predictions_word2vec_without_RF, y_word2vec_without_test, test_sentences
    ),
    predictions_counting_RF_word_doc_correct,
    name="word2vec_I",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_doc2vec_without_RF, y_test, test_sentences),
    predictions_counting_RF_word_doc_correct,
    name="doc2vec_I",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(
        predictions_word2vec_with_RF, y_word2vec_with_test, test_sentences
    ),
    predictions_counting_RF_word_doc_correct,
    name="word2vec_II",
    format_dict_=True,
    divider=1,
)

create_predicition_data(
    predictions_correct(predictions_doc2vec_with_RF, y_test, test_sentences),
    predictions_counting_RF_word_doc_correct,
    name="doc2vec_II",
    format_dict_=True,
    divider=1,
)

s1 = plot_countings(predictions_counting_RF_word_doc_correct, palette_02, "frequency")
s2 = sns.barplot(
    data=df_labels,
    x="kldb classes",
    y="frequency",
    linewidth=1.5,
    facecolor=(0, 0, 1, 0),
    errcolor=".1",
    edgecolor=".1",
)
fig = s2.get_figure()
fig.savefig("visualization/results/level1/predictions_correct_relative_word_doc_RF.jpg")

# Analysis III
# LR
cmap = sns.color_palette("cubehelix_r", as_cmap=True)
ax = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_count_LR, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_count_LR.jpg")

axs = ConfusionMatrixDisplay.from_predictions(
    y_word2vec_without_test,
    predictions_word2vec_without_LR,
    cmap=cmap,
    include_values=False,
)
plt.savefig("visualization/results/level1/cm_word2vec_without_LR.jpg")


axs = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_doc2vec_without_LR, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_doc2vec_without_LR.jpg")


axs = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_bert_LR, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_bert_LR.jpg")


fig = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_doc2vec_with_LR, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_doc2vec_with_LR.jpg")


fig = ConfusionMatrixDisplay.from_predictions(
    y_word2vec_with_test, predictions_word2vec_with_LR, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_word2vec_with_LR.jpg")

# RF
fig = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_count_RF, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_count_RF.jpg")


fig = ConfusionMatrixDisplay.from_predictions(
    y_word2vec_without_test,
    predictions_word2vec_without_RF,
    cmap=cmap,
    include_values=False,
)
plt.savefig("visualization/results/level1/cm_word2vec_without_RF.jpg")

fig = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_doc2vec_without_RF, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_doc2vec_without_RF.jpg")

fig = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_doc2vec_with_RF, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_doc2vec_with_RF.jpg")

fig = ConfusionMatrixDisplay.from_predictions(
    y_word2vec_with_test, predictions_word2vec_with_RF, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_word2vec_with_RF.jpg")

fig = ConfusionMatrixDisplay.from_predictions(
    y_test, predictions_bert_RF, cmap=cmap, include_values=False
)
plt.savefig("visualization/results/level1/cm_bert_RF.jpg")
