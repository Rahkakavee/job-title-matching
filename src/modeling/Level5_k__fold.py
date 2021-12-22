# import
from itertools import count
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
import pandas as pd

# LOAD TRAINING DATA
logger.debug("LOAD TRAINING DATA")
with open(file="data/processed/training_data_long_l3.json") as fp:
    training_data_long = json.load(fp=fp)

with open(file="data/processed/training_data_short_l3.json") as fp:
    training_data_short = json.load(fp=fp)

sentences_short = [job["title"] for job in training_data_short]
labels_short = [job["id"] for job in training_data_short]
sentences_long = [job["title"] for job in training_data_long]
labels_long = [job["id"] for job in training_data_long]


# TRANSFORM I - VECTORIZING
logger.debug("TRANSFORM I - VECTORIZING")
# Count
logger.debug("CountVectorizer")
countvectorizer = CountVectorizer_(train_sentences=sentences_short, test_sentences=[])
train_vecs_count = countvectorizer.transform_data()

# TFIDF
logger.debug("TFIDF")
tfidfvectorizer = TFIDF(train_sentences=sentences_short, test_sentences=[])
tfidfvectorizer.fit_vectorizer()
train_vecs_tfidf = tfidfvectorizer.transform_data()

# Word2vec
logger.debug("Word2Vec")
word2vec_vectorizer_without = Word2VecVectorizer(
    train_sentences=sentences_short,
    test_sentences=[],
    y_train=labels_short,
    y_test=[],
    modelname="src/vectorizer/word2vec/word2vec_without_prior",
)
(
    train_vecs_word2vec_without,
    y_word2vec_without_train,
) = word2vec_vectorizer_without.transform_data()

word2vec_vectorizer_with = Word2VecVectorizer(
    train_sentences=sentences_short,
    test_sentences=[],
    y_train=labels_short,
    y_test=[],
    modelname="src/vectorizer/word2vec/word2vec_with_prior",
)
(
    train_vecs_word2vec_with,
    y_word2vec_with_train,
) = word2vec_vectorizer_with.transform_data()

# Doc2vec
logger.debug("Doc2Vec")
doc2vec_vectorizer_without = Doc2VecVectorizer(
    train_sentences=sentences_short,
    test_sentences=[],
    modelname="src/vectorizer/Doc2vec/doc2_vec_without_prior",
)
train_vecs_doc2vec_without = doc2vec_vectorizer_without.transform_data()

doc2vec_vectorizer_with = Doc2VecVectorizer(
    train_sentences=sentences_short,
    test_sentences=[],
    modelname="src/vectorizer/Doc2vec/doc2_vec_with_prior",
)
train_vecs_doc2vec_with = doc2vec_vectorizer_with.transform_data()

# BERT
logger.debug("BERT")
bert_vectorizer = BertVectorizer(
    train_sentences=sentences_short,
    test_sentences=[],
    modelname="all-distilroberta-v1",
)

train_vecs_bert = bert_vectorizer.transform_data()

# TRANSFORM II - REDUCTION"
logger.debug("TRANSFORM II - REDUCTION")
# countvectorizer
logger.debug("CountVectorizer")
pca_count = dimension_reduction(
    train_vecs=train_vecs_count, test_vecs=[], components=0.95
)
pca_count.fit_model()
pca_count.evalute_reduction()
train_count = pca_count.transform_data()

# TFIDF
logger.debug("TFIDF")
pca_tfidvectoizer = dimension_reduction(
    train_vecs=train_vecs_tfidf, test_vecs=[], components=0.95
)
pca_tfidvectoizer.fit_model()
pca_tfidvectoizer.evalute_reduction()
train_tfidf = pca_tfidvectoizer.transform_data()


# Word2vec
logger.debug("Word2Vec")
pca_word2vec_without = dimension_reduction(
    train_vecs=train_vecs_word2vec_without,
    test_vecs=[],
    components=0.95,
)
pca_word2vec_without.fit_model()
pca_word2vec_without.evalute_reduction()
train_word2vec_without = pca_word2vec_without.transform_data()

pca_word2vec_with = dimension_reduction(
    train_vecs=train_vecs_word2vec_with,
    test_vecs=[],
    components=0.95,
)
pca_word2vec_with.fit_model()
pca_word2vec_with.evalute_reduction()
train_word2vec_with = pca_word2vec_with.transform_data()


# Doc2vec
logger.debug("Doc2Vec")
pca_doc2vec_without = dimension_reduction(
    train_vecs=train_vecs_doc2vec_without,
    test_vecs=[],
    components=0.95,
)
pca_doc2vec_without.fit_model()
pca_doc2vec_without.evalute_reduction()
train_doc2vec_without = pca_doc2vec_without.transform_data()

pca_doc2vec_with = dimension_reduction(
    train_vecs=train_vecs_doc2vec_with, test_vecs=[], components=0.95
)
pca_doc2vec_with.fit_model()
pca_doc2vec_with.evalute_reduction()
train_doc2vec_with = pca_doc2vec_with.transform_data()

# BERT
logger.debug("BERT")
pca_bert = dimension_reduction(
    train_vecs=train_vecs_bert, test_vecs=[], components=0.95
)
pca_bert.fit_model()
pca_bert.evalute_reduction()
train_bert = pca_bert.transform_data()


# TRAIN CLASSIFIERS
logger.debug("TRAIN CLASSIFIERS")
# LR
logger.debug("LR")

# CountVectorizer
logger.debug("CountVectorizer")
count_clf_LR = LRClassifier(train=train_count, test=[], y_train=labels_short, y_test=[])

# TFIDF
logger.debug("TFIDF")
tfidf_clf_LR = LRClassifier(train=train_tfidf, test=[], y_train=labels_short, y_test=[])

# Word2vec
logger.debug("Word2Vec")
word2vec_without_clf_LR = LRClassifier(
    train=train_word2vec_without,
    test=[],
    y_train=y_word2vec_without_train,
    y_test=[],
)

word2vec_with_clf_LR = LRClassifier(
    train=train_word2vec_with,
    test=[],
    y_train=y_word2vec_with_train,
    y_test=[],
)


# Doc2vec
logger.debug("Doc2Vec")
doc2vec_without_clf_LR = LRClassifier(
    train=train_doc2vec_without,
    test=[],
    y_train=labels_short,
    y_test=[],
)


doc2vec_with_clf_LR = LRClassifier(
    train=train_doc2vec_with, test=[], y_train=labels_short, y_test=[]
)


# BERT
logger.debug("BERT")
bert_clf_LR = LRClassifier(train=train_bert, test=[], y_train=labels_short, y_test=[])


# SVM
logger.debug("SVM")
# CountVectorizer
logger.debug("CountVectorizer")
count_clf_SVM = SVMClassifier(
    train=train_count, test=[], y_train=labels_short, y_test=[]
)


# TFIDF
logger.debug("TFIDF")
tfidf_clf_SVM = SVMClassifier(
    train=train_tfidf, test=[], y_train=labels_short, y_test=[]
)


# Word2vec
logger.debug("Word2Vec")
word2vec_without_clf_SVM = SVMClassifier(
    train=train_word2vec_without,
    test=[],
    y_train=y_word2vec_without_train,
    y_test=[],
)


word2vec_with_clf_SVM = SVMClassifier(
    train=train_word2vec_with,
    test=[],
    y_train=y_word2vec_with_train,
    y_test=[],
)


# Doc2vec
logger.debug("Doc2Vec")
doc2vec_without_clf_SVM = SVMClassifier(
    train=train_doc2vec_without,
    test=[],
    y_train=labels_short,
    y_test=[],
)


doc2vec_with_clf_SVM = SVMClassifier(
    train=train_doc2vec_with, test=[], y_train=labels_short, y_test=[]
)


# BERT
logger.debug("BERT")
bert_clf_SVM = SVMClassifier(train=train_bert, test=[], y_train=labels_short, y_test=[])


# RF
logger.debug("RF")
# CountVectorizer
logger.debug("CountVectorizer")
count_clf_RF = RFClassifier(train=train_count, test=[], y_train=labels_short, y_test=[])


# TFIDF
logger.debug("TFIDF")
tfidf_clf_RF = RFClassifier(train=train_tfidf, test=[], y_train=labels_short, y_test=[])


# Word2vec
logger.debug("Word2Vec")
word2vec_without_clf_RF = RFClassifier(
    train=train_word2vec_without,
    test=[],
    y_train=y_word2vec_without_train,
    y_test=[],
)


word2vec_with_clf_RF = RFClassifier(
    train=train_word2vec_with,
    test=[],
    y_train=y_word2vec_with_train,
    y_test=[],
)


# Doc2vec
logger.debug("Doc2Vec")
doc2vec_without_clf_RF = RFClassifier(
    train=train_doc2vec_without,
    test=[],
    y_train=labels_short,
    y_test=[],
)


doc2vec_with_clf_RF = RFClassifier(
    train=train_doc2vec_with, test=[], y_train=labels_short, y_test=[]
)


# BERT
logger.debug("BERT")
bert_clf_RF = RFClassifier(train=train_bert, test=[], y_train=labels_short, y_test=[])


# EVALUATE CLASSIFIERS
logger.debug("EVALUATE CLASSIFIERS")
logger.debug("LR")
count_clf_LR_report = count_clf_LR.cross_validate()
tfidf_clf_LR_report = tfidf_clf_LR.cross_validate()
word2vec_without_clf_LR_report = word2vec_without_clf_LR.cross_validate()
word2vec_with_clf_LR_report = word2vec_with_clf_LR.cross_validate()
doc2vec_without_clf_LR_report = doc2vec_without_clf_LR.cross_validate()
doc2vec_with_clf_LR_report = doc2vec_with_clf_LR.cross_validate()
bert_clf_LR_report = bert_clf_LR.cross_validate()

logger.debug("SVM")
count_clf_SVM_report = count_clf_SVM.cross_validate()
tfidf_clf_SVM_report = tfidf_clf_SVM.cross_validate()
word2vec_without_clf_SVM_report = word2vec_without_clf_SVM.cross_validate()
word2vec_with_clf_SVM_report = word2vec_with_clf_SVM.cross_validate()
doc2vec_without_clf_SVM_report = doc2vec_without_clf_SVM.cross_validate()
doc2vec_with_clf_SVM_report = doc2vec_with_clf_SVM.cross_validate()
bert_clf_SVM_report = bert_clf_SVM.cross_validate()

logger.debug("RF")
count_clf_RF_report = count_clf_RF.cross_validate()
tfidf_clf_RF_report = tfidf_clf_RF.cross_validate()
word2vec_without_clf_RF_report = word2vec_without_clf_RF.cross_validate()
word2vec_with_clf_RF_report = word2vec_with_clf_RF.cross_validate()
doc2vec_without_clf_RF_report = doc2vec_without_clf_RF.cross_validate()
doc2vec_with_clf_RF_report = doc2vec_with_clf_RF.cross_validate()
bert_clf_RF_report = bert_clf_RF.cross_validate()


# TO_LATEX
logger.debug("LATEX")
logger.debug("Accuracy")
df_accuracy = pd.DataFrame(
    {
        "LR": [
            f"{round(count_clf_LR_report['accuracy'][0], 2)} {count_clf_LR_report['accuracy'][0]}",
            f"{round(tfidf_clf_LR_report['accuracy'][0], 2)} {tfidf_clf_LR_report['accuracy'][1]}",
            f"{round(word2vec_without_clf_LR_report['accuracy'][0], 2)} {word2vec_without_clf_LR_report['accuracy'][1]}",
            f"{round(word2vec_with_clf_LR_report['accuracy'][0], 2)} {word2vec_with_clf_LR_report['accuracy'][1]}",
            f"{round(doc2vec_without_clf_LR_report['accuracy'][0], 2)} {doc2vec_without_clf_LR_report['accuracy'][1]}",
            f"{round(doc2vec_with_clf_LR_report['accuracy'][0], 2)} {doc2vec_with_clf_LR_report['accuracy'][1]}",
            f"{round(bert_clf_LR_report['accuracy'][0], 2)} {bert_clf_LR_report['accuracy'][1]}",
        ],
        "SVM": [
            f"{round(count_clf_SVM_report['accuracy'][0], 2)} {count_clf_SVM_report['accuracy'][1]}",
            f"{round(tfidf_clf_SVM_report['accuracy'][0], 2)} {tfidf_clf_SVM_report['accuracy'][1]}",
            f"{round(word2vec_without_clf_SVM_report['accuracy'][0], 2)} {word2vec_without_clf_SVM_report['accuracy'][1]}",
            f"{round(word2vec_with_clf_SVM_report['accuracy'][0], 2)} {word2vec_with_clf_SVM_report['accuracy'][1]}",
            f"{round(doc2vec_without_clf_SVM_report['accuracy'][0], 2)} {doc2vec_without_clf_SVM_report['accuracy'][1]}",
            f"{round(doc2vec_with_clf_SVM_report['accuracy'][0], 2)} {doc2vec_with_clf_SVM_report['accuracy'][1]}",
            f"{round(bert_clf_SVM_report['accuracy'][0], 2)} {bert_clf_SVM_report['accuracy'][1]}",
        ],
        "RF": [
            f"{round(count_clf_RF_report['accuracy'][0], 2)} {count_clf_RF_report['accuracy'][1]}",
            f"{round(tfidf_clf_RF_report['accuracy'][0], 2)} {tfidf_clf_RF_report['accuracy'][1]}",
            f"{round(word2vec_without_clf_RF_report['accuracy'][0], 2)} {word2vec_without_clf_RF_report['accuracy'][1]}",
            f"{round(word2vec_with_clf_RF_report['accuracy'][0], 2)} {word2vec_with_clf_RF_report['accuracy'][1]}",
            f"{round(doc2vec_without_clf_RF_report['accuracy'][0], 2)} {doc2vec_without_clf_RF_report['accuracy'][1]}",
            f"{round(doc2vec_with_clf_RF_report['accuracy'][0], 2)} {doc2vec_with_clf_RF_report['accuracy'][1]}",
            f"{round(bert_clf_RF_report['accuracy'][0], 2)} {bert_clf_RF_report['accuracy'][1]}",
        ],
    },
    index=[
        "CountVectorizer",
        "TFIDF",
        "Word2Vec_I",
        "Word2Vec_II",
        "Doc2Vec_I",
        "Doc2Vec_II",
        "BERT",
    ],
)

print(df_accuracy.to_latex())

logger.debug("Precision, Recall, F1-score macro")
df_prf = pd.DataFrame(
    {
        "LR": [
            f"p: {round(count_clf_LR_report['precision_macro'], 2)}, r: {round(count_clf_LR_report['recall_macro'], 2)}, F1: {round(count_clf_LR_report['f1_macro'], 2)}",
            f"p: {round(tfidf_clf_LR_report['precision_macro'], 2)}, r: {round(tfidf_clf_LR_report['recall_macro'], 2)}, F1: {round(tfidf_clf_LR_report['f1_macro'], 2)}",
            f"p: {round(word2vec_without_clf_LR_report['precision_macro'], 2)}, r: {round(word2vec_without_clf_LR_report['recall_macro'], 2)}, F1: {round(word2vec_without_clf_LR_report['f1_macro'], 2)}",
            f"p: {round(word2vec_with_clf_LR_report['precision_macro'], 2)}, r: {round(word2vec_with_clf_LR_report['recall_macro'], 2)}, F1: {round(word2vec_with_clf_LR_report['f1_macro'], 2)}",
            f"p: {round(doc2vec_without_clf_LR_report['precision_macro'], 2)}, r: {round(doc2vec_without_clf_LR_report['recall_macro'], 2)}, F1: {round(doc2vec_without_clf_LR_report['f1_macro'], 2)}",
            f"p: {round(doc2vec_with_clf_LR_report['precision_macro'], 2)}, r: {round(doc2vec_with_clf_LR_report['recall_macro'], 2)}, F1: {round(doc2vec_with_clf_LR_report['f1_macro'], 2)}",
            f"p: {round(bert_clf_LR_report['precision_macro'], 2)}, r: {round(bert_clf_LR_report['recall_macro'], 2)}, F1: {round(bert_clf_LR_report['f1_macro'], 2)}",
        ],
        "SVM": [
            f"p: {round(count_clf_SVM_report['precision_macro'], 2)}, r: {round(count_clf_SVM_report['recall_macro'], 2)}, F1: {round(count_clf_SVM_report['f1_macro'], 2)}",
            f"p: {round(tfidf_clf_SVM_report['precision_macro'], 2)}, r: {round(tfidf_clf_SVM_report['recall_macro'], 2)}, F1: {round(tfidf_clf_SVM_report['f1_macro'], 2)}",
            f"p: {round(word2vec_without_clf_SVM_report['precision_macro'], 2)}, r: {round(word2vec_without_clf_SVM_report['recall_macro'], 2)}, F1: {round(word2vec_without_clf_SVM_report['f1_macro'], 2)}",
            f"p: {round(word2vec_with_clf_SVM_report['precision_macro'], 2)}, r: {round(word2vec_with_clf_SVM_report['recall_macro'], 2)}, F1: {round(word2vec_with_clf_SVM_report['f1_macro'], 2)}",
            f"p: {round(doc2vec_without_clf_SVM_report['precision_macro'], 2)}, r: {round(doc2vec_without_clf_SVM_report['recall_macro'], 2)}, F1: {round(doc2vec_without_clf_SVM_report['f1_macro'], 2)}",
            f"p: {round(doc2vec_with_clf_SVM_report['precision_macro'], 2)}, r: {round(doc2vec_with_clf_SVM_report['recall_macro'], 2)}, F1: {round(doc2vec_with_clf_SVM_report['f1_macro'], 2)}",
            f"p: {round(bert_clf_SVM_report['precision_macro'], 2)}, r: {round(bert_clf_SVM_report['recall_macro'], 2)}, F1: {round(bert_clf_SVM_report['f1_macro'], 2)}",
        ],
        "RF": [
            f"p: {round(count_clf_RF_report['precision_macro'], 2)}, r: {round(count_clf_RF_report['recall_macro'], 2)}, F1: {round(count_clf_RF_report['f1_macro'], 2)}",
            f"p: {round(tfidf_clf_RF_report['precision_macro'], 2)}, r: {round(tfidf_clf_RF_report['recall_macro'], 2)}, F1: {round(tfidf_clf_RF_report['f1_macro'], 2)}",
            f"p: {round(word2vec_without_clf_RF_report['precision_macro'], 2)}, r: {round(word2vec_without_clf_RF_report['recall_macro'], 2)}, F1: {round(word2vec_without_clf_RF_report['f1_macro'], 2)}",
            f"p: {round(word2vec_with_clf_RF_report['precision_macro'], 2)}, r: {round(word2vec_with_clf_RF_report['recall_macro'], 2)}, F1: {round(word2vec_with_clf_RF_report['f1_macro'], 2)}",
            f"p: {round(doc2vec_without_clf_RF_report['precision_macro'], 2)}, r: {round(doc2vec_without_clf_RF_report['recall_macro'], 2)}, F1: {round(doc2vec_without_clf_RF_report['f1_macro'], 2)}",
            f"p: {round(doc2vec_with_clf_RF_report['precision_macro'], 2)}, r: {round(doc2vec_with_clf_RF_report['recall_macro'], 2)}, F1: {round(doc2vec_with_clf_RF_report['f1_macro'], 2)}",
            f"p: {round(bert_clf_RF_report['precision_macro'], 2)}, r: {round(bert_clf_RF_report['recall_macro'], 2)}, F1: {round(bert_clf_RF_report['f1_macro'], 2)}",
        ],
    },
    index=[
        "CountVectorizer",
        "TFIDF",
        "Word2Vec_I",
        "Word2vec_II",
        "Doc2Vec_I",
        "Doc2Vec_II",
        "BERT",
    ],
)

print(df_prf.to_latex())


logger.debug("Precision, Recall, F1-score micro")
df_prf = pd.DataFrame(
    {
        "LR": [
            f"p: {round(count_clf_LR_report['precision_micro'], 2)}, r: {round(count_clf_LR_report['recall_micro'], 2)}, F1: {round(count_clf_LR_report['f1_micro'], 2)}",
            f"p: {round(tfidf_clf_LR_report['precision_micro'], 2)}, r: {round(tfidf_clf_LR_report['recall_micro'], 2)}, F1: {round(tfidf_clf_LR_report['f1_micro'], 2)}",
            f"p: {round(word2vec_without_clf_LR_report['precision_micro'], 2)}, r: {round(word2vec_without_clf_LR_report['recall_micro'], 2)}, F1: {round(word2vec_without_clf_LR_report['f1_micro'], 2)}",
            f"p: {round(word2vec_with_clf_LR_report['precision_micro'], 2)}, r: {round(word2vec_with_clf_LR_report['recall_micro'], 2)}, F1: {round(word2vec_with_clf_LR_report['f1_micro'], 2)}",
            f"p: {round(doc2vec_without_clf_LR_report['precision_micro'], 2)}, r: {round(doc2vec_without_clf_LR_report['recall_micro'], 2)}, F1: {round(doc2vec_without_clf_LR_report['f1_micro'], 2)}",
            f"p: {round(doc2vec_with_clf_LR_report['precision_micro'], 2)}, r: {round(doc2vec_with_clf_LR_report['recall_micro'], 2)}, F1: {round(doc2vec_with_clf_LR_report['f1_micro'], 2)}",
            f"p: {round(bert_clf_LR_report['precision_micro'], 2)}, r: {round(bert_clf_LR_report['recall_micro'], 2)}, F1: {round(bert_clf_LR_report['f1_micro'], 2)}",
        ],
        "SVM": [
            f"p: {round(count_clf_SVM_report['precision_micro'], 2)}, r: {round(count_clf_SVM_report['recall_micro'], 2)}, F1: {round(count_clf_SVM_report['f1_micro'], 2)}",
            f"p: {round(tfidf_clf_SVM_report['precision_micro'], 2)}, r: {round(tfidf_clf_SVM_report['recall_micro'], 2)}, F1: {round(tfidf_clf_SVM_report['f1_micro'], 2)}",
            f"p: {round(word2vec_without_clf_SVM_report['precision_micro'], 2)}, r: {round(word2vec_without_clf_SVM_report['recall_micro'], 2)}, F1: {round(word2vec_without_clf_SVM_report['f1_micro'], 2)}",
            f"p: {round(word2vec_with_clf_SVM_report['precision_micro'], 2)}, r: {round(word2vec_with_clf_SVM_report['recall_micro'], 2)}, F1: {round(word2vec_with_clf_SVM_report['f1_micro'], 2)}",
            f"p: {round(doc2vec_without_clf_SVM_report['precision_micro'], 2)}, r: {round(doc2vec_without_clf_SVM_report['recall_micro'], 2)}, F1: {round(doc2vec_without_clf_SVM_report['f1_micro'], 2)}",
            f"p: {round(doc2vec_with_clf_SVM_report['precision_micro'], 2)}, r: {round(doc2vec_with_clf_SVM_report['recall_micro'], 2)}, F1: {round(doc2vec_with_clf_SVM_report['f1_micro'], 2)}",
            f"p: {round(bert_clf_SVM_report['precision_micro'], 2)}, r: {round(bert_clf_SVM_report['recall_micro'], 2)}, F1: {round(bert_clf_SVM_report['f1_micro'], 2)}",
        ],
        "RF": [
            f"p: {round(count_clf_RF_report['precision_micro'], 2)}, r: {round(count_clf_RF_report['recall_micro'], 2)}, F1: {round(count_clf_RF_report['f1_micro'], 2)}",
            f"p: {round(tfidf_clf_RF_report['precision_micro'], 2)}, r: {round(tfidf_clf_RF_report['recall_micro'], 2)}, F1: {round(tfidf_clf_RF_report['f1_micro'], 2)}",
            f"p: {round(word2vec_without_clf_RF_report['precision_micro'], 2)}, r: {round(word2vec_without_clf_RF_report['recall_micro'], 2)}, F1: {round(word2vec_without_clf_RF_report['f1_micro'], 2)}",
            f"p: {round(word2vec_with_clf_RF_report['precision_micro'], 2)}, r: {round(word2vec_with_clf_RF_report['recall_micro'], 2)}, F1: {round(word2vec_with_clf_RF_report['f1_micro'], 2)}",
            f"p: {round(doc2vec_without_clf_RF_report['precision_micro'], 2)}, r: {round(doc2vec_without_clf_RF_report['recall_micro'], 2)}, F1: {round(doc2vec_without_clf_RF_report['f1_micro'], 2)}",
            f"p: {round(doc2vec_with_clf_RF_report['precision_micro'], 2)}, r: {round(doc2vec_with_clf_RF_report['recall_micro'], 2)}, F1: {round(doc2vec_with_clf_RF_report['f1_micro'], 2)}",
            f"p: {round(bert_clf_RF_report['precision_micro'], 2)}, r: {round(bert_clf_RF_report['recall_micro'], 2)}, F1: {round(bert_clf_RF_report['f1_micro'], 2)}",
        ],
    },
    index=[
        "CountVectorizer",
        "TFIDF",
        "Word2Vec_I",
        "Word2vec_II",
        "Doc2Vec_I",
        "Doc2Vec_II",
        "BERT",
    ],
)

print(df_prf.to_latex())
