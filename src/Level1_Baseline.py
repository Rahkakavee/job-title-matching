from src.preprocessing.preprocessing_functions import *
from src.modeling.baselines.SVM.svm_classifier import SVMClassifier
from src.modeling.baselines.naive_bayes.bayes_classifier import BayesClassifier
from src.modeling.baselines.LR.lr_classifier import LRClassifier
from src.logger import logger
import pickle
import json
import pandas as pd

logger.debug("#######LOAD TRAINING DATA#######")
with open(file="data/processed/training_data_long.json") as fp:
    training_data_long = json.load(fp=fp)

with open(file="data/processed/training_data_short.json") as fp:
    training_data_short = json.load(fp=fp)

# logger.debug("#######SVM COUNTVECTORIZER#######")
# # Train SVM classifier with TfidfVectorizer
# SVM_CountVectorizer_clf = SVMClassifier(
#     data=training_data_short, vectorizer="CountVectorizer"
# )
# SVM_CountVectorizer_clf.train_classifier()
# SVM_CountVectorizer_clf.evaluate(output_dict=True)
# print(SVM_CountVectorizer_clf.accuracy)
# print(SVM_CountVectorizer_clf.classfication_report)
# df = pd.DataFrame(SVM_CountVectorizer_clf.classfication_report).transpose()
# print(df.to_latex())

# logger.debug("#######SVM TF-IDF")
# # Train SVM classifier with TfidfVectorizer
# SVM_TfidfVectorizer_clf = SVMClassifier(
#     data=training_data_short, vectorizer="TfidfVectorizer"
# )
# SVM_TfidfVectorizer_clf.train_classifier()
# SVM_TfidfVectorizer_clf.evaluate(output_dict=True)
# print(SVM_TfidfVectorizer_clf.accuracy)
# print(SVM_TfidfVectorizer_clf.classfication_report)
# df = pd.DataFrame(SVM_TfidfVectorizer_clf.classfication_report).transpose()
# print(df.to_latex())


# # Save SVM classifiers
# with open("model/SVM_CountVectorizer_clf_level1_short.pkl", "wb") as f:
#     pickle.dump(SVM_CountVectorizer_clf, f)
# with open("model/SVM_TfidfVectorizer_clf_level1_short.pkl", "wb") as f:
#     pickle.dump(SVM_TfidfVectorizer_clf, f)

# logger.debug("#######NB COUNTVECTORIZER#######")
# # Train Bayes classifier with CountVectorizer
# NB_CountVectorizer_clf = BayesClassifier(
#     data=training_data, vectorizer="CountVectorizer"
# )
# NB_CountVectorizer_clf.train_classifier()
# NB_CountVectorizer_clf.evaluate(output_dict=True)
# print(NB_CountVectorizer_clf.classfication_report)
# df = pd.DataFrame(NB_CountVectorizer_clf.classfication_report).transpose()
# print(df.to_latex())

# logger.debug("#######NB TF-IDF#######")
# # Train Bayes classifier with TfidfVectorizer
# NB_TfidfVectorizer_clf = BayesClassifier(
#     data=training_data, vectorizer="TfidfVectorizer"
# )
# NB_TfidfVectorizer_clf.train_classifier()
# NB_TfidfVectorizer_clf.evaluate(output_dict=True)
# print(NB_TfidfVectorizer_clf.classfication_report)
# df = pd.DataFrame(NB_TfidfVectorizer_clf.classfication_report).transpose()
# print(df.to_latex())

# # Save NB classifiers
# with open("model/NB_CountVectorizer_clf_level1.pkl", "wb") as f:
#     pickle.dump(NB_CountVectorizer_clf.clf, f)
# with open("model/SVM_clf_TfidfVectorizer_level1.pkl", "wb") as f:
#     pickle.dump(NB_TfidfVectorizer_clf.clf, f)

# logger.debug("#######LR COUNTVECTORIZER#######")
# Train LR classifier with CountVectorizer
LR_CountVectorizer_clf = LRClassifier(
    data=training_data_short, vectorizer="CountVectorizer"
)
LR_CountVectorizer_clf.train_classifier()
LR_CountVectorizer_clf.evaluate(output_dict=False)
print(LR_CountVectorizer_clf.classfication_report)

# logger.debug("#######LR TF-IDF#######")
# # Train LR classifier with TfidfVectorizer
# LR_TfidfVectorizer_clf = LRClassifier(
#     data=training_data[:5000], vectorizer="TfidfVectorizer"
# )
# LR_TfidfVectorizer_clf.train_classifier()
# LR_TfidfVectorizer_clf.evaluate(output_dict=False)
# print(LR_TfidfVectorizer_clf.classfication_report)

# Save NB classifiers
# with open("model/LR_CountVectorizer_clf_level1.pkl", "wb") as f:
#     pickle.dump(LR_CountVectorizer_clf, f)
# with open("model/LR_TfidfVectorizer_clf_level1.pkl", "wb") as f:
#     pickle.dump(LR_TfidfVectorizer_clf, f)


# with open('filename.pkl', 'rb') as f:
#     clf = pickle.load(f)


#     accuracy                           0.66      2500
#    macro avg       0.68      0.51      0.54      2500
# weighted avg       0.71      0.66      0.64      2500

#     accuracy                           0.69      2500
#    macro avg       0.77      0.57      0.60      2500
# weighted avg       0.72      0.69      0.68      2500
