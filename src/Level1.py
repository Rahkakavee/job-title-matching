from src.preprocessing.preprocessing_functions import *
from src.preprocessing.training_data import TrainingData
from collections import Counter
from src.modeling.baselines.SVM.svm_classifier import SVMClassifier
import pandas as pd
import pickle

# Training Data
data_level_1_old = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/processed/data_old_format.json",
    kldb_level=1,
    new_data=False,
)
data_level_1_new = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/processed/data_new_format.json",
    kldb_level=1,
    new_data=True,
)

data_level_1_old.create_training_data()
data_level_1_new.create_training_data()

training_data_level_1 = data_level_1_old.training_data + data_level_1_new.training_data

data = [
    dict(t) for t in {tuple(example.items()) for example in training_data_level_1}
]  # source: "https://stackoverflow.com/questions/9427163/remove-duplicate-dict-in-list-in-python"

with open("src/preprocessing/specialwords.ob", "rb") as fp:
    specialwords = pickle.load(fp)


# Preprocess
training_data = preprocess(data=data, special_words_ovr=specialwords)


# Train classifier
SVM_clf = SVMClassifier(data=training_data[:1000], vectorizer="CountVectorizer")
SVM_clf.train_classifier()
SVM_clf.evaluate(output_dict=True)
print(SVM_clf.accuracy)
print(SVM_clf.classfication_report)
df = pd.DataFrame(SVM_clf.classfication_report).transpose()
print(df.to_latex())


# Save classifier
with open("model/SVM_clf_level1.pkl", "wb") as f:
    pickle.dump(SVM_clf, f)


# with open('filename.pkl', 'rb') as f:
#     clf = pickle.load(f)

#               precision    recall  f1-score   support

#            1       0.83      0.63      0.72       210
#            2       0.75      0.85      0.80      4982
#            3       0.80      0.69      0.74      1434
#            4       0.77      0.76      0.77      2080
#            5       0.87      0.81      0.84      2883
#            6       0.73      0.70      0.72      2222
#            7       0.71      0.74      0.73      3303
#            8       0.90      0.83      0.86      2156
#            9       0.60      0.54      0.57       730

#     accuracy                           0.77     20000
#    macro avg       0.77      0.73      0.75     20000
# weighted avg       0.78      0.77      0.77     20000

#             precision    recall  f1-score   support

#            1       0.83      0.64      0.72       210
#            2       0.75      0.85      0.80      4982
#            3       0.80      0.69      0.74      1434
#            4       0.77      0.76      0.77      2080
#            5       0.88      0.81      0.84      2883
#            6       0.74      0.70      0.72      2222
#            7       0.71      0.75      0.73      3303
#            8       0.89      0.83      0.86      2156
#            9       0.62      0.56      0.59       730

#     accuracy                           0.77     20000
#    macro avg       0.78      0.73      0.75     20000
# weighted avg       0.78      0.77      0.77     20000
