from src.preprocessing.preprocessing_functions import *
from src.preprocessing.training_data import TrainingData
from src.modeling.baselines.SVM.svm_classifier import SVMClassifier
from src.modeling.baselines.naive_bayes.bayes_classifier import BayesClassifier
from src.modeling.baselines.LR.lr_classifier import LRClassifier
import pandas as pd
from src.logger import logger
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

logger.debug("#######TRAINING DATA#######")
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

with open("src/preprocessing/specialwords.tex", "rb") as fp:
    specialwords = pickle.load(fp)

logger.debug("#######Preprocessing#######")
# Preprocess
training_data = preprocess(
    data=data, lowercase_whitespace=False, special_words_ovr=specialwords
)
vectorizer = CountVectorizer(lowercase=False)
df = pd.DataFrame(training_data[5000])
training_data = vectorizer.fit_transform(df["title"]).toarray()
labels = df["id"]
train, test, y_train, y_test = train_test_split(training_data, labels)

# clf = LogisticRegression(n_jobs=1, penalty = "l2", solver="lbfgs")
#  accuracy                           0.64      1250
#    macro avg       0.69      0.54      0.59      1250
# weighted avg       0.67      0.64      0.63      1250

# #    accuracy                           0.49      1250
#    macro avg       0.50      0.56      0.46      1250
# weighted avg       0.64      0.49      0.52      1250


#    accuracy                           0.65      1250
#    macro avg       0.73      0.53      0.56      1250
# weighted avg       0.67      0.65      0.64      1250
