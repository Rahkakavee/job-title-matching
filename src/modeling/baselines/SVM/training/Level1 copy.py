from src.preparation.training_data import TrainingData
from src.modeling.baselines.SVM.svm_classifier import SVMClassifier
import pandas as pd
import pickle

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

training_data_level_1_cleaned = [
    dict(t) for t in {tuple(example.items()) for example in training_data_level_1}
]


clf = SVMClassifier(data=training_data_level_1_cleaned, vectorizer="CountVectorizer")

clf.train_classifier()
clf.evaluate(output_dict=True)

print(clf.accuracy)
print(clf.classfication_report)

# Export
df = pd.DataFrame(clf.classfication_report).transpose()
print(df.to_latex())

# now you can save it to a file
with open("model/SVM_level1_clf.pkl", "wb") as f:
    pickle.dump(clf, f)

# # and later you can load it
# with open('filename.pkl', 'rb') as f:
#     clf = pickle.load(f)
