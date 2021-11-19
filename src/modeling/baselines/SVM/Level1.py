from src.preparation.training_data import TrainingData
from src.modeling.baselines.SVM.svm_classifier import SVMClassifier


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
clf.evaluate(output_dict=False)
print(clf.accuracy)
print(clf.classfication_report)
