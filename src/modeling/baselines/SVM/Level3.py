from src.preparation.training_data import TrainingData
from src.modeling.baselines.SVM.svm_classifier import SVMClassifier


kldb_level_3 = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/raw/2021-10-22_12-21-00_all_jobs_7.json",
    kldb_level=3,
)
kldb_level_3.create_training_data()

clf = SVMClassifier(data=kldb_level_3.training_data, vectorizer="CountVectorizer")

clf.train_classifier()
clf.evaluate(output_dict=False)
print(clf.accuracy)
print(clf.classfication_report)
