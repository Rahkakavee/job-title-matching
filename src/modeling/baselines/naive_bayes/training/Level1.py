from src.preparation.training_data import TrainingData
from src.modeling.naive_bayes.bayes_classifier import BayesClassifier
import pandas as pd

## Level 1
# create data
# Level 1
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


# classify
classfication_level1 = BayesClassifier(
    data=training_data_level_1_cleaned, vectorizer="CountVectorizer"
)
classfication_level1.train_classifier()
classfication_level1.evaluate(output_dict=True)
print(classfication_level1.accuracy)
print(classfication_level1.classfication_report)

# Export
df = pd.DataFrame(classfication_level1.classfication_report).transpose()
print(df.to_latex())
