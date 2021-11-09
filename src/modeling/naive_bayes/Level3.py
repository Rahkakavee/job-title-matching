from src.preparation.training_data import TrainingData
from src.preparation.json_load import load_json
from src.modeling.naive_bayes.bayes_classifier import BayesClassifier
import pandas as pd

# load data
kldbs = load_json("data/raw/dictionary_occupations_complete_update.json")
jobs = load_json("data/raw/2021-10-22_12-21-00_all_jobs_7.json")

## Level 1
# create data
kldb_level_3 = TrainingData(kldbs=kldbs, data=jobs, kldb_level=3)
kldb_level_3.create_training_data()

# classify
classfication_level3 = BayesClassifier(data=kldb_level_3.training_data)
classfication_level3.train_classifier()
classfication_level3.evaluate(output_dict=True)
print(classfication_level3.accuracy)
print(classfication_level3.classfication_report)

# Export
df = pd.DataFrame(classfication_level3.classfication_report).transpose()
print(df.to_latex())
