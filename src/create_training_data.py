from src.preprocessing.preprocessing_functions import *
from src.focussing.training_data import TrainingData
from src.logger import logger
import pickle
import random
import json


"""Create a short and long version of the training data, which is used for the classifier"""

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

with open("src/preprocessing/specialwords.txt", "rb") as fp:
    specialwords = pickle.load(fp)

logger.debug("#######Preprocessing#######")
# Preprocess
training_data = preprocess(
    data=data, lowercase_whitespace=False, special_words_ovr=specialwords
)

training_data_short = random.sample(training_data, 15000)

with open(file="data/processed/training_data_long.json", mode="w") as fp:
    json.dump(obj=training_data, fp=fp)

with open(file="data/processed/training_data_short.json", mode="w") as fp:
    json.dump(obj=training_data_short, fp=fp)
