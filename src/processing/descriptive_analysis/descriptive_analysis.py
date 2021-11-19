from src.preparation.training_data import TrainingData
from src.processing.descriptive_analysis.descriptive_analysis_functions import (
    class_distribution, counting_per_job, counting_per_title
)
import pandas as pd
from re import search 
from collections import Counter


# Level 5
data_level_5_old = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/processed/data_old_format.json",
    kldb_level=5,
    new_data=False,
)

data_level_5_new = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/processed/data_new_format.json",
    kldb_level=5,
    new_data=True,
)

data_level_5_old.create_training_data()
data_level_5_new.create_training_data()

training_data_level_5 = data_level_5_old.training_data + data_level_5_new.training_data

training_data_level_5_cleaned = [dict(t) for t in {tuple(example.items()) for example in training_data_level_5}] #source: "https://stackoverflow.com/questions/9427163/remove-duplicate-dict-in-list-in-python"

class_distribution_level_5, plt_level_5 = class_distribution(
    data=training_data_level_5_cleaned, variable="id", level="5"
)

job_titles_count = counting_per_title(training_data_level_5_cleaned, 20)

for job in job_titles_count:
    