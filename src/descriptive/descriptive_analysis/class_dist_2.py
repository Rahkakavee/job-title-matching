from src.preparation.training_data import TrainingData
from src.processing.descriptive_analysis.descriptive_analysis_functions import (
    class_distribution,
)

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

class_distribution_level_1, plt_level_1 = class_distribution(
    data=training_data_level_1_cleaned, variable="id", level="1"
)


# Level 3
data_level_3_old = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/processed/data_old_format.json",
    kldb_level=3,
    new_data=False,
)

data_level_3_new = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/processed/data_new_format.json",
    kldb_level=3,
    new_data=True,
)

data_level_3_old.create_training_data()
data_level_3_new.create_training_data()

training_data_level_3 = data_level_3_old.training_data + data_level_3_new.training_data


training_data_level_3_cleaned = [
    dict(t) for t in {tuple(example.items()) for example in training_data_level_3}
]

class_distribution_level_3, plt_level_3 = class_distribution(
    data=training_data_level_3_cleaned, variable="id", level="3"
)


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

training_data_level_5_cleaned = [
    dict(t) for t in {tuple(example.items()) for example in training_data_level_5}
]  # source: "https://stackoverflow.com/questions/9427163/remove-duplicate-dict-in-list-in-python"

class_distribution_level_5, plt_level_5 = class_distribution(
    data=training_data_level_5_cleaned, variable="id", level="5"
)
