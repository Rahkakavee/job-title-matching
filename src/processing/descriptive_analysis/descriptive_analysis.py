# import
from src.preparation.json_load import load_json
from src.preparation.training_data import TrainingData
from src.processing.descriptive_analysis.descriptive_analysis_functions import (
    class_distribution,
)

# # Level 1
# # create training_data
# kldb_level_1 = TrainingData(kldbs=kldbs, data=jobs, kldb_level=1)
# kldb_level_1.create_training_data()

# # # class distribution
# class_distribution, plt = class_distribution(
#     data=kldb_level_1.training_data, variable="id", level="1"
# )

# plt.savefig(
#     "visualization/descriptive_analysis/2021-10-22_12-21-00_all_jobs_7_level1.png"
# )

# # Level3
# # create training data
# kldb_level_3 = TrainingData(kldbs=kldbs, data=jobs, kldb_level=3)
# kldb_level_3.create_training_data()

# # class distribution
# class_distribution, plt = class_distribution(
#     data=kldb_level_3.training_data, variable="id", level="3"
# )

# plt.savefig(
#     "visualization/descriptive_analysis/2021-10-22_12-21-00_all_jobs_7_level3.png"
# )


# Level5
# create training data
kldb_level_5 = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/raw/2021-10-22_12-21-00_all_jobs_7.json",
    kldb_level=5,
)
kldb_level_5.create_training_data()

# class distribution
class_distribution, plt = class_distribution(
    data=kldb_level_5.training_data, variable="id", level="5"
)
plt.savefig(
    "visualization/descriptive_analysis/2021-10-22_12-21-00_all_jobs_7_level5.png"
)
