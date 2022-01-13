# import
from src.descriptive.descriptive_analysis.descriptive_analysis_functions import *
import json

with open(file="data/processed/training_data_short_l1.json") as fp:
    training_data_short_l1 = json.load(fp=fp)

with open(file="data/processed/training_data_short_l3.json") as fp:
    training_data_short_l3 = json.load(fp=fp)

with open(file="data/processed/training_data_short_l5.json") as fp:
    training_data_short_l5 = json.load(fp=fp)


class_distribution, plt = class_distribution(
    data=training_data_short_l1, variable="id", level="1"
)
plt.savefig("visualization/descriptive_analysis/training_data_short_L1.png")


class_distribution, plt = class_distribution(
    data=training_data_short_l3, variable="id", level="3"
)

plt.savefig("visualization/descriptive_analysis//training_data_short_L3.png")


class_distribution, plt = class_distribution(
    data=training_data_short_l5, variable="id", level="5"
)
plt.savefig("visualization/descriptive_analysis//training_data_short_L5.png")
