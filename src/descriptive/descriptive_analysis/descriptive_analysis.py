from src.descriptive.descriptive_analysis_functions import *
from src.preprocessing.preprocessing_functions import *
import json


"""Analysis of frequencies of kldbs per occupation"""


logger.debug("LOAD TRAINING DATA")
with open(file="data/processed/training_data_short_l3.json") as fp:
    training_data_short = json.load(fp=fp)

with open(file="data/raw/dictionary_occupations_complete_update.json") as fp:
    kldbs = json.load(fp=fp)

searchterms_servicekraft = [
    "servicekraft",
    "service kraft",
    "servicekräfte",
    "service kräfte",
]

plot_servicekraft = kldbs_counting(
    data=training_data_short, searchterms=searchterms_servicekraft
)

fig = plot_servicekraft.get_figure()
fig.savefig("visualization/limitations/kldbs_frequency_servicekraft.jpg")

searchterms_maurer = ["maurer", "maurerin"]

plot_maurer = kldbs_counting(data=training_data_short, searchterms=searchterms_maurer)

fig = plot_maurer.get_figure()
fig.savefig("visualization/limitations/kldbs_frequency_maurer.jpg")

searchterms_elektroniker = ["elektriker", "elektrikerin"]

plot_elektroniker = kldbs_counting(
    data=training_data_short, searchterms=searchterms_elektroniker
)

fig = plot_elektroniker.get_figure()
fig.savefig("visualization/limitations/kldbs_frequency_elektroniker.jpg")

searchterms_softwareentwickler = [
    "softwareentwickler",
    "software entwickler",
    "softwareentwicklerin",
    "software entwicklerin",
]

plot_softwareentwickler = kldbs_counting(
    data=training_data_short, searchterms=searchterms_softwareentwickler
)

fig = plot_softwareentwickler.get_figure()
fig.savefig("visualization/limitations/kldbs_frequency_softwareentwickler.jpg")
