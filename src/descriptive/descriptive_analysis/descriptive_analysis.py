from src.preprocessing.training_data import TrainingData
from src.descriptive.descriptive_analysis.descriptive_analysis_functions import *
from src.preprocessing.preprocessing_functions import *
import pandas as pd

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

data = remove_lc_ws(training_data_level_5_cleaned)
data = remove_special_characters(data)

searchterms_servicekraft = [
    "servicekraft",
    "service kraft",
    "servicekräfte",
    "service kräfte",
]
servicekraft, servicekraft_counting = counting_per_job(
    data=data, job_terms=searchterms_servicekraft, top=20
)
servicekraft_counting_kldb = counting_per_job_with_kldb(
    data=servicekraft,
    counts=servicekraft_counting,
    title=True,
    kldbs=data_level_5_old.kldbs,
)
plot_servicekraft = visualize_counting_per_job_with_kldb(
    data=servicekraft_counting, job="Servicekraft"
)

searchterms_maurer = [
    "maurer",
]
maurer, maurer_counting = counting_per_job(
    data=data, job_terms=searchterms_maurer, top=20
)
maurer_counting_kldb = counting_per_job_with_kldb(
    data=maurer,
    counts=maurer_counting,
    title=True,
    kldbs=data_level_5_old.kldbs,
)
plot_maurer = visualize_counting_per_job_with_kldb(data=maurer_counting, job="Maurer")


searchterms_elektroniker = [
    "elektroniker",
]
elektroniker, elektroniker_counting = counting_per_job(
    data=data, job_terms=searchterms_elektroniker, top=20
)
maurer_counting_kldb = counting_per_job_with_kldb(
    data=elektroniker,
    counts=elektroniker_counting,
    title=True,
    kldbs=data_level_5_old.kldbs,
)
plot_elektroniker = visualize_counting_per_job_with_kldb(
    data=elektroniker_counting, job="Elektroniker"
)


searchterms_elektriker = [
    "elektriker",
]
elektriker, elektriker_counting = counting_per_job(
    data=data, job_terms=searchterms_elektriker, top=20
)
maurer_counting_kldb = counting_per_job_with_kldb(
    data=elektriker,
    counts=elektriker_counting,
    title=True,
    kldbs=data_level_5_old.kldbs,
)
plot_elektroniker = visualize_counting_per_job_with_kldb(
    data=elektriker_counting, job="Elektriker"
)

searchterms_softwareentwickler = [
    "softwareentwickler, software entwickler, softwaredeveloper, software developer",
    "softwareingenieur",
    "softwareingenieur" "programmierer",
    "coder",
]
softwareentwickler, softwareentwickler_counting = counting_per_job(
    data=data, job_terms=searchterms_softwareentwickler, top=20
)
softwareentwickler_counting_kldbs = counting_per_job_with_kldb(
    data=softwareentwickler,
    counts=softwareentwickler_counting,
    title=True,
    kldbs=data_level_5_old.kldbs,
)
plot_softwareentwickler = visualize_counting_per_job_with_kldb(
    data=softwareentwickler_counting, job="Softwareentwickler"
)

searchterms_softwaretester = ["softwaretester", "software tester"]
softwaretester, softwaretester_counting = counting_per_job(
    data=data, job_terms=searchterms_softwaretester, top=20
)
softwaretester_counting_kldbs = counting_per_job_with_kldb(
    data=softwaretester,
    counts=softwaretester_counting,
    title=True,
    kldbs=data_level_5_old.kldbs,
)
plot_softwareentwickler = visualize_counting_per_job_with_kldb(
    data=softwaretester_counting, job="Softwaretester"
)


all_titles_counting = counting_per_title(data=data, top=20)
all_titles_kldb = counting_per_job_with_kldb(
    data=data, counts=all_titles_counting, title=True, kldbs=data_level_5_new.kldbs
)
plot_all_titles = visualize_counting_per_job_with_kldb(
    data=all_titles_counting, job="all_titles"
)
