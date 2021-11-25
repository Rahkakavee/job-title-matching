from src.preparation.training_data import TrainingData
from collections import Counter
from tqdm import tqdm

data = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/raw/2021-10-22_12-21-00_all_jobs_7.json",
    kldb_level=5,
)

data.create_training_data()

jobs = data.training_data
kldbs = data.kldbs

searchwords = set()
kldb_searchwords = []

kldbs_level_5 = [kldb for kldb in kldbs if kldb["level"] == 5]

for kldb in kldbs_level_5:
    searchwords_ = [searchword["name"] for searchword in kldb["searchwords"]]
    for searchword in searchwords_:
        searchwords.add(searchword)
    kldb_searchwords.append({"id": kldb["id"], "searchwords": searchwords_})

searchwords_list_ = list(searchwords)

prior_probs = []

for searchword in tqdm(searchwords_list_):
    ids = []
    for job in data.training_data:
        if searchword in job["title"]:
            ids.append(job["id"])
    ids_count = dict(Counter(ids))
    if len(ids_count) > 0:
        prior_probs.append({"searchword": searchword, "ids": ids_count})

for searchword in prior_probs:
    sum = 0
    print(type(searchword["ids"]))
    counts = list(searchword["ids"].values())
    for count in counts:
        sum += count
    print(counts)
    for key, value in searchword["ids"].items():
        searchword["ids"][key] = value / sum
