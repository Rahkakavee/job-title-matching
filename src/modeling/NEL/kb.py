from typing import Union, List, Dict
import jsonlines
from tqdm import tqdm
import json
from src.preparation.training_data import TrainingData
from collections import Counter
import re


def load_entities(kldbs: Union[List, Dict]) -> Union[List, Dict]:
    entities = [
        {"id": kldb["id"], "description": kldb["description"]}
        for kldb in kldbs
        if kldb["level"] == 5
    ]
    return entities


def load_aliases(kldbs: Union[List, Dict], prior_probs: List) -> Union[List, Dict]:
    kldbs_level_5 = [kldb for kldb in kldbs if kldb["level"] == 5]

    searchwords = set()
    kldb_searchwords = []
    aliases: list = []

    for kldb in kldbs_level_5:  # loop over the klbds
        searchwords_ = [
            searchword["name"] for searchword in kldb["searchwords"]
        ]  # get searchwords per id
        for searchword in searchwords_:
            searchwords.add(searchword)  # add to set with all searchwords
        kldb_searchwords.append(
            {"id": kldb["id"], "searchwords": searchwords_}  # ids with searchwords
        )  # dict_ with id + searchword

    searchwords_list_ = list(searchwords)

    for searchword in tqdm(searchwords_list_):
        ids_ = [
            kldb_searchword["id"]
            for kldb_searchword in kldb_searchwords
            if searchword in kldb_searchword["searchwords"]
        ]
        if len(ids_) > 0:
            prob = 1 / len(ids_)
        aliases.append(
            {"id": searchword, "entities": ids_, "probabilities": [prob] * len(ids_)}
        )

    for alias in tqdm(aliases):
        updated_ids_ = []
        updated_probs = []
        for searchword in prior_probs:  # loop through searchwords
            if searchword["searchword"] == alias["id"]:  # if the searchword is in alias
                for key, value in searchword["ids"].items():
                    print(key)  # get for this searchword the id and probs
                    updated_ids_.append(key)
                    updated_probs.append(value)
                alias["entities"] = updated_ids_
                alias["probabilities"] = updated_probs
    return aliases


def to_jsonl(data: Union[List, Dict], filename: str) -> None:
    with jsonlines.open(filename + ".jsonl", "w") as f:
        f.write_all(data)


if __name__ == "__main__":
    data_old = TrainingData(
        kldbs_path="data/raw/dictionary_occupations_complete_update.json",
        data_path="data/processed/data_old_format.json",
        kldb_level=5,
        new_data=False,
    )

    data_new = TrainingData(
        kldbs_path="data/raw/dictionary_occupations_complete_update.json",
        data_path="data/processed/data_new_format.json",
        kldb_level=5,
        new_data=True,
    )

    data_old.create_training_data()
    data_new.create_training_data()

    data = data_old.training_data + data_new.training_data

    jobs = data
    kldbs = data_old.kldbs

    searchwords = set()
    kldb_searchwords = []

    kldbs_level_5 = [kldb for kldb in kldbs if kldb["level"] == 5]

    jobs_cleaned = [dict(t) for t in {tuple(example.items()) for example in jobs}]

    for job in tqdm(jobs_cleaned):
        job["title"] = re.sub("(m/w/d)", " ", job["title"])
        job["title"] = re.sub("(w/m/d)", " ", job["title"])
        job["title"] = re.sub("\W+", " ", job["title"])
        job["title"] = job["title"].lower()
        job["title"] = job["title"].lstrip()
        job["title"] = job["title"].rstrip()

    for kldb in kldbs_level_5:
        searchwords_ = [searchword["name"] for searchword in kldb["searchwords"]]
        for searchword in searchwords_:
            searchwords.add(searchword)
        kldb_searchwords.append({"id": kldb["id"], "searchwords": searchwords_})

    searchwords_list_ = list(searchwords)

    prior_probs = []

    for searchword in tqdm(searchwords_list_):
        ids = []
        for job in jobs_cleaned:
            if searchword in job["title"]:
                ids.append(job["id"])
        ids_count = dict(Counter(ids))
        if len(ids_count) > 0:
            prior_probs.append({"searchword": searchword, "ids": ids_count})

    for searchword in tqdm(prior_probs):
        sum = 0
        print(type(searchword["ids"]))
        counts = list(searchword["ids"].values())
        for count in counts:
            sum += count
        for key, value in searchword["ids"].items():
            searchword["ids"][key] = value / sum

    entities = load_entities(kldbs=kldbs)
    aliases = load_aliases(kldbs=kldbs, prior_probs=prior_probs)

    for alias in aliases:
        for key, value in alias.items():
            if value == "Studiotechnik":
                print(alias)

    to_jsonl(entities, "src/modeling/NEL/kb_dir/entities")
    to_jsonl(aliases, "src/modeling/NEL/kb_dir/aliases")
