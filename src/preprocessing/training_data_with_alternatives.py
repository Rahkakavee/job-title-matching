import json
from collections import Counter
import pandas as pd
import re
import itertools
from src.preprocessing.preprocessing_functions import *
import pickle
from matplotlib import pyplot
import seaborn as sns
from src.descriptive.descriptive_analysis.descriptive_analysis_functions import *


# load data
with open(
    file="data/processed/data_old_format.json", mode="r", encoding="utf-8"
) as file:
    json_dict = json.load(fp=file)

# load kldbs
with open(
    file="data/raw/dictionary_occupations_complete_update.json",
    mode="r",
    encoding="utf-8",
) as file:
    kldbs = json.load(fp=file)

# get relevant data
data_old = [job for job in json_dict if "freieBezeichnung" in job.keys()]

data_old = [
    {
        "title": job["freieBezeichnung"],
        "hauptDkz": job["hauptDkz"],
        "alternativDkzs": job["alternativDkzs"],
    }
    for job in data_old
    if "alternativDkzs" in job.keys()
]

# create lookup table
kldbs_dkzs = {}
kldb_level5 = [kldb for kldb in kldbs if kldb["level"] == 5]
for kldb in kldb_level5:
    for dkz in kldb["dkzs"]:
        kldbs_dkzs.update({str(dkz["id"]): kldb["id"]})

# match with kldbs
data = []
for job in data_old:
    try:
        alternative_kldb = []
        job_kldb = kldbs_dkzs[job["hauptDkz"]]
        data.append(
            {
                "title": job["title"],
                "hauptKldB": job_kldb,
                "alternativeDkzs": job["alternativDkzs"],
            }
        )
    except:
        pass

# preprocess data
with open(file="src/preprocessing/specialwords.tex", mode="rb") as fp:
    specialwords = pickle.load(fp)
training_data = preprocess(data=data, special_words_ovr=specialwords)

# extract data according to occupations and important parts
servicekraft = [
    job
    for job in training_data
    if re.search(
        r"servicekraft|service kraft|service kräfte|servicekräfte", job["title"]
    )
]
servicekraft_6000 = [
    example for example in servicekraft if example["hauptKldB"][:1] == "6"
]

plt_servicekraft = co_occurence_with_kldbs(
    data=servicekraft_6000,
    kldbs_dkzs=kldbs_dkzs,
    title="Co-occurence for Servicekraft (only HauptkldB category 6 considered",
)
