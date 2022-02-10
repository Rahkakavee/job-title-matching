import json
import re
from src.preprocessing.preprocessing_functions import *
import pickle
from src.descriptive.descriptive_analysis_functions import *

"""co-occurence analysis of kldbs"""

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
        kldbs_dkzs.update({str(dkz["id"]): kldb["id"][:3]})

# match with kldbs
data = []
for job in data_old:
    try:
        alternative_kldb: List = []
        job_kldb = kldbs_dkzs[job["hauptDkz"]]
        data.append(
            {
                "title": job["title"],
                "hauptKldB": job_kldb[:3],
                "alternativeDkzs": job["alternativDkzs"][:3],
            }
        )
    except:
        pass

# preprocess data
with open(file="src/preprocessing/specialwords.txt", mode="rb") as fp:
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

plt_servicekraft = co_occurence_with_kldbs(data=servicekraft, kldbs_dkzs=kldbs_dkzs)

fig_servicekraft = plt_servicekraft.get_figure()
fig_servicekraft.savefig("visualization/limitations/co_occurence_servicekraft.jpg")

maurer = [
    job for job in training_data if re.search(r"maurer|maurerin kraft", job["title"])
]

plt_maurer = co_occurence_with_kldbs(data=maurer, kldbs_dkzs=kldbs_dkzs)

fig_servicekraft = plt_servicekraft.get_figure()
fig_servicekraft.savefig("visualization/limitations/co_occurence_maurer.jpg")

elektriker = [
    job for job in training_data if re.search(r"elektriker|elektrikerin", job["title"])
]

plt_elektriker = co_occurence_with_kldbs(data=elektriker, kldbs_dkzs=kldbs_dkzs)

fig_servicekraft = plt_servicekraft.get_figure()
fig_servicekraft.savefig("visualization/limitations/co_occurence_elektriker.jpg")

softwareentwickler = [
    job
    for job in training_data
    if re.search(
        r"softwareentwickler|software entwickler",
        job["title"],
    )
]

plt_softwareentwickler = co_occurence_with_kldbs(
    data=softwareentwickler, kldbs_dkzs=kldbs_dkzs
)

fig_softwareentwickler = plt_softwareentwickler.get_figure()
fig_softwareentwickler.savefig(
    "visualization/limitations/co_occurence_softwareentwickler.jpg"
)
