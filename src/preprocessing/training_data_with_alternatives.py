import json

with open(
    file="data/processed/data_old_format.json", mode="r", encoding="utf-8"
) as file:
    json_dict = json.load(fp=file)

with open(
    file="data/raw/dictionary_occupations_complete_update.json",
    mode="r",
    encoding="utf-8",
) as file:
    kldbs = json.load(fp=file)

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

kldbs_dkzs = {}
kldb_level5 = [kldb for kldb in kldbs if kldb["level"] == 5]
for kldb in kldb_level5:
    for dkz in kldb["dkzs"]:
        kldbs_dkzs.update({str(dkz["id"]): kldb["id"]})


training_data = []
for job in data_old:
    try:
        alternative_kldb = []
        job_kldb = kldbs_dkzs[job["hauptDkz"]]
        training_data.append(
            {
                "title": job["title"],
                "hauptKldB": job_kldb,
                "alternativeDkzs": job["alternativDkzs"],
            }
        )
    except:
        pass


training_data_alternatives = []
for job in training_data:
    alternative_kldbs = []
    for alternative in job["alternativDkzs"]:
        try:
            alternative_kldb = kldbs_dkzs[alternative]
            alternative_kldbs.append(alternative_kldb)
        except:
            pass
    training_data_alternatives.append(
        {
            "title": training_data["title"],
            "hauptKldB": training_data["hauptKldB"],
            "alternativeDkzs": alternative_kldb,
        }
    )
