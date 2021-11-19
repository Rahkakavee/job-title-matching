import json
from typing import Union, List, Dict
import os
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
stdout_logger = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_logger)


def json_load(path: str) -> Union[List, Dict]:
    with open(file=path, mode="r", encoding="utf-8") as fp:
        data = json.load(fp=fp)
    return data


def create_jsonfiles(path_list=List) -> Dict:
    sum = 0
    jsonfiles = [
        {"name": os.path.basename(path), "path": path, "data": None}
        for path in path_list
    ]
    for jsonfile_ in jsonfiles:
        path = jsonfile_["path"]
        name = jsonfile_["name"]
        jsonfile_["data"] = json_load(path=path)
        sum += len(jsonfile_["data"])
    return jsonfiles, sum


def append_jsons(jsonfiles: Union[List, Dict], sum: int) -> Union[Dict, List]:
    data = jsonfiles[0]["data"]
    for jsonfile in jsonfiles[1:]:
        data += jsonfile["data"]
    if sum == len(data):
        logger.debug("Data successfully merged!")
    return data


paths_old_format = [
    "data/raw/2021-07-01_09-43-57_all_jobs.json",
    "data/raw/2021-07-01_13-39-53_all_jobs.json",
    "data/raw/2021-07-01_15-57-58_all_jobs.json",
    "data/raw/2021-07-01_19-52-57_all_jobs.json",
    "data/raw/2021-07-01_21-29-55_all_jobs.json",
    "data/raw/2021-07-02_07-55-05_all_jobs.json",
    "data/raw/2021-07-02_07-55-05_all_jobs.json",
    "data/raw/2021-07-02_08-03-15_all_jobs.json",
    "data/raw/2021-08-27_17-03-57_all_jobs.json",
    "data/raw/2021-08-27_17-16-42_all_jobs.json",
    "data/raw/2021-09-07_13-40-31_all_jobs (1).json",
    "data/raw/2021-09-07_13-40-31_all_jobs.json",
    "data/raw/2021-10-07_12-08-21_all_jobs.json",
    "data/raw/2021-10-08_07-25-57_all_jobs.json",
    "data/raw/2021-10-08_11-10-04_all_jobs.json",
    "data/raw/2021-10-09_12-40-26_all_jobs.json",
    "data/raw/2021-10-11_09-21-06_all_jobs.json",
    "data/raw/2021-10-12_08-50-57_all_jobs.json",
    "data/raw/2021-10-13_08-34-31_all_jobs.json",
    "data/raw/2021-10-14_16-19-36_all_jobs.json",
    "data/raw/2021-10-22_12-21-00_all_jobs_7.json",
]

paths_new_format = [
    "data/raw/2021-11-03_18-16-58_all_jobs.json",
    "data/raw/2021-11-05_10-33-53_all_jobs.json",
    "data/raw/2021-11-15_17-32-06_all_jobs.json",
    "data/raw/2021-11-16_09-14-21_all_jobs.json",
]

jsonfiles_old_format, sum = create_jsonfiles(paths_old_format)
data_old_format = append_jsons(jsonfiles_old_format, sum)

jsonfiles_new_format, sum = create_jsonfiles(paths_new_format)
data_new_format = append_jsons(jsonfiles_new_format, sum)

with open(file="data/processed/data_old_format.json", mode="w") as fp:
    json.dump(obj=data_old_format, fp=fp)

with open(file="data/processed/data_new_format.json", mode="w") as fp:
    json.dump(obj=data_new_format, fp=fp)
