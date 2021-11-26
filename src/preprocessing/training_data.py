from typing import Union, Dict, List
import json
from tqdm import tqdm

from src.logger import logger


class TrainingData:
    """generates training data for classification"""

    def __init__(
        self, kldbs_path: str, data_path: str, kldb_level: int, new_data: bool
    ) -> None:
        """init method

        Parameters
        ----------
        kldbs : Union[Dict, List]
            data with kldbs for matching
        data : Union[Dict, List]
            data with jobs
        kldb_level : int
            level of analysis (1-5)
        """
        self.kldbs = self.load_json(kldbs_path)
        self.data = self.load_json(data_path)
        self.kldb_level = kldb_level
        self.new_data = new_data
        self.training_data = []

    def load_json(self, path: str) -> Union[Dict, List]:
        """read and load json data

        Parameters
        ----------
        path : str
            path with json data

        Returns
        -------
        Union[Dict, List]
            list with dictionaries
        """
        with open(file=path, mode="r", encoding="utf-8") as file:
            json_dict = json.load(fp=file)
        return json_dict

    def extract_kldbs_dkzs(self) -> List:
        """extract dkzs from kldbs

        Returns
        -------
        List
            includes dictionaries with id of kldb and matching dkz
        """
        kldbs_dkzs = {}
        kldb_level5 = [kldb for kldb in self.kldbs if kldb["level"] == 5]
        if self.new_data == False:
            for kldb in kldb_level5:
                for dkz in kldb["dkzs"]:
                    kldbs_dkzs.update({str(dkz["id"]): kldb["id"]})

        if self.new_data == True:
            for kldb in kldb_level5:
                for dkz in kldb["dkzs"]:
                    kldbs_dkzs.update({dkz["title"]: kldb["id"]})
        return kldbs_dkzs

    def extract_jobs_dkzs(self) -> List:
        """extract dkzs from jobs

        Returns
        -------
        List
            includes dictionaries with title of job and matching dkz
        """
        if self.new_data == False:
            job_dkzs = [
                {"freieBezeichnung": job["freieBezeichnung"], "dkz": job["hauptDkz"]}
                for job in self.data
                if "freieBezeichnung" in job.keys()
            ]

        if self.new_data == True:
            job_dkzs = [
                {"freieBezeichnung": job["titel"], "dkz": job["beruf"]}
                for job in self.data
                if "titel" in job.keys()
            ]

        return job_dkzs

    def create_training_data(self) -> None:
        """matches dkz with kldb and append id of kldb (depending on level) with job title based on dkz"""
        job_dkzs = self.extract_jobs_dkzs()
        kldbs_dkzs = self.extract_kldbs_dkzs()
        for job_dkz in tqdm(job_dkzs):
            dkz = job_dkz["dkz"]
            try:
                self.training_data.append(
                    {
                        "id": kldbs_dkzs[dkz][: self.kldb_level],
                        "title": job_dkz["freieBezeichnung"],
                    }
                )
            except KeyError as e:
                continue
        logger.debug(
            f"{len(job_dkzs) - len(self.training_data)} cannot be assigned to any kldb class. Skipped!"
        )
