from typing import Union, Dict, List
import json


class TrainingData:
    """generates training data for classification"""

    def __init__(self, kldbs_path: str, data_path: str, kldb_level: int) -> None:
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
        kldbs_dkzs = []
        kldb_level5 = [kldb for kldb in self.kldbs if kldb["level"] == 5]
        for kldb in kldb_level5:
            for dkz in kldb["dkzs"]:
                kldbs_dkzs.append({"id": kldb["id"], "dkz": dkz["id"]})
        return kldbs_dkzs

    def extract_jobs_dkzs(self) -> List:
        """extract dkzs from jobs

        Returns
        -------
        List
            includes dictionaries with title of job and matching dkz
        """
        job_dkzs = [
            {"freieBezeichnung": job["freieBezeichnung"], "dkz": job["hauptDkz"]}
            for job in self.data
            if "freieBezeichnung" in job.keys()
        ]

        return job_dkzs

    def create_training_data(self) -> None:
        """matches dkz with kldb and append id of kldb (depending on level) with job title based on dkz"""
        job_dkzs = self.extract_jobs_dkzs()
        kldbs_dkzs = self.extract_kldbs_dkzs()
        for job_dkz in job_dkzs:
            for kldb_dkz in kldbs_dkzs:
                if job_dkz["dkz"] == str(kldb_dkz["dkz"]):
                    self.training_data.append(
                        {
                            "id": kldb_dkz["id"][: self.kldb_level],
                            "title": job_dkz["freieBezeichnung"],
                        }
                    )
