# import modules
from enum import unique
from typing import Tuple, Union, List, Dict, Any, Literal
from matplotlib import axes
from pandas.core.series import Series
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from collections import Counter
import re


def class_distribution(data: Union[Dict, List], variable: str, level: str) -> Series:
    """summarize class distribution of variable (includes plot)

    Parameters
    ----------
    data : Union[Dict, List]
        data with variable for which class distribution is required
    variable : str
        variable for which class distribution is required

    Returns
    -------
    Series
        class distribution, plot opens automatically
    """

    df = pd.DataFrame(data=data)
    class_distribution = df.groupby(variable).size()  # get frequencies
    df = df.groupby([variable]).count().reset_index()  # convert to data frame

    if level == "1":
        # draw plot
        plt.vlines(x=df["id"], ymin=0, ymax=df["title"], alpha=0.4)
        plt.scatter(df["id"], df["title"], s=1, alpha=1)
    else:
        df["level1_id"] = df["id"].astype(str).str[0]  # get level 1 variables
        conditions = [  # define conditions for grouping colors
            (df["level1_id"] == "0"),
            (df["level1_id"] == "1"),
            (df["level1_id"] == "2"),
            (df["level1_id"] == "3"),
            (df["level1_id"] == "4"),
            (df["level1_id"] == "5"),
            (df["level1_id"] == "6"),
            (df["level1_id"] == "7"),
            (df["level1_id"] == "8"),
            (df["level1_id"] == "9"),
        ]

        values = [  # define colors
            "orange",
            "green",
            "cyan",
            "blue",
            "red",
            "purple",
            "pink",
            "olive",
            "skyblue",
            "grey",
        ]

        colors = np.select(conditions, values)  # map colors to level 1 ids

        # draw plot
        plt.vlines(x=df["id"], ymin=0, ymax=df["title"], color=colors, alpha=0.4)
        plt.scatter(df["id"], df["title"], color=colors, s=1, alpha=1)
        plt.xticks([])  # hide colors

        # add legend
        mpatches_ = {}
        for i in range(0, len(values)):
            mpatches_[f"id_{i}"] = mpatches.Patch(color=values[i], label=i)
        plt.legend(handles=[mpatches_[f"id_{i}"] for i in range(0, len(mpatches_))])
    # label plot
    plt.title(f"Class Distribution Level {level}")
    plt.xlabel("KldB 2010 classes")
    plt.ylabel("Frequency")

    return class_distribution, plt


def counting_per_job(
    data: List, job_terms: List, top: int
) -> tuple[list, dict[Any, int]]:
    jobs = []
    for term in job_terms:
        for job in data:
            if re.search(term, job["title"]):
                jobs.append(job)
    df = pd.DataFrame(jobs)
    titles = df["title"]
    return jobs, dict(Counter(titles).most_common(top))


def counting_per_job_with_kldb(
    data: List, counts: dict, kldbs, title: Literal[True]
) -> List:
    countings = []
    for key, value in counts.items():
        ids = []
        for job in data:
            ids.append(job["id"])
        unique_ids = list(set(ids))
        countings.append({key: value, "ids": unique_ids})
    if title == True:
        kldbs_level_5 = [kldb for kldb in kldbs if kldb["level"] == 5]
        kldb_lookup = {}
        for kldb in kldbs_level_5:
            kldb_lookup.update({kldb["id"]: kldb["title"]})
        for example in countings:
            kldb_titles = []
            for id in example["ids"]:
                kldb_titles.append(kldb_lookup[id])
                example["ids"] = kldb_titles
    return countings


def counting_per_title(data: List, top: int) -> Dict:
    df = pd.DataFrame(data)
    titles = df["title"]
    return dict(Counter(titles).most_common(top))


def visualize_counting_per_job_with_kldb(data: Dict, title: str):
    data_new_format = []
    for key, value in data.items():
        data_new_format.append({"title": key, "counting": value})
    df = pd.DataFrame(data_new_format)
    df["counting"] = df["counting"] / df["counting"].sum()
    print(df.head(10))
    plot = df.plot(
        x="title",
        y="counting",
        kind="bar",
        title=title,
        xlabel="Job titles",
        ylabel="Frequency in percent",
        legend=False,
    )
    return plot


def counting_per_kldb_ids(data: List, job_terms: List, top: int, kldbs) -> Dict:
    jobs = []
    for term in job_terms:
        for job in data:
            if re.search(term, job["title"]):
                jobs.append(job)
    kldbs_level_5 = [kldb for kldb in kldbs if kldb["level"] == 5]
    kldb_lookup = {}
    for kldb in kldbs_level_5:
        kldb_lookup.update({kldb["id"]: kldb["title"]})
    jobs_with_kldbs = [
        {"id": job["id"], "kldb": kldb_lookup[job["id"]], "title": job["title"]}
        for job in jobs
    ]
    df = pd.DataFrame(jobs_with_kldbs)
    kldbs_for_count = df["id"]
    kldbs_count = dict(Counter(kldbs_for_count).most_common(top))
    return kldbs_count
