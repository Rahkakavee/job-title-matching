# import modules
from typing import Union, List, Dict, Any, Literal
from pandas.core.series import Series
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from collections import Counter
import re
from matplotlib import pyplot
import itertools


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
        kldb_lookup = {}
        for kldb in kldbs:
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


def co_occurence_with_kldbs(data: List, kldbs_dkzs: List, title: str):
    training_data_alternatives = []
    for job in data:
        alternative_kldbs = []
        for alternative_dkz in job["alternativeDkzs"]:
            try:
                alternative_dkz = kldbs_dkzs[alternative_dkz]
                alternative_kldbs.append(alternative_dkz)
            except:
                pass
        training_data_alternatives.append(
            {
                "title": job["title"],
                "hauptKldB": job["hauptKldB"],
                "alternativeKldB": list(set(alternative_kldbs)),
            }
        )

    training_data_per_alternatives = []
    for job in training_data_alternatives:
        for alternative_kldb in job["alternativeKldB"]:
            training_data_per_alternatives.append(
                {
                    "title": job["title"],
                    "hauptKldB": job["hauptKldB"],
                    "alternativeKldB": alternative_kldb,
                }
            )

    kldb_ids = []
    for example in training_data_per_alternatives:
        kldb_ids.append(example["hauptKldB"])
        kldb_ids.append(example["alternativeKldB"])
    kldb_ids = list(set(kldb_ids))

    kldb_alternatives = {}
    for kldb_id in kldb_ids:
        kldb_alternatives.update({kldb_id: []})

    for job in training_data_per_alternatives:
        kldb_alternatives[job["hauptKldB"]].append(job["alternativeKldB"])

    kldb_alternatives_countings = []
    for key, value in kldb_alternatives.items():
        values = dict(Counter(value))
        kldb_alternatives_countings.append({"kldb": key, "countings": values})

    kldb_alternatives_with_counts = []
    for example in kldb_alternatives_countings:
        for key, value in example["countings"].items():
            kldb_alternatives_with_counts.append(
                {"kldb": example["kldb"], "alternativeKldB": key, "countings": value}
            )

    combinations = []
    for kldb_id in kldb_ids:
        combinations.append(
            {"kldb": kldb_id, "alternativeKldB": kldb_id, "countings": float("nan")}
        )

    for pair in itertools.permutations(kldb_ids, 2):
        combinations.append(
            {"kldb": pair[0], "alternativeKldB": pair[1], "countings": float("nan")}
        )

    for combination in combinations:
        for example in kldb_alternatives_with_counts:
            if (
                combination["kldb"] == example["kldb"]
                and combination["alternativeKldB"] == example["alternativeKldB"]
            ):
                combination["countings"] = example["countings"]

    df = pd.DataFrame(combinations)
    df_sorted = df.sort_values(by=["alternativeKldB", "kldb"], ascending=False)
    df_pivot_ = df_sorted.pivot(
        index="alternativeKldB", columns="kldb", values="countings"
    )

    pyplot.figure(figsize=(13, 13))
    ax = sns.heatmap(
        df_pivot_, cmap="mako_r", annot=True, annot_kws={"size": 8}, fmt="g"
    )
    ax.invert_xaxis()
    ax.set_title("Co-occurence of KldBs")
    ax.set_xlabel("Hauptkldbs")
    ax.set_ylabel("Alternativekldbs")
    return ax
