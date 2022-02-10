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


"""Functions for limitation analysis and descriptive analysis"""


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


def co_occurence_with_kldbs(data: List, kldbs_dkzs: Dict):
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

    cmap = sns.color_palette("Blues", as_cmap=True)
    pyplot.figure(figsize=(13, 13))
    ax = sns.heatmap(df_pivot_, cmap=cmap, annot=True, annot_kws={"size": 8}, fmt="g")
    ax.invert_xaxis()
    ax.set_xlabel("Hauptkldbs")
    ax.set_ylabel("Alternativekldbs")
    return ax


def kldbs_counting(data: List, searchterms: List):
    jobs = []
    for term in searchterms:
        for job in data:
            if re.search(term, job["title"]):
                jobs.append(job)
    df = pd.DataFrame(jobs)
    df_ids = df.groupby(["id"]).count().sort_values(by=["title"], ascending=False)
    df_ids["frequency"] = df_ids["title"] / df_ids["title"].sum()
    df_ids["id"] = df_ids.index
    plot = df_ids.plot(
        x="id",
        y="frequency",
        kind="bar",
        xlabel="kldbs level 3",
        ylabel="Frequency in percent",
        legend=False,
        color="lightblue",
    )
    return plot
