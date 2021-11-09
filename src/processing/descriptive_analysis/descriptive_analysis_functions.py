# import modules
from typing import Union, List, Dict
from pandas.core.series import Series
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches


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
