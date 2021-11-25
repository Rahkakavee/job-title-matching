import re
import string
from typing import List


def clean_basic(data: List) -> List:
    """remove trailing and ending whitespace and convert text to lowercase

    Parameters
    ----------
    data : List
        training data

    Returns
    -------
    List
        training data cleaned
    """

    for example in data:
        example["title"] = example["title"].lower()
        example["title"] = example["title"].lstrip()
        example["title"] = example["title"].rstrip()

    return data


def remove_mwd(data: List) -> List:
    """removes mwd and wdm from text. Use before remove_special_characters

    Parameters
    ----------
    data : List
        training data

    Returns
    -------
    List
        cleaned training data
    """
    for example in data:
        example["title"] = re.sub("(m/w/d)", " ", example["title"])
        example["title"] = re.sub("(w/m/d)", " ", example["title"])

    return data


def remove_special_characters(data: List) -> List:
    """removes special characters

    Parameters
    ----------
    data : List
        training data

    Returns
    -------
    List
        cleaned training data
    """
    for example in data:
        example["title"] = re.sub("\W+", " ", example["title"])
    return data
