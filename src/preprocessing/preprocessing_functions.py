from os import remove
import re
from typing import List, Literal
import emoji
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
from src.logger import logger
from tqdm import tqdm


def remove_special_characters(data: List) -> List:
    """removes special characters and numbers

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
        example["title"] = "".join(
            [char for char in example["title"] if not char.isdigit()]
        )
    return data


def remove_emojis(data: List) -> List:
    """remove emojis from text

    Parameters
    ----------
    data : List
        trainin data

    Returns
    -------
    List
        training data cleaned
    """
    for example in tqdm(data):
        example["title"] = re.sub(emoji.get_emoji_regexp(), "", example["title"])
    return data


def remove_lc_ws(data: List) -> List:
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


def remove_stopwords(data: List) -> List:
    german_stop_words = stopwords.words("german")
    for example in data:
        title_tokens = example["title"].split()
        title_tokens_cleaned = [
            title_token
            for title_token in title_tokens
            if title_token.lower() not in german_stop_words
        ]
        example["title"] = " ".join(title_tokens_cleaned)
    return data


def remove_special_words(data: List, specialwords: List = ["m", "w", "d", "f"]) -> List:
    for example in data:
        title_tokens = example["title"].split()
        title_tokens_cleaned = [
            title_token
            for title_token in title_tokens
            if title_token.lower() not in specialwords
        ]
        example["title"] = " ".join(title_tokens_cleaned)
    return data


def preprocess(
    data: List,
    special_characters: bool = True,
    emojis: bool = True,
    lowercase_whitespace: bool = True,
    stopwords: bool = True,
    special_words: bool = True,
    special_words_ovr: List = [],
) -> List:
    if special_characters:
        logger.debug("Remove special characters and numbers")
        data = remove_special_characters(data=data)
    if emojis:
        logger.debug("Remove emojis")
        data = remove_emojis(data=data)
    if lowercase_whitespace:
        logger.debug("Convert to lowercase")
        data = remove_lc_ws(data=data)
    if stopwords:
        logger.debug("Remove stopwords")
        data = remove_stopwords(data=data)
    if special_words:
        logger.debug("Remove special words")
        data = remove_special_words(data=data)
    if special_words and len(special_words_ovr) > 0:
        logger.debug("Remove special words")
        data = remove_special_words(data=data, specialwords=special_words_ovr)
    return data
