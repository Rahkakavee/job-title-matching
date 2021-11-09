import json
from typing import Union, Dict, List


def load_json(path: str) -> Union[Dict, List]:
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
