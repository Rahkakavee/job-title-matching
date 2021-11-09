from src.preparation.json_load import load_json
from typing import Union, List, Dict
import jsonlines
from tqdm import tqdm


def load_entities(kldbs: Union[List, Dict]) -> Union[List, Dict]:
    entities = [
        {"id": kldb["id"], "description": kldb["description"]}
        for kldb in kldbs
        if kldb["level"] == 5
    ]
    return entities


def load_aliases(kldbs: Union[List, Dict]) -> Union[List, Dict]:
    kldbs_level_5 = [kldb for kldb in kldbs if kldb["level"] == 5]

    searchwords = set()
    kldb_searchwords = []
    aliases: list = []

    for kldb in kldbs_level_5:  # loop over the klbds
        searchwords_ = [
            searchword["name"] for searchword in kldb["searchwords"]
        ]  # get searchwords per id
        for searchword in searchwords_:
            searchwords.add(searchword)  # add to set with all searchwords
        kldb_searchwords.append(
            {"id": kldb["id"], "searchwords": searchwords_}  # ids with searchwords
        )  # dict_ with id + searchword

    searchwords_list_ = list(searchwords)

    for searchword in tqdm(searchwords_list_):
        ids_ = [
            kldb_searchword["id"]
            for kldb_searchword in kldb_searchwords
            if searchword in kldb_searchword["searchwords"]
        ]
        if len(ids_) > 0:
            prob = 1 / len(ids_)
        aliases.append(
            {"id": searchword, "entities": ids_, "probabilities": [prob] * len(ids_)}
        )

    return aliases


def to_jsonl(data: Union[List, Dict], filename: str) -> None:
    with jsonlines.open(filename + ".jsonl", "w") as f:
        f.write_all(data)


if __name__ == "__main__":
    kldbs = load_json("data/raw/dictionary_occupations_complete_update.json")
    entities = load_entities(kldbs=kldbs)
    aliases = load_aliases(kldbs=kldbs)

    print(aliases[0])

    to_jsonl(entities, "src/modeling/NEL/kb_dir/entities")
    to_jsonl(aliases, "src/modeling/NEL/kb_dir/aliases")
