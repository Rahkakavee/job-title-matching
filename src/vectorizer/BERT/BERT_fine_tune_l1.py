from sentence_transformers import (
    SentenceTransformer,
    models,
    InputExample,
    losses,
    util,
)
import json
import random
from torch.utils.data import DataLoader
import re
from typing import List


def random_pairs(number_list: List) -> List:
    """genertes a random pair from a list

    Parameters
    ----------
    number_list : List
        list with elements from which random pairs should be generated

    Returns
    -------
    List
        with two elements (random pair)
    """
    return [number_list[i] for i in random.sample(range(len(number_list)), 2)]


# load model
word_embedding_model = models.Transformer(
    "dbmdz/bert-base-german-cased", max_seq_length=256
)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# load data
with open(
    file="/content/drive/MyDrive/MA/training_data_short_l1.json",
    mode="r",
    encoding="utf-8",
) as fp:
    data = json.load(fp=fp)

with open(
    file="/content/drive/MyDrive/MA/dictionary_occupations_complete_update.json",
    mode="r",
    encoding="utf-8",
) as fp:
    kldbs = json.load(fp=fp)

# preprocess data
data_levels: dict = {}
for el in data:
    data_levels.update({el["id"]: []})

for el in data:
    data_levels[el["id"]].append(el["title"])

kldb_level_5 = [kldb for kldb in kldbs if kldb["level"] == 5]
searchword_data = []
for kldb in kldb_level_5:
    for searchword in kldb["searchwords"]:
        if searchword["type"] == "jobtitle":
            searchword_data.append({"title": searchword["name"], "id": kldb["id"][:1]})

for searchword in searchword_data:
    searchword["title"] = searchword["title"].lower()
    searchword["title"] = searchword["title"].replace("§66 bbig/§42r hwo", "")
    searchword["title"] = re.sub("\W+", " ", searchword["title"])
    searchword["title"] = "".join(
        [char for char in searchword["title"] if not char.isdigit()]
    )
    searchword["title"] = searchword["title"].lstrip()
    searchword["title"] = searchword["title"].rstrip()

searchword_level = {}
for el in searchword_data:
    searchword_level.update({el["id"]: []})
for el in data:
    data_levels[el["id"]].append(el["title"])

kldb_level_5 = [kldb for kldb in kldbs if kldb["level"] == 5]
searchword_data = []
for kldb in kldb_level_5:
    for searchword in kldb["searchwords"]:
        if searchword["type"] == "jobtitle":
            searchword_data.append({"title": searchword["name"], "id": kldb["id"][:1]})

for searchword in searchword_data:
    searchword["title"] = searchword["title"].lower()
    searchword["title"] = searchword["title"].replace("§66 bbig/§42r hwo", "")
    searchword["title"] = re.sub("\W+", " ", searchword["title"])
    searchword["title"] = "".join(
        [char for char in searchword["title"] if not char.isdigit()]
    )
    searchword["title"] = searchword["title"].lstrip()
    searchword["title"] = searchword["title"].rstrip()

searchword_level = {}
for el in searchword_data:
    searchword_level.update({el["id"]: []})

for el in searchword_data:
    searchword_level[el["id"]].append(el["title"])
for el in searchword_data:
    searchword_level[el["id"]].append(el["title"])

similar = []
for key, value in searchword_level.items():
    for i in range(0, 3250):
        similar.append(random_pairs(value))

for key, value in data_levels.items():
    for i in range(0, 3250):
        similar.append(random_pairs(value))

keys_data = list(data_levels.keys())

unsimilar = []
for i in range(0, 1750):
    temp = []
    key_pair = random_pairs(keys_data)
    if key_pair[0] != key_pair[1]:
        temp.append(random.choice(data_levels[key_pair[0]]))
        temp.append(random.choice(data_levels[key_pair[1]]))
        unsimilar.append(temp)

keys_serarchword = list(searchword_level.keys())

for i in range(0, 1750):
    temp = []
    key_pair = random_pairs(keys_serarchword)
    if key_pair[0] != key_pair[1]:
        temp.append(random.choice(searchword_level[key_pair[0]]))
        temp.append(random.choice(searchword_level[key_pair[1]]))
        unsimilar.append(temp)

train_examples = []

for el in similar:
    train_examples.append(InputExample(texts=el, label=0.8))

for el in unsimilar:
    train_examples.append(InputExample(texts=el, label=0.2))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)

model.save("/content/drive/MyDrive/MA/fine_tuned_bert_IV")
