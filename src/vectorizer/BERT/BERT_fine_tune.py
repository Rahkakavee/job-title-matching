from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
import random

model = SentenceTransformer("src/vectorizer/BERT/all-mpnet-base-v2")

with open(file="data/processed/training_data_short_l1.json") as fp:
    training_data_short = json.load(fp=fp)

labels = list(set([el["id"] for el in training_data_short]))

training_data_sorted = {}
for label in labels:
    training_data_sorted.update({label: []})
for el in training_data_short:
    training_data_sorted[el["id"]].append(el["title"])


def random_pairs(texts: list):
    """takes a list of texts and output a pair of texts

    Parameters
    ----------
    text : list
        job title

    Returns
    -------
    [type]
        pair of list
    """
    return [texts[i] for i in random.sample(range(len(texts)), 2)]


training_data_pairs = {}
for key, value in training_data_sorted.items():
    pairs = []
    for i in range(0, 170):
        pairs.append(random_pairs(value))
    training_data_pairs.update({float(key): pairs})

train_examples = []

for key, value in training_data_pairs.items():
    for pair in value:
        train_examples.append(InputExample(texts=pair, label=key))

train_examples = [
    InputExample(texts=["My first sentence", "My second sentence"], label=0.8),
    InputExample(texts=["Another pair", "Unrelated sentence"], label=0.3),
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
