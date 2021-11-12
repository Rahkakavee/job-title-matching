# from spacy.training import Example
from typing import List
from src.preparation.training_data import TrainingData
import spacy
from spacy.kb import KnowledgeBase
import random
from spacy.util import minibatch, compounding
from tqdm import tqdm
from spacy.training import Example
from spacy.ml.models import load_kb


## Level 5
# create data
kldb_level_5 = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/raw/2021-10-22_12-21-00_all_jobs_7.json",
    kldb_level=5,
)
kldb_level_5.create_training_data()

nlp = spacy.load("src/modeling/NEL/models/nlp")  # load trained nlp model
kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)  # load knowledgebase


kb.from_disk("src/modeling/NEL/models/kb")

dataset: list = []  # create trainingdata

for entry in kldb_level_5.training_data:
    text = entry["title"]
    offset = (0, len(entry["title"]))
    id = entry["id"]
    links_dict = {id: 1.0}
    entity_label = "occupation"
    entities = [(offset[0], offset[1], entity_label)]
    dataset.append((text, {"links": {offset: links_dict}, "entities": entities}))

gold_ids = []  # create ids
for text, annot in dataset:
    for span, links_dict in annot["links"].items():
        for link, value in links_dict.items():
            if value:
                gold_ids.append(link)

# split into test and training dataset
ids = list(set(kb.get_entity_strings()))  # ids for checking
train: List = []
test: List = []
for id in ids:
    indices = [i for i, j in enumerate(gold_ids) if j == id]
    train.extend(dataset[index] for index in indices[0:8])  # first 8 in training
    test.extend(dataset[index] for index in indices[8:10])

# # shuffle
random.shuffle(train)
random.shuffle(test)

TRAIN_EXAMPLES = []
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
sentencizer = nlp.get_pipe("sentencizer")
for text, annotation in train:
    example = Example.from_dict(nlp.make_doc(text), annotation)
    example.reference = sentencizer(example.reference)
    TRAIN_EXAMPLES.append(example)

# set up pipeline
entity_linker = nlp.add_pipe("entity_linker", config={"incl_prior": False}, last=True)
entity_linker.initialize(
    get_examples=lambda: TRAIN_EXAMPLES, kb_loader=load_kb("src/modeling/NEL/models/kb")
)

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]

with nlp.select_pipes(enable=["entity_linker"]):  # train only the entity_linker
    optimizer = nlp.resume_training()
    for itn in tqdm(range(500)):
        random.shuffle(TRAIN_EXAMPLES)
        batches = minibatch(
            TRAIN_EXAMPLES, size=compounding(4.0, 32.0, 1.001)
        )  # increasing batch sizes
        losses = {}
        for batch in batches:
            nlp.update(
                batch,
                drop=0.2,  # prevent overfitting
                losses=losses,
                sgd=optimizer,
            )
        if itn % 50 == 0:
            print(itn, "Losses", losses)  # print the training loss
print(itn, "Losses", losses)

nlp.to_disk("src/modeling/NEL/models/nlp_NEL")
