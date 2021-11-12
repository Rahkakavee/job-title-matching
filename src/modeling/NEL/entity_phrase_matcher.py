import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
from tqdm import tqdm
from src.preparation.training_data import TrainingData
from typing import List
from spacy.kb import KnowledgeBase
import random
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.ml.models import load_kb
import jsonlines


## define Phrase Matcher
class EntityPhraseMatcher(object):
    """pipline component for PhraseMatching

    Parameters
    ----------
    object : None
        None

    Returns
    -------
    doc object
        creates doc object and call the individual pipeline components on the Doc order
    """

    name = "phrase_entity_matcher"

    def __init__(self, nlp, terms, label):
        patterns = [nlp(term) for term in terms]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add(label, None, *patterns)

    def __call__(self, doc):
        matches = self.matcher(doc)
        # filter matches
        filtered_matches = []
        current_start = -1
        current_stop = -1
        for label, start, stop in sorted(matches):
            if start > current_stop:
                # this segment starts after the last segment stops
                # just add a new segment
                filtered_matches.append((start, stop, label))
                current_start, current_stop = start, stop
            else:
                # segments overlap, replace
                filtered_matches[-1] = (current_start, stop, label)
                # current_start already guaranteed to be lower
                current_stop = max(current_stop, stop)
        # convert to spans
        spans = []
        for start, end, label in filtered_matches:
            span = Span(doc, start, end, label=label)
            spans.append(span)

        # add to docs
        doc.ents = list(doc.ents) + spans

        # filtered = filter_spans(spans)
        # doc.ents = list(doc.ents) + filtered
        return doc


## load nlp model
nlp = spacy.load("src/modeling/NEL/models/nlp", exclude="ner")  # load trained nlp model


## add terms for PhraseMatcher
# load data
kldb_level_1 = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/raw/2021-10-22_12-21-00_all_jobs_7.json",
    kldb_level=5,
)

# add terms
terms = []
for kldb in tqdm(kldb_level_1.kldbs):
    if "searchwords" in kldb.keys():
        for searchword in kldb["searchwords"]:
            terms.append(searchword["name"])
# remove duplicates
terms = list(set(terms))

## set up PhraseMatcher
@Language.factory("phrase_entity_matcher")
def create_phrase_matcher(nlp, name):
    """initialize EntityPhraseMatcher class"""
    return EntityPhraseMatcher(nlp, terms, "occupation")


## add to pipeline
nlp.add_pipe("phrase_entity_matcher")

## train PhraseMatcher
# load data

kldb_level_1 = TrainingData(
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    data_path="data/raw/2021-10-22_12-21-00_all_jobs_7.json",
    kldb_level=5,
)
kldb_level_1.create_training_data()
docs = []
# apply data
for job in tqdm(kldb_level_1.training_data):
    try:
        docs.append(nlp(job["title"]))
    except:
        print(
            "[E1010] Unable to set entity information for token 0 which is included in more than one span in entities, blocked, missing or outside."
        )


## preprocess data for entity linking
dataset = []
for doc in tqdm(docs):
    if len(doc.ents) > 0:
        for ent in doc.ents:
            text = doc.text
            offset = (ent.start_char, ent.end_char)
            entity_label = ent.label_
            for entry in kldb_level_1.training_data:
                if entry["title"] == doc.text:
                    id = entry["id"]
                    links_dict = {id: 1.0}
                    entities = [(offset[0], offset[1], entity_label)]
                    dataset.append(
                        (text, {"links": {offset: links_dict}, "entities": entities})
                    )

## load knowledgebase
kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)  # load knowledgebase
kb.from_disk("src/modeling/NEL/models/kb")

## create golden_ids
gold_ids = []  # create ids
for text, annot in dataset:
    for span, links_dict in annot["links"].items():
        for link, value in links_dict.items():
            if value:
                gold_ids.append(link)

## split into test and training dataset
ids = list(set(kb.get_entity_strings()))  # ids for checking
train: List = []
test: List = []
for id in ids:
    indices = [i for i, j in enumerate(gold_ids) if j == id]
    train.extend(dataset[index] for index in indices[0:8])  # first 8 in training
    test.extend(dataset[index] for index in indices[8:10])

## shuffle train and test
random.shuffle(train)
random.shuffle(test)

## create train examples
TRAIN_EXAMPLES = []
if "parser" not in nlp.pipe_names:
    nlp.add_pipe("parser")
sentencizer = nlp.get_pipe("parser")
for text, annotation in train:
    example = Example.from_dict(nlp.make_doc(text), annotation)
    example.reference = sentencizer(example.reference)
    TRAIN_EXAMPLES.append(example)

## add entity linker
entity_linker = nlp.add_pipe("entity_linker", config={"incl_prior": False}, last=True)
entity_linker.initialize(
    get_examples=lambda: TRAIN_EXAMPLES, kb_loader=load_kb("src/modeling/NEL/models/kb")
)

# other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "entity_linker"]

## train entity linker
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

# export NEL model
nlp.to_disk("src/modeling/NEL/models")

examples = []
for text, annots in test:
    doc = nlp.make_doc(text)
    examples.append(Example.from_dict(doc, annots))

scores = nlp.evaluate(examples)
print(f"entity_linker_performance: {scores}")


tp = 0  # richtig Klasse
fp = 0  # falsche Klasse

for text, true_annot in test:
    doc = nlp(text)
    if len(doc.ents) == 1:
        links = true_annot["links"]
        ids = links[list(true_annot["links"].keys())[0]]
        id = list(ids.keys())[0]
        if id == doc.ents[0].kb_id_:
            tp += 1
        if id != doc.ents[0].kb_id_:
            fp += 1

precision = tp / (tp + fp)

print(f"precision: {precision}")


results = []

for text, true_annot in test:
    doc = nlp(text)
    links = true_annot["links"]
    ids = links[list(true_annot["links"].keys())[0]]
    id = list(ids.keys())[0]
    print(f"title: {text}, id: {id}")
    predictions = [{"entity_text": ent.text, "id": ent.kb_id_} for ent in doc.ents]
    results.append(
        {"Gold annotation": {"text": text, "id": id}, "predictions": predictions}
    )

with jsonlines.open("src/modeling/NEL/results_2.jsonl", "w") as f:
    f.write_all(results)
