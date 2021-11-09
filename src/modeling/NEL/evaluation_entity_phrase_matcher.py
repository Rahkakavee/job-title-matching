from src.preparation.training_data import TrainingData
from src.preparation.json_load import load_json
import spacy
from spacy.training import Example


# load data
kldbs = load_json("data/raw/dictionary_occupations_complete_update.json")
jobs = load_json("data/raw/2021-09-07_13-40-31_all_jobs (1).json")

## Level 1
# create data
kldb_level_5 = TrainingData(kldbs=kldbs, data=jobs, kldb_level=5)
kldb_level_5.create_training_data()

dataset: list = []

for entry in kldb_level_5.training_data:
    text = entry["title"]
    offset = (0, len(entry["title"]))
    QID = entry["id"]
    links_dict = {QID: 1.0}
    entity_label = entry["id"]
    entities = [(offset[0], offset[1], entity_label)]
    dataset.append((text, {"links": {offset: links_dict}, "entities": entities}))

nlp = spacy.load("src/modeling/NEL/models/nlp_NEL")


# examples = []
# for text, annots in dataset:
#     doc = nlp.make_doc(text)
#     examples.append(Example.from_dict(doc, annots))

# scores = nlp.evaluate(examples)
# print(f"entity_linker_performance: {scores}")

tp = 0  # richtige predicted
fn = 0  # keine ents
fp = 0  # prediction falsch

for text, true_annot in dataset:
    doc = nlp(text)
    if len(doc.ents) == 0:
        fn += 1
    if len(doc.ents) != 0:
        links = true_annot["links"]
        ids = links[list(true_annot["links"].keys())[0]]
        id = list(ids.keys())[0]
        doc = nlp(text)
        for ent in doc.ents:
            if id == ent.kb_id_:
                tp += 1
            if id == ent.kb_id_:
                fp += 1

# TODO: Nochmal überprüfen, ob das wirklich Sinn macht
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f"precision: {precision}")
print(f"recall: {recall}")
