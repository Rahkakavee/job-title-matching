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

# for text, true_annot in dataset:
#     print(text)
#     print(f"Gold annotation: {true_annot}")
#     doc = nlp(text)
#     for ent in doc.ents:
#         print(f"Prediction: {ent.text}, {ent.kb_id_}")

examples = []
for text, annots in dataset:
    doc = nlp.make_doc(text)
    try:
        examples.append(Example.from_dict(doc, annots))
    except:
        (
            "[E981] The offsets of the annotations for `links` could not be aligned to token boundaries."
        )

len(examples)

scores = nlp.evaluate(examples)
# print(f"ents_precision: {scores['ents_p']}")
# print(f"ents_recall: {scores['ents_r']}")
# print(f"ents_f:{scores['ents_f']}")
