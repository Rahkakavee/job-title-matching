import spacy
from spacy.kb import KnowledgeBase
import json
import os

nlp = spacy.load("de_core_news_lg")
kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)

with open("src/modeling/NEL/kb_dir/entities.jsonl", "r") as f:
    entities = [json.loads(line) for line in f]

with open("src/modeling/NEL/kb_dir/aliases.jsonl", "r") as f:
    aliases = [json.loads(line) for line in f]

for entity in entities:
    desc_doc = nlp(entity["description"])
    desc_enc = desc_doc.vector
    kb.add_entity(entity=entity["id"], entity_vector=desc_enc, freq=342)

for alias in aliases:
    kb.add_alias(
        alias=alias["id"],
        entities=alias["entities"],
        probabilities=alias["probabilities"],
    )

if not os.path.exists("src/modeling/NEL/models"):
    os.mkdir("src/modeling/NEL/models")

kb.to_disk("src/modeling/NEL/models/kb")
nlp.to_disk("src/modeling/NEL/models/nlp")
