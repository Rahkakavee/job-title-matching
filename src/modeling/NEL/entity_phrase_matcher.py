from os import path
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
from src.preparation.json_load import load_json
from tqdm import tqdm
from src.preparation.training_data import TrainingData
from spacy import displacy
import json

nlp = spacy.load("de_core_news_lg", exclude="ner")


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


kldb_ontologies = load_json(path="data/raw/dictionary_occupations_complete_update.json")

terms = []

for kldb in tqdm(kldb_ontologies):
    if "searchwords" in kldb.keys():
        for searchword in kldb["searchwords"]:
            if searchword["type"] == "jobtitle":
                terms.append(searchword["name"])

terms = list(set(terms))


@Language.factory("phrase_entity_matcher")
def create_phrase_matcher(nlp, name):
    """initialize EntityPhraseMatcher class"""
    return EntityPhraseMatcher(nlp, terms, "occupation")


nlp.add_pipe("phrase_entity_matcher")


jobs = load_json(path="data/raw/2021-10-22_12-21-00_all_jobs_7.json")

kldb_level_1 = TrainingData(kldbs=kldb_ontologies, data=jobs, kldb_level=5)
kldb_level_1.create_training_data()

docs = []

for job in tqdm(kldb_level_1.training_data):
    try:
        docs.append(nlp(job["title"]))
    except:
        print(
            "[E1010] Unable to set entity information for token 0 which is included in more than one span in entities, blocked, missing or outside."
        )

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
