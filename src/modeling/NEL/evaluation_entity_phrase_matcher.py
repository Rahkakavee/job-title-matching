from src.preparation.training_data import TrainingData
from src.preparation.json_load import load_json
import spacy
from tqdm import tqdm
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language


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

terms = []
for kldb in tqdm(kldbs):
    if "searchwords" in kldb.keys():
        for searchword in kldb["searchwords"]:
            terms.append(searchword["name"])


@Language.factory("phrase_entity_matcher")
def create_phrase_matcher(nlp, name):
    """initialize EntityPhraseMatcher class"""
    return EntityPhraseMatcher(nlp, terms, "occupation")


nlp = spacy.load("src/modeling/NEL/models/nel")

tp = 0  # richtig Klasse
fp = 0  # falsche Klasse

for text, true_annot in tqdm(dataset):
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
