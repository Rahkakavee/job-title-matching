from keras.models import Sequential
from keras import layers
from typing import Union, List, Dict
from numpy import array
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from keras.backend import clear_session
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from sklearn.utils import validation
from src.preparation.json_load import load_json
from src.preparation.training_data import TrainingData
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# load data
kldbs = load_json("data/raw/dictionary_occupations_complete_update.json")
jobs = load_json("data/raw/2021-10-22_12-21-00_all_jobs_7.json")

## Level 1
# create data
kldb_level_1 = TrainingData(kldbs=kldbs, data=jobs, kldb_level=1)
kldb_level_1.create_training_data()
df = pd.DataFrame(data=kldb_level_1.training_data)

data = [df["title"][row] for row in range(0, len(df))]
labels = [int(df["id"][row]) for row in range(0, len(df))]

train, test, labels_train, labels_test = train_test_split(data, labels)
