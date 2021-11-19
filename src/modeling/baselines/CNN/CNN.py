from src.preparation.training_data import TrainingData
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from keras import layers
import string
import re

## Level 1
# create data
kldb_level_1 = TrainingData(
    data_path="data/processed/data_old_format.json",
    kldbs_path="data/raw/dictionary_occupations_complete_update.json",
    kldb_level=1,
    new_data=False,
)
kldb_level_1.create_training_data()

df = pd.DataFrame(data=kldb_level_1.training_data)


# preprocess
# remove punctuation
df["title"] = df["title"].apply(
    lambda x: re.sub("[%s]" % re.escape(string.punctuation), "", x)
)
df["title"] = df["title"].apply(lambda x: re.sub("mwd", "", x))
df["title"] = df["title"].str.lower()

df["id"] = df["id"].astype(int)

# get data
sentences = df["title"].values
y = df["id"].values


sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.20, random_state=100
)

# Attempt 1 with CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)


input_dim = X_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

history = model.fit(
    X_train, y_train, epochs=100, validation_data=[X_test, y_test], batch_size=1000
)

# Attempt 2 with Tokenizer
t = Tokenizer()
t.fit_on_texts(sentences_train)
t.word_index["<PAD>"] = 0
train_sequences = t.texts_to_sequences(sentences_train)
test_sequences = t.texts_to_sequences(sentences_test)
MAX_SEQUENCE_LENGTH = 1000
X_train = sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

EMBED_SIZE = 300
EPOCHS = 2
BATCH_SIZE = 128
VOCAB_SIZE = len(t.word_index)

model = Sequential()
model.add(layers.Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAX_SEQUENCE_LENGTH))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
