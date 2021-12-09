import json
import pickle
from src.preprocessing.preprocessing_functions import *
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

with open(file="data/processed/training_data_long.json") as fp:
    training_data_long = json.load(fp=fp)

with open(file="data/processed/training_data_short.json") as fp:
    training_data_short = json.load(fp=fp)

with open(file="data/raw/dictionary_occupations_complete_update.json") as fp:
    kldbs = json.load(fp=fp)

sentences_short = [job["title"] for job in training_data_short]
labels_short = [job["id"] for job in training_data_short]
sentences_long = [job["title"] for job in training_data_long]
labels_long = [job["id"] for job in training_data_long]


prior_knowledge = []
for kldb in kldbs:
    if kldb["level"] == 5:
        for searchword in kldb["searchwords"]:
            prior_knowledge.append({"title": searchword["name"]})

with open(file="src/preprocessing/specialwords.txt", mode="rb") as fp:
    specialwords = pickle.load(fp)

prior_knowledge_cleaned = preprocess(
    data=prior_knowledge, special_words_ovr=specialwords
)

sbert_model = SentenceTransformer("bert-base-nli-mean-tokens")

train, test, y_train, y_test = train_test_split(sentences_short, labels_short)

train_vec = sbert_model.encode(train)
test_vec = sbert_model.encode(test)

clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")
clf.fit(train_vec, y_train)
print(metrics.classification_report(y_test, clf.predict(test_vec), output_dict=False))


import tensorflow as tf
import numpy as np
from sklearn import metrics
import transformers
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import json

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

batch_size = 32 * tpu_strategy.num_replicas_in_sync
print("Batch size:", batch_size)
AUTOTUNE = tf.data.experimental.AUTOTUNE

with open(
    file="data/jobörse_data/training_data_short_l5.json", mode="r", encoding="utf-8"
) as fp:
    trainings_data = json.load(fp=fp)

sentences = [el["title"] for el in trainings_data]
labels = [el["id"] for el in trainings_data]

categories = sorted(list(set(labels)))  # set will return the unique different entries
n_categories = len(categories)


def indicize_labels(labels):
    """Transforms string labels into indices"""
    indices = []
    for j in range(len(labels)):
        for i in range(n_categories):
            if labels[j] == categories[i]:
                indices.append(i)
    return indices


tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")  # Tokenizer
inputs = tokenizer(
    sentences, padding=True, truncation=True, return_tensors="tf"
)  # Tokenized text

indices = indicize_labels(labels)
dataset = tf.data.Dataset.from_tensor_slices(
    (dict(inputs), indices)
)  # Create a tensorflow dataset
# train test split, we use 10% of the data for validation
val_data_size = int(0.1 * len(sentences))
val_ds = dataset.take(val_data_size).batch(batch_size, drop_remainder=True)
train_ds = dataset.skip(val_data_size).batch(batch_size, drop_remainder=True)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)


with tpu_strategy.scope():
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "bert-base-german-cased", num_labels=n_categories
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=3, name="Sparse_Top_3_Categorical_Accuracy"
            ),
        ],
    )

history = model.fit(train_ds, validation_data=val_ds, epochs=6, verbose=1)
model.save_weights("./saved_weights.h5")

trained_model = TFAutoModelForSequenceClassification.from_pretrained(
    "bert-base-german-cased", num_labels=n_categories
)
trained_model.load_weights("./saved_weights.h5")

logits = model.predict(val_ds).logits
prob = tf.nn.softmax(logits, axis=1).numpy()
predictions = np.argmax(prob, axis=1)

y = np.concatenate([y for x, y in val_ds], axis=0)

precision = metrics.precision_score(y, predictions, average="weighted")
recall = metrics.recall_score(y, predictions, average="weighted")
f1 = metrics.f1_score(y, predictions, average="weighted")
