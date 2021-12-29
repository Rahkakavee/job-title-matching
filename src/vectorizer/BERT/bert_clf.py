import tensorflow as tf
import numpy as np
from sklearn import metrics
import transformers
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import json

"""BERT Model with one Fine tuning layer and classification. This model uses TPU for training. Run in Google Colab"""

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
val_data_size = int(0.25 * len(sentences))
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


# Source: https://towardsdatascience.com/news-category-classification-fine-tuning-roberta-on-tpus-with-tensorflow-f057c37b093
