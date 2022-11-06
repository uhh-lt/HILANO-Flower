import argparse
import os
from pathlib import Path

import tensorflow as tf

import flwr as fl

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
import spacy
import numpy as np


#nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('de_core_news_md')
EMB_DIM = nlp.vocab.vectors_length
MAX_LEN = 50

def load_data(filename: str):
     with open(filename, 'r') as file:
         lines = [line[:-1].split() for line in file]
     samples, start = [], 0
     for end, parts in enumerate(lines):
         if not parts:
             sample = [(token, tag.split('-')[-1]) for token, tag in lines[start:end]]
             samples.append(sample)
             start = end + 1
     if start < end:
        samples.append(lines[start:end])
     return samples

def preprocess(samples):
    tag_index = {tag: index for index, tag in enumerate(schema)}
    #print (tag_index)
    X = np.zeros((len(samples), MAX_LEN, EMB_DIM), dtype=np.float32)
    #rint(X)
    y = np.zeros((len(samples), MAX_LEN), dtype=np.uint8)
    #print (y)
    vocab = nlp.vocab
    for i, sentence in enumerate(samples):
        #print (i)
        #print (sentence)
        for j, (token, tag) in enumerate(sentence[:MAX_LEN]):
            X[i, j] = vocab.get_vector(token)
            y[i,j] = tag_index[tag]
    #print (X)
    #print (y)
    return X, y


#train_samples = load_data('en_ner/client_1/conll2003.eng.train.preprocessed')
#val_samples = load_data('en_ner/client_1/conll2003.eng.testa.processed')
#test_samples = load_data('en_ner/client_1/conll2003.eng.testb.processed')

train_samples = load_data('de_ner/new/deu.train.processed')
val_samples = load_data('de_ner/new/deu.testa.processed')
test_samples = load_data('de_ner/new/deu.testb.processed')


all_samples = train_samples + val_samples + test_samples

schema = sorted({tag for sentence in all_samples for _, tag in sentence})

print (schema)


tag_index={}
tag_index = {tag: index for index, tag in enumerate(schema)}

x_train, y_train = preprocess(train_samples)
x_val, y_val = preprocess(val_samples)
x_test, y_test = preprocess(test_samples)


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        #history = self.model.fit(
        #    self.x_train,
        #    self.y_train,
        #    batch_size,
        #    epochs,
        #    validation_split=0.1,
        #)
        
        history = self.model.fit(x_train, y_train,
                        #validation_split=0.2,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size) 


        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        predictions = predict(self.model)
        print(evaluate_p(predictions))
        return loss, num_examples_test, {"accuracy": accuracy}


def predict(model):
    y_probs = model.predict(x_test)
    #print (y_probs)
    y_pred = np.argmax(y_probs, axis=-1)
    return [
        [(token, tag, schema[index]) for (token, tag), index in zip(sentence, tag_pred)]
        for sentence, tag_pred in zip(test_samples, y_pred)
    ]

def evaluate_p(predictions):
     y_t = [pos[1] for sentence in predictions for pos in sentence]
     y_p = [pos[2] for sentence in predictions for pos in sentence]
     report = classification_report(y_t, y_p, output_dict=True)
     return pd.DataFrame.from_dict(report).transpose().reset_index()


def build_model(nr_filters=256):
     input_shape = (MAX_LEN, EMB_DIM)
     lstm = LSTM(nr_filters, return_sequences=True)
     bi_lstm = Bidirectional(lstm, input_shape=input_shape)
     tag_classifier = Dense(len(schema), activation='softmax')
     sequence_labeller = TimeDistributed(tag_classifier)
     return Sequential([bi_lstm, sequence_labeller])

def main() -> None:
    # Parse command line argument `partition`
    #parser = argparse.ArgumentParser(description="Flower")
    #parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    #args = parser.parse_args()

    # Load and compile Keras model
    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10)
    model=build_model()

    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.15),
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy')

    # Load a subset of CIFAR-10 to simulate the local data partition
    #(x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[idx * 5000 : (idx + 1) * 5000],
        y_train[idx * 5000 : (idx + 1) * 5000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )


if __name__ == "__main__":
    main()
