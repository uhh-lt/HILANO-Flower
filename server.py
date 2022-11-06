from typing import Dict, Optional, Tuple
from pathlib import Path

import flwr as fl
import tensorflow as tf

import pandas as pd 
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
import spacy
import numpy as np


nlp = spacy.load('en_core_web_lg')
EMB_DIM = nlp.vocab.vectors_length
MAX_LEN = 50

def load_data(filename: str):
     with open(filename, 'r') as file:
         lines = [line[:-1].split() for line in file]
     samples, start = [], 0
     for end, parts in enumerate(lines):
         if not parts:
             sample = [(token, tag.split('-')[-1]) for token, tag in lines[start:end]]
             #print (sample)
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


test_samples = load_data('en_ner/server/conll2003.eng.testb.processed')
#print (test_samples)
schema = sorted({tag for sentence in test_samples for _, tag in sentence})
tag_index={}
tag_index = {tag: index for index, tag in enumerate(schema)}
#test_samples = load_data('en_ner/conll2003.eng.testb.processed')

print (schema)
#schema = sorted({tag for sentence in test_samples for _, tag in sentence})

X_test, y_test = preprocess(test_samples)



def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10
    #)
    
    
    #schema = sorted({tag for sentence in all_samples for _, tag in sentence})
    #tag_index={}
    #tag_index = {tag: index for index, tag in enumerate(schema)}
    #test_samples = load_data('en_ner/conll2003.eng.testb.processed')
    #X_test, y_test = preprocess(test_samples)


    model= build_model()
	
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.15),
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy')    



    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        #server_address="0.0.0.0:8080",
	#server_address="127.0.0.1:8080"
        server_address="[::]:4466",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
   #     certificates=(
   #         Path(".cache/certificates/ca.crt").read_bytes(),
   #         Path(".cache/certificates/server.pem").read_bytes(),
   #        Path(".cache/certificates/server.key").read_bytes(),
   #     ),
    )


def build_model(nr_filters=256):
     input_shape = (MAX_LEN, EMB_DIM)
     lstm = LSTM(nr_filters, return_sequences=True)
     bi_lstm = Bidirectional(lstm, input_shape=input_shape)
     tag_classifier = Dense(len(schema), activation='softmax')
     sequence_labeller = TimeDistributed(tag_classifier)
     return Sequential([bi_lstm, sequence_labeller])

'''
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

'''

def predict(model):
    y_probs = model.predict(X_test)
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


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    #(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    # Use the last 5k training examples as a validation set
    #x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    #schema = sorted({tag for sentence in all_samples for _, tag in sentence})
    #tag_index={}
    #tag_index = {tag: index for index, tag in enumerate(schema)}
    #test_samples = load_data('en_ner/conll2003.eng.testb.processed')
    #x_test, y_test = preprocess(test_samples)
    

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        predictions = predict(model)
        print(evaluate_p(predictions))
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, {"accuracy": accuracy}
        #predictions = predict(model)
        #return evaluate_p(predictions)

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
