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

train_samples = load_data('en_ner/conll2003.eng.train.preprocessed')
val_samples = load_data('en_ner/conll2003.eng.testa.processed')
test_samples = load_data('en_ner/conll2003.eng.testb.processed')
all_samples = train_samples + val_samples + test_samples

#print (train_samples [0])
#print (train_samples [1])




schema = sorted({tag for sentence in all_samples for _, tag in sentence})

print (schema)


import spacy
import numpy as np
nlp = spacy.load('en_core_web_lg')
EMB_DIM = nlp.vocab.vectors_length
print (EMB_DIM)
MAX_LEN = 50

tag_index={}
tag_index = {tag: index for index, tag in enumerate(schema)}

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

X_train, y_train = preprocess(train_samples)
X_val, y_val = preprocess(val_samples)
X_test, y_test = preprocess(test_samples)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
import tensorflow as tf

def build_model(nr_filters=256):
     input_shape = (MAX_LEN, EMB_DIM)
     lstm = LSTM(nr_filters, return_sequences=True)
     bi_lstm = Bidirectional(lstm, input_shape=input_shape)
     tag_classifier = Dense(len(schema), activation='softmax')
     sequence_labeller = TimeDistributed(tag_classifier)
     return Sequential([bi_lstm, sequence_labeller])

model = build_model()

def train(model, epochs=10, batch_size=32):
    #model.compile(optimizer='Adam',
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.15),
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy')
    history = model.fit(X_train, y_train,
                        #validation_split=0.2,
                        validation_data=(X_val, y_val),			
                        epochs=epochs,
                        batch_size=batch_size)
    return history.history

history = train(model)


def predict(model):
    y_probs = model.predict(X_test)
    #print (y_probs)
    y_pred = np.argmax(y_probs, axis=-1)
    return [
        [(token, tag, schema[index]) for (token, tag), index in zip(sentence, tag_pred)]
        for sentence, tag_pred in zip(test_samples, y_pred)
    ]

predictions = predict(model)


import pandas as pd
from sklearn.metrics import classification_report

def evaluate(predictions):
     y_t = [pos[1] for sentence in predictions for pos in sentence]
     y_p = [pos[2] for sentence in predictions for pos in sentence]
     report = classification_report(y_t, y_p, output_dict=True)
     return pd.DataFrame.from_dict(report).transpose().reset_index()
 
print (evaluate(predictions))

