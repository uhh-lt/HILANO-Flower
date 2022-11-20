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

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from absl import logging
import collections
from tensorflow_privacy.privacy.dp_query import gaussian_query



def make_optimizer_class(cls):
  """Constructs a DP optimizer class from an existing one."""
  parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
  child_code = cls.compute_gradients.__code__
  GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
  if child_code is not parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):
    """Differentially private subclass of given class cls."""

    _GlobalState = collections.namedtuple(
      '_GlobalState', ['l2_norm_clip', 'stddev'])
    
    def __init__(
        self,
        dp_sum_query,
        num_microbatches=None,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        dp_sum_query: DPQuery object, specifying differential privacy
          mechanism to use.
        num_microbatches: How many microbatches into which the minibatch is
          split. If None, will default to the size of the minibatch, and
          per-example gradients will be computed.
        unroll_microbatches: If true, processes microbatches within a Python
          loop instead of a tf.while_loop. Can be used if using a tf.while_loop
          raises an exception.
      """
      super(DPOptimizerClass, self).__init__(*args, **kwargs)
      self._dp_sum_query = dp_sum_query
      self._num_microbatches = num_microbatches
      self._global_state = self._dp_sum_query.initial_global_state()
      # TODO(b/122613513): Set unroll_microbatches=True to avoid this bug.
      # Beware: When num_microbatches is large (>100), enabling this parameter
      # may cause an OOM error.
      self._unroll_microbatches = unroll_microbatches

    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None,
                          gradient_tape=None,
                          curr_noise_mult=0,
                          curr_norm_clip=1):

      self._dp_sum_query = gaussian_query.GaussianSumQuery(curr_norm_clip, 
                                                           curr_norm_clip*curr_noise_mult)
      self._global_state = self._dp_sum_query.make_global_state(curr_norm_clip, 
                                                                curr_norm_clip*curr_noise_mult)
      

      # TF is running in Eager mode, check we received a vanilla tape.
      if not gradient_tape:
        raise ValueError('When in Eager mode, a tape needs to be passed.')

      vector_loss = loss()
      if self._num_microbatches is None:
        self._num_microbatches = tf.shape(input=vector_loss)[0]
      sample_state = self._dp_sum_query.initial_sample_state(var_list)
      microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
      sample_params = (self._dp_sum_query.derive_sample_params(self._global_state))

      def process_microbatch(i, sample_state):
        """Process one microbatch (record) with privacy helper."""
        microbatch_loss = tf.reduce_mean(input_tensor=tf.gather(microbatches_losses, [i]))
        grads = gradient_tape.gradient(microbatch_loss, var_list)
        sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)
        return sample_state
    
      for idx in range(self._num_microbatches):
        sample_state = process_microbatch(idx, sample_state)

      if curr_noise_mult > 0:
        grad_sums, self._global_state = (self._dp_sum_query.get_noised_result(sample_state, self._global_state))
      else:
        grad_sums = sample_state

      def normalize(v):
        return v / tf.cast(self._num_microbatches, tf.float32)

      final_grads = tf.nest.map_structure(normalize, grad_sums)
      grads_and_vars = final_grads#list(zip(final_grads, var_list))
    
      return grads_and_vars

  return DPOptimizerClass


def make_gaussian_optimizer_class(cls):
  """Constructs a DP optimizer with Gaussian averaging of updates."""

  class DPGaussianOptimizerClass(make_optimizer_class(cls)):
    """DP subclass of given class cls using Gaussian averaging."""

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        ledger=None,
        unroll_microbatches=False,
        *args,  # pylint: disable=keyword-arg-before-vararg
        **kwargs):
      dp_sum_query = gaussian_query.GaussianSumQuery(
          l2_norm_clip, l2_norm_clip * noise_multiplier)

      #if ledger:
      #  dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query,
      #                                                ledger=ledger)

      super(DPGaussianOptimizerClass, self).__init__(
          dp_sum_query,
          num_microbatches,
          unroll_microbatches,
          *args,
          **kwargs)

    @property
    def ledger(self):
      return self._dp_sum_query.ledger

  return DPGaussianOptimizerClass



nlp = spacy.load('en_core_web_lg')
#nlp = spacy.load('de_core_news_md')
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
'''
train_samples = load_data('client_1_data/train.conll')
val_samples = load_data('client_1_data/val.conll')
test_samples = load_data('client_1_data/test.conll')


all_samples = train_samples + val_samples + test_samples

schema = sorted({tag for sentence in all_samples for _, tag in sentence})

print (schema)


tag_index={}
tag_index = {tag: index for index, tag in enumerate(schema)}

x_train, y_train = preprocess(train_samples)
x_val, y_val = preprocess(val_samples)
x_test, y_test = preprocess(test_samples)
'''

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val, x_test, y_test, test_samples):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
        self.noise_multiplier = args.noise_multiplier
        self.epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.test_samples=test_samples



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
        #batch_size: int = config["batch_size"]
        #epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        #history = self.model.fit(
        #    self.x_train,
        #    self.y_train,
        #    batch_size,
        #    epochs,
        #    validation_split=0.1,
        #)
        
        history = self.model.fit(self.x_train, self.y_train,
                        #validation_split=0.2,
                        validation_data=(self.x_val, self.y_val),
                        epochs=self.epochs,
                        batch_size=self.batch_size) 


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
        predictions = predict(self.model, self.x_test, self.test_samples)
        print(evaluate_p(predictions))
        return loss, num_examples_test, {"accuracy": accuracy}


def predict(model, x_test, test_samples):
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
     global schema
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
    train_samples = load_data(args.directory+'/train.conll')
    val_samples = load_data(args.directory+'/val.conll')
    test_samples = load_data(args.directory+'/test.conll')

    global schema
    all_samples = train_samples + val_samples + test_samples

    schema = sorted({tag for sentence in all_samples for _, tag in sentence})

    print (schema)


    tag_index={}
    tag_index = {tag: index for index, tag in enumerate(schema)}

    x_train, y_train = preprocess(train_samples)
    x_val, y_val = preprocess(val_samples)
    x_test, y_test = preprocess(test_samples)


    # Load and compile Keras model
    #model = tf.keras.applications.EfficientNetB0(
    #    input_shape=(32, 32, 3), weights=None, classes=10)
    model=build_model()
    

    GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
    DPGradientDescentGaussianOptimizer_NEW = make_gaussian_optimizer_class(GradientDescentOptimizer)
    dp_optimizer = DPGradientDescentGaussianOptimizer_NEW(learning_rate=args.learning_rate, l2_norm_clip=args.l2_norm_clip,noise_multiplier=args.noise_multiplier,num_microbatches=args.microbatches)

    dp_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.losses.Reduction.NONE
            )

    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=dp_optimizer,
                  loss=dp_loss,
                  metrics='accuracy')

    # Load a subset of CIFAR-10 to simulate the local data partition
    #(x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_val, y_val, x_test, y_test, test_samples)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )
    eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(len(x_train), 1, args.noise_multiplier, args.local_epochs, (1/len(x_train)))
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)

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
    parser = argparse.ArgumentParser(description="Flower Client")

    parser.add_argument(
        "--local-epochs",
        default=5,
        type=int,
        help="Total number of local epochs to train",
    )
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--learning-rate", default=0.25, type=float, help="Learning rate for training"
    )
    # DPSGD specific arguments
    parser.add_argument(
        "--dpsgd",
        default=False,
        type=bool,
        help="If True, train with DP-SGD. If False, " "train with vanilla SGD.",
    )
    parser.add_argument(
        "--server-address",
        default="127.0.0.1:8080",
        type=str,
        help="Server IP:Port",
    )
    parser.add_argument("--l2-norm-clip", default=1.5, type=float, help="Clipping norm")
    parser.add_argument(
        "--noise-multiplier",
        default=1,
        type=float,
        help="Ratio of the standard deviation to the clipping norm",
    )
    parser.add_argument(
        "--directory", type=str, help="Directory path to train, validation and test files"
    )
    parser.add_argument(
        "--microbatches",
        default=1,
        type=int,
        help="Number of microbatches " "(must evenly divide batch_size)",
    )
    args = parser.parse_args()
    main()
