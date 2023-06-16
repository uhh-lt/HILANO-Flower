# HILANO
The Purpose of this project is Anonymization with Distributed Human in the Loop Learning. 
Since anonymization is a sequence tagging task, we have taken up Named Entity Recognition model to experiment with. 

# Priliminary Study
The priliminary analysis with our prototype is presented in the Proceedings of the Third Workshop on Privacy in Natural Language Processing (2021). Here is the link to the paper -  https://aclanthology.org/2021.privatenlp-1.4.pdf  .
The comprehensive analysis paper is coming soon, stay tuned. 


# Implementation details
1. We have used state-of-the-art NER model.
2. We have used FLower Framework (https://github.com/adap/flower) for our study to incorporate Federated Learning.
3. We have used tensorflow-privacy (https://github.com/tensorflow/privacy) to incorporate Differential Privacy.
4. In this repository we present a working framework of our system. Please follow the instructions below.

# Setting Up Environment (For both Server and Client)
1. python3 -m venv 'sample_env'. [can choose any environment name]
2. source sample_env/bin/activate
3. pip install --upgrade pip
4. pip install pipx
5. pipx install poetry==1.2.0
6. pipx ensure path
7. poetry shell
8. pip install tensorflow
9. pip install tensorflow-privacy
10. pip install spacy
11. pip install flwr
12. python -m spacy download en_core_web_lg [for English]

# Contents
1. server_data - This directory contains one file (test.conll) to evaluate the performance of aggregated model on server side.
2. client-1-data - This directory contains three files (train.conll, val.conll, test.conll) to train and evaluate the performance of model on one of the client side.
3. client-2-data - This directory contains three files (train.conll, val.conll, test.conll) to train and evaluate the performance of model on one of the client side.
4. server.py- for server instance
5. client.py- for client instance

# Execution
## Server

1. python server.py --server-address [ip_address:port] --test-file [directory of server_data] 
  --client-number [Minimum number of clients] --num-rounds [Number of Federated Rounds]
2. default server-address : 127.0.0.1:8080
3. default client-number : 2 
4. default num-rounds : 2 
5. Have to specify the test-file
  
##  Client
  
 1. python client.py --server-address [ip_address:port] --directory [directory of client_data] --batch-size [batch size for training] --local-epochs [epochs for training] --learning-rate [learning rate for training] --noise-multiplier [noise multiplier for differential privacy] --l2-norm-clip [l2 normalization clippping factor for differential privacy] --microbatches [number of microbatches for differential privacy]
2. default server-address : 127.0.0.1:8080
3. default batch-size : 32 
4. default local-epochs : 5
5. default learning-rate : 0.25
6. default noise-multiplier : 1
7. default l2-norm-clip : 1.5
8. default microbatches : 1
9. Have to specify the directory
