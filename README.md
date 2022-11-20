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

1. python server.py --server-address <ip_address:port> --test-file <directory of server_data> 
  --client-number <Minimum number of clients> ----num-rounds <Number of Federated Rounds>
2. default server-address : 127.0.0.1:8080
3. default client-number : 2 
4. default num-rounds : 2 
  
##  Client
  
  
