# Active_Learning
Link Prediction on graph

## Clone the repository using the following command:-
git clone https://github.com/Arnabjana1999/Active_Learning.git

## Download the CORA dataset from the following link
https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

Put the 2 files cora.cites and cora.content inside Active_Learning directory

## Installing libraries
pip install torch
pip install networkx
pip install dgl

## Run the code using the following command
python train.py

The code has been written using networkx and Deep Graph Library(dgl).
After running, the code will initially output some graph statistics followed by MAP values
on the test-set after every validation interval. The code will also save
the pytorch model after every 5 epochs in the current directory.  

A notebook named active_learning.ipynb is also available