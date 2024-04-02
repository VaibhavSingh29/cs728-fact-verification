# Fact Verification using Two Stage Retriever with GPT2 

To run the code:
1. `bash download.sh`
2. `python gen_docs.py`
3. `bash index.sh`

We are all set to train the second stage retriever and the NLI model. 

To train the retriever: `accelerate launch train.py --train_retriever`

To train the NLI model (GPT2): `accelerate launch train.py --train_nli`

To create environment:

* install latest version of transformers, pytorch 
* install tensorboard
* install faiss-cpu, pyserini 
    [**Note**: We encountered an error when installing pyserini due to the package nmslib. Highly recommending to run: `conda install -c conda-forge nmslib -y` before installing pyserini]