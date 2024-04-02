import argparse
import logging
import random
import numpy as np
import os
import torch
import warnings
from dataclasses import dataclass
from experiment import Experiment

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

parser = argparse.ArgumentParser()
parser.add_argument('--train_nli', action='store_true')
parser.add_argument('--train_retriever', action='store_true')

@dataclass
class RetrieverLoaderParams:
    dataset_type: str = 'train'
    num_positives: int = 4
    num_negatives: int = 1
    num_docs_to_retrieve: int = 8
    max_sent_per_doc: int = 8

class NLILoaderParams:
    dataset_type: str = 'train'
    num_docs_to_use: int = 8
    max_sent_per_doc: int = 8
    max_evidence_length: int = 64
    use_gold_as_evidence: bool = True
    dpr_model_path: str = ''
    max_docs_to_retrieve: int = 10


# @dataclass
# class DatasetParams:
#     num_passages: int = 16
#     k: int = 8

# @dataclass
# class DataParams:
#     bsz: int = 32
#     train: DatasetParams = DatasetParams(num_passages = 16, k = 8)
#     val: DatasetParams = DatasetParams(num_passages=16, k=8)

# @dataclass
# class ModelParams:
#     num_passages: int = 16
#     bsz: int = 32

@dataclass
class TrainingParams:
    num_epochs: int = 20
    seed: int = 42
    bsz: int = 32
    lr: float = 1e-5
    log_dir: str = './logs/bm25_dpr_gpt2/'
    es_step: int = 4
    exp_name: str = 'bm25_dpr_gpt2'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    args = parser.parse_args()
    assert (args.train_retriever ^ args.train_nli), f'Can train only one at a time!'


    if args.train_retriever:
# set_seed(training_params.seed)
        retriever_loader_params = RetrieverLoaderParams()
        training_params = TrainingParams()
        os.makedirs(os.path.join('runs', training_params.exp_name, 'retriever'), exist_ok=True)  
        exp = Experiment(retriever_loader_params, training_params, retriever_exp=True)
        
        exp.train()
    
    if args.train_nli:
        nli_loader_params = NLILoaderParams()
        training_params = TrainingParams()
        os.makedirs(os.path.join('runs', training_params.exp_name, 'nli'), exist_ok=True)

        exp = Experiment(nli_loader_params, training_params, retriever_exp=False)

        exp.train()
