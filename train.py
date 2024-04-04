import argparse
import logging
import random
import numpy as np
import os
import torch
import warnings
from accelerate.utils import tqdm
from dataclasses import dataclass
from experiment import Experiment
from loader import NLICollate, FeverDataset
from torch.utils.data import DataLoader


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
    use_ner: bool = True

@dataclass
class NLILoaderParams:
    dataset_type: str = 'train'
    num_docs_to_use: int = 4
    max_sent_per_doc: int = 4
    max_evidence_length: int = 64
    use_gold_as_evidence: bool = True
    dpr_model_path: str = './runs/bm25_dpr_gpt2/retriever/dpr_final.pth'
    max_evidence_to_retrieve: int = 5
    truncate_evidence: bool = True
    device: str = 'cuda:6'
    use_ner: bool = True


@dataclass
class TrainingParams:
    num_epochs: int = 4
    seed: int = 42
    bsz: int = 16
    lr: float = 1e-5
    log_dir: str = './logs/bm25_dpr_gpt2/'
    es_step: int = 2
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
