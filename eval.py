import argparse
import json
import logging
import os
import torch
import transformers
import warnings
from dataclasses import dataclass
from fever.scorer import fever_score
from loader import FeverDataset, NLICollate
from nli import NLI
from pprint import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import NLILoaderParams

parser = argparse.ArgumentParser()
parser.add_argument('--test_only', action='store_true')

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def _to_device(batch, device):
    for key in batch:
        if type(batch[key]) == torch.Tensor:
            batch[key] = batch[key].to(device)
        if type(batch[key]) == transformers.tokenization_utils_base.BatchEncoding:
            batch[key]['input_ids'] = batch[key]['input_ids'].to(device)
            batch[key]['attention_mask'] = batch[key]['attention_mask'].to(device)

def evaluate(loader, fever_ds, params, is_test=False):
    instances = []
    with tqdm(total=len(loader), desc='Evaluating: ') as pbar:
        for batch in loader:
            _to_device(batch, eval_params.device)
            predicted_label = [fever_ds.id_to_label[id] for id in nli.predict(batch).tolist()]
            predicted_evidence = batch['retrieved_evidence']
            if not is_test:
                label = [fever_ds.id_to_label[id] for id in batch['label'].tolist()]
                evidence = batch['gold_evidence_original']

                for i in range(len(predicted_label)):
                    instances.append({
                        'label': label[i],
                        'predicted_label': predicted_label[i],
                        'predicted_evidence': [[docid, lineid] for docid, lineid in predicted_evidence[i]],
                        'evidence': evidence[i]
                    })
            else:
                for i in range(len(predicted_label)):
                    instances.append({
                        'predicted_label': predicted_label[i],
                        'predicted_evidence': [[docid, lineid] for docid, lineid in predicted_evidence[i]],
                    })

            pbar.update(1)

    os.makedirs(os.path.join(params.exp_name, 'results'), exist_ok=True)
    if not is_test:
        strict_score, label_accuracy, precision, recall, f1 = fever_score(instances)   
        metrics = {
            'strict_score': strict_score,
            'label_accuracy': label_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        } 
        pprint(metrics)
        with open(os.path.join(params.exp_name, 'results', 'val_metrics.json'), 'w') as file:
            file.write(json.dumps(
                metrics,
                indent=4
            ))

    
    with open(os.path.join(params.exp_name, 'results', f"{'test' if is_test else 'val'}_predictions.json"), 'w') as file:
        for instance in instances:
            file.write(json.dumps(instance) + '\n')

@dataclass
class EvalParams():
    exp_name: str = 'runs/bm25_dpr_gpt2/'
    bsz: int = 32
    device: str = 'cuda:3'

if __name__ == '__main__':
    args = parser.parse_args()
    eval_params = EvalParams()
    loader_params = NLILoaderParams(
        dataset_type='val',
        num_docs_to_use=8,
        max_sent_per_doc=8,
        use_gold_as_evidence=False,
        max_evidence_to_retrieve=16,
        dpr_model_path=os.path.join(eval_params.exp_name, 'retriever/dpr_final.pth'),
        device='cuda:2',
    )

    # load dataset
    fever_val = FeverDataset(loader_params)
    loader_params.dataset_type = 'test'
    fever_test = FeverDataset(loader_params)

    # make dataloader
    collate_fn = NLICollate(loader_params)
    val_loader = DataLoader(fever_val, batch_size=eval_params.bsz, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(fever_test, batch_size=eval_params.bsz, collate_fn=collate_fn, shuffle=False)

    # NLI model
    nli = NLI().to(eval_params.device)
    nli.load_state_dict(torch.load(os.path.join(eval_params.exp_name, 'nli/nli_final.pth')))

    if not args.test_only: 
        print('=============================== VALIDATION ===============================')
        evaluate(val_loader, fever_val, eval_params, is_test=False)
        print('=============================== TEST ===============================')
    evaluate(test_loader, fever_test, eval_params, is_test=True)

    
    
