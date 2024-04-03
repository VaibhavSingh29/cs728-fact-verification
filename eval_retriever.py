import json
import numpy as np
import os
import pandas as pd
from pprint import pprint
from pyserini.search.lucene import LuceneSearcher
from sparse_retriever import SparseRetriever
from tqdm import tqdm
from typing import List


# create for each claim. a list of gold documents

def extract_docids(evidence_list: List):
    docids = []
    for evidence in evidence_list:
        docid = []
        for doc_data in evidence:
            for sent_data in doc_data:
                if sent_data[2] not in docid: docid.append(sent_data[2])
        docids.append(docid)
    return docids



def get_data(path: str, ner: bool):
    data = pd.read_json(path, lines=True)
    data = data[data.label != 'NOT ENOUGH INFO']
    claim = data['claim'].tolist()

    evidence = data['evidence'].tolist()
    docids = extract_docids(evidence)

    if ner:
        tags = data["tags"].tolist()
    else:
        tags = None

    pd.DataFrame({
        'claim': claim,
        'docid': docids,
    }).to_csv('val_retrieval.csv')

    return claim, docids, tags

def scorer(all_claims, all_docids, all_hits):
    assert len(all_claims) == len(all_docids) == len(all_hits)
    mrr_at_1 = 0
    mrr_at_10 = 0
    mrr_at_100 = 0

    for i in range(len(all_claims)):
        hits = all_hits[i]
        docids = all_docids[i]

        relevance = np.zeros(100, dtype=np.float32)
        for i in range(100):
            if hits[i] in docids:
                relevance[i] = 1
                break
        rr = relevance / np.arange(1,101, dtype=np.float32)

        mrr_at_1 += rr[0]
        mrr_at_10 += rr[:10].sum()
        mrr_at_100 += rr.sum()

    n = len(all_claims)
    mrr_at_1 = mrr_at_1 / n
    mrr_at_10 = mrr_at_10 / n
    mrr_at_100 = mrr_at_100 / n

    return {
        'mrr@1': mrr_at_1,
        'mrr@10': mrr_at_10,
        'mrr@100': mrr_at_100
    }
        

if __name__ == '__main__':
    ner_retriever = False
    if ner_retriever:
        searcher = SparseRetriever()
        claims, docids, tags = get_data('./data/fever_ner/shared_task_dev.jsonl', ner=True)
    else:
        searcher = LuceneSearcher('indexes/wiki-pages')
        claims, docids, _ = get_data('./data/fever/shared_task_dev.jsonl', ner=False)

    num_instances = len(claims)
    bsz = 128
    num_batches = 1+(num_instances // bsz)

    print(f'--> Evaluating {num_instances} instances.')

    def data():
        for i in range(num_batches):
            if ner_retriever:
                yield claims[i*bsz:(i+1)*bsz], docids[i*bsz:(i+1)*bsz], tags[i*bsz:(i+1)*bsz]
            else:
                yield claims[i*bsz:(i+1)*bsz], docids[i*bsz:(i+1)*bsz]

    all_hits = []
    with tqdm(total=num_batches, desc='Evaluating: ') as pbar:
        for batch in data():
            if ner_retriever:
                batch_claim, batch_docid, batch_tags = batch
                batch_hits = searcher.batch_search(
                    queries=batch_claim,
                    tags=batch_tags,
                    qids=[str(i) for i in range(len(batch_claim))],
                    k=100
                )
            else:
                batch_claim, batch_docid = batch
                batch_hits = searcher.batch_search(
                    queries=batch_claim, 
                    qids=[str(i) for i in range(len(batch_claim))], 
                    k=100
                )
            batch_hits = [
                [hit.docid for hit in hits] 
                for hits in batch_hits.values()
            ]
            all_hits.extend(batch_hits)
            pbar.update(1)

    results = scorer(claims, docids, all_hits)
    pprint(results)

    if ner_retriever:
        fpath = os.path.join('results', 'ner_retriever.json')
    else:
        fpath = os.path.join('results', 'simple_retriever.json')
    with open(fpath, 'w') as f:
        json.dump(results, f)
