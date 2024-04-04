from train_retriever import Retriever
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import faiss
from pyserini.search.lucene import LuceneSearcher
import pandas as pd
from tqdm import tqdm
from whoosh import qparser
from whoosh.qparser import QueryParser
from whoosh.query import *
from pathlib import Path
import numpy as np
from whoosh.index import open_dir
from collections import Counter
import random
from sklearn.metrics import accuracy_score
from torch.nn.functional import cosine_similarity
import json
from sparse_ret import SparseRetriever

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU
        
        
        
def search_sparse_index(index, qry_collection, k=None):
    
    searcher = LuceneSearcher('save/indexes/wiki-pages/')
    model_mapping= {}   
    for qry_id in tqdm(qry_collection, desc='Sparse Retrieval'):
        query = qry_collection[qry_id]
        hits = searcher.search(query, k=k)
        doc_list = []
        sentence_list = []
        for result in hits:
            sentences = json.loads(searcher.doc(result.docid).raw())['sentences']
            sentence_list.extend(sentences)
            doc_list.extend([result.docid] * len(sentences))
        
        model_mapping[qry_id] = (doc_list, sentence_list)
    
    return model_mapping

def search_ner_sparse_index(index, qry_collection, k=None):
    
    searcher = SparseRetriever()
    model_mapping= {}   
    for qry_id in tqdm(qry_collection, desc='Sparse Retrieval'):
        query, tags = qry_collection[qry_id]
        hits = searcher.search(query, tags, k=k)
        doc_list = []
        sentence_list = []
        for result in hits:
            sentences = json.loads(searcher.doc(result.docid).raw())['sentences']
            sentence_list.extend(sentences)
            doc_list.extend([result.docid] * len(sentences))
        
        model_mapping[qry_id] = (doc_list, sentence_list)
    
    return model_mapping

def get_embedding(text, tokenizer, model, device):
    
    inputs = tokenizer(text, return_tensors='pt', return_attention_mask=True, max_length=128, padding='max_length', truncation=True)
    inputs.to(device)
    inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
            }
    model.eval()
    with torch.no_grad():
        embeddings = model(**inputs)
        embeddings = embeddings.detach().cpu().numpy()
    return [embeddings[idx, :] for idx in range(embeddings.shape[0])]

def create_mips_index(dim, use_gpu=True):
    index = faiss.IndexFlatIP(dim)
    # if use_gpu:
    #     gpu_index = faiss.index_cpu_to_all_gpus(index)
    return index

def get_nli_encodings(inputs, tokenizer, device):
    encoding = tokenizer(
            inputs,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
    encoding.to(device)

    return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
def aggregate_evidence_with_indices(evidences):
    
    verdict_counts = Counter(evidences)
    majority_verdict, _ = verdict_counts.most_common(1)[0]
    majority_indices = [index for index, verdict in enumerate(evidences) if verdict == majority_verdict]
    
    return majority_verdict, majority_indices

def main():
    
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'bert-base-uncased'
    retriever_model_path = 'save/retrieval_model.pth'
    nli_model_path = 'save/nli_model.pth'

    retriever_model = Retriever(MODEL_NAME=MODEL_NAME, return_dense_vector=True)
    retriever_model.load_state_dict(torch.load(retriever_model_path))
    retriever_model.to(device)
    
    nli_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    nli_model.load_state_dict(torch.load(nli_model_path))
    nli_model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    wiki_doc_to_lines = pd.read_pickle('save/wiki_mapping.pkl')
    query_file_path = 'data//fever_ner/shared_task_test.jsonl'
    query_df = pd.read_json(query_file_path, lines=True)
    query_temp_df = query_df.set_index('id')
    qry_collection = query_temp_df.apply(lambda x: (x['claim'], x['tags']), axis=1).to_dict()
    values = list(qry_collection.values())
    claims, _ = zip(*values)
    claims = list(claims)
    claim_embeddings = []
    for idx in tqdm(range(0, len(claims), 32), desc='Claim embeddings'):
        text = claims[idx: idx + 32]
        embeds = get_embedding(text=text, tokenizer=tokenizer, model=retriever_model, device=device)
        claim_embeddings.extend(embeds)
        
    bm25_doc_mapping = search_ner_sparse_index(index=None, qry_collection=qry_collection, k=1)
    claim_result_dict = {}
    output_predictions = []

    for idx, claim_key in enumerate(tqdm(bm25_doc_mapping, desc='Dense Retrieval')):
        retrieved_doc_list, retrieved_sentence_list  = bm25_doc_mapping[claim_key]
        wiki_lines = pd.DataFrame({
            'doc_id': retrieved_doc_list,
            'sentence': retrieved_sentence_list
        })
        wiki_lines['line_index'] = wiki_lines.groupby('doc_id').cumcount()
        sentences = wiki_lines['sentence'].tolist()
        embeddings_list = get_embedding(text=sentences, tokenizer=tokenizer, model=retriever_model, device=device)
        embeddings_tensor = torch.tensor(np.vstack(embeddings_list))
        claim_embedding = torch.tensor(claim_embeddings[idx][None, :])
        wiki_lines['sim_score'] = cosine_similarity(claim_embedding, embeddings_tensor, dim=1, eps=1e-8).tolist()
        wiki_lines = wiki_lines.sort_values(by=['sim_score'], ascending=False)
        dpr_map = wiki_lines.head(2)
        
        key = list(qry_collection.keys())[idx]
        claim = claims[idx]
        evidences = dpr_map['sentence'].tolist()
        concat_evidence = ' '.join(evidences)
        
        if concat_evidence == '':
            nli_inputs = [f"{claim} [SEP] [NO_EVIDENCE]"]
        else:
            nli_inputs = [f"{claim} [SEP] {concat_evidence}"]
        
        inputs = get_nli_encodings(inputs=nli_inputs, tokenizer=tokenizer, device=device)
        outputs = nli_model(**inputs).logits
        
        label2idx = {
            'SUPPORTS': 0,
            'REFUTES': 1,
            'NOT ENOUGH INFO': 2
        }
        idx2label = {id: label for label, id in label2idx.items()}
        prediction = idx2label[torch.argmax(outputs, dim=1).item()]
                
        output_predictions.append(prediction)
        if prediction == 'NOT ENOUGH INFO':
            doc_line_list = []
        else:
            doc_line_list = dpr_map.apply(lambda row: (row['doc_id'], row['line_index']), axis=1).tolist()

        claim_result_dict[key] = doc_line_list
   
    query_df['results'] = query_df['id'].map(claim_result_dict)
    query_df['prediction'] = output_predictions
    query_df.to_pickle('save/task_results_new_3.pkl')
    
if __name__ == '__main__':
    main()