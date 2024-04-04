import json
import numpy as np
import os
import pickle
import torch
import unicodedata
from dataclasses import dataclass
from dpr import DPR
from itertools import chain
from pyserini.search.lucene import LuceneSearcher
from sparse_retriever import SparseRetriever
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer

class FeverDataset(Dataset):
    def __init__(self, params):
        self.type = params.dataset_type
        self.num_docs_to_use = params.num_docs_to_use
        self.max_sent_per_doc = params.max_sent_per_doc
        self.max_evidence_to_retrieve = params.max_evidence_to_retrieve
        self.use_ner = params.use_ner
        if self.type == 'train':
            with open('./data/fever/train.jsonl', 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        elif self.type == 'val':
            with open('./data/fever_ner/shared_task_dev.jsonl', 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        elif self.type == 'test':
            with open('./data/fever_ner/shared_task_test.jsonl', 'r', encoding='utf-8') as f:        
                self.data = [json.loads(line) for line in f]
        else:
            raise ValueError(f'Wrong dset type: {self.type}')

        self.label_to_id = {
            'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2
        }
        self.id_to_label = {v:k for k,v in self.label_to_id.items()}

        if self.use_ner:
            self.searcher = LuceneSearcher('indexes/wiki-pages/')
        else:
            self.searcher = SparseRetriever()
        
    def __getitem__(self, index):
        instance = self.data[index]

        claim = instance['claim']
        tags = instance["tags"]
        if self.type == 'test':
            return {
                'claim': claim,
                'tags': tags if self.use_ner else None
            }
        
        if self.type == 'val':
            label = self.label_to_id[instance['label']]
        elif self.type == 'train':
            label = np.zeros(3)
            label[self.label_to_id[instance['label']]] = 1
        
        if instance['label'] == 'NOT ENOUGH INFO':
            evidence = self.retrieve_evidence(claim)
        else:
            evidence = self.get_evidence(claim, instance['evidence'])
    
        return {
            'claim': claim,
            'tags': tags if self.use_ner else None,
            'label': label,
            'gold_evidence_original': instance['evidence'],
            **evidence
        }
    
    def retrieve_evidence(self, claim):
        hits = self.searcher.search(claim, k=self.num_docs_to_use)
        mapping = []
        sentences = []
        for hit in hits:
            doc = json.loads(self.searcher.doc(hit.docid).raw())
            for i, sentence in enumerate(doc['sentences']):
                if sentence != '':
                    sentences.append(sentence)
                    mapping.append((hit.docid, i))
        return {
            'doc_sent_id_mapping': mapping[:self.max_evidence_to_retrieve],
            'sentences': sentences[:self.max_evidence_to_retrieve],
            'gold_evidence_dict': {},
        }

    def get_evidence(self, claim, evidence_data):
        fetch = lambda x: (x[2], x[3])

        gold_evidence = {}
        for doc_data in evidence_data:
            for sent_data in doc_data:
                docid, sentid = fetch(sent_data)
                if docid == None: continue
                if docid not in gold_evidence:
                    gold_evidence[docid] = []
                gold_evidence[docid].append(sentid)

        docs_added = 0
        mapping = []
        sentences = []
        for docid, sentids in gold_evidence.items():
            if docs_added >= self.num_docs_to_use: break
            docs_added += 1
            sents_added = 0
            for sentid in sentids:
                if sents_added >= self.max_sent_per_doc: break
                sents_added += 1
                doc = self.searcher.doc(unicodedata.normalize('NFC', docid))
                doc = json.loads(doc.raw())
                sentences.append(doc['sentences'][sentid])
                mapping.append((docid, sentid))

        if len(sentences) < self.max_evidence_to_retrieve:
            added = len(sentences)
            retrieved = self.retrieve_evidence(claim)
            sentences.extend(retrieved['sentences'][:(self.max_evidence_to_retrieve - added)])
            mapping.extend(retrieved['doc_sent_id_mapping'][:(self.max_evidence_to_retrieve - added)])

        return {
            'doc_sent_id_mapping': mapping,
            'sentences': sentences,
            'gold_evidence_dict': gold_evidence,
        }
    
    def __len__(self):
        return len(self.data)
    
class NLICollate():
    def __init__(self, params) -> None:
        self.use_gold_as_evidence = params.use_gold_as_evidence
        self.num_docs_to_use = params.num_docs_to_use
        self.max_sent_per_doc = params.max_sent_per_doc
        self.max_evidence_length = params.max_evidence_length
        self.max_evidence_to_retrieve = params.max_evidence_to_retrieve
        self.truncate_evidence = params.truncate_evidence
        self.use_ner = params.use_ner
        if not self.use_gold_as_evidence:
            if self.use_ner:
                self.first_stage_retriever = SparseRetriever()
            else:
                self.first_stage_retriever = LuceneSearcher('indexes/wiki-pages')              
            self.second_stage_retriever = DPR().to(params.device)
            self.second_stage_retriever.load_state_dict(torch.load(params.dpr_model_path))
            self.tokenize_retriever_batch = RetrieverCollate().tokenize_batch
            self.device = params.device

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialogRPT-updown")
        self.tokenizer_config = {
            'padding': True,
            'truncation': True,
            'max_length': 512,
            'return_attention_mask': True,
            'return_tensors': 'pt'
        }

        # self.truncate_evidence = lambda x: self.tokenizer.decode(self.tokenizer(x, add_special_tokens=True)[:self.max_evidence_length])

    def truncate(self, evidence: str):
        encoded_evidence = self.tokenizer.encode(evidence, add_special_tokens=True, max_length=self.max_evidence_length)
        decoded_evidence = self.tokenizer.decode(encoded_evidence)
        return decoded_evidence


    def tokenize_batch(self, claim, evidence):
        input_strs = []
        for i in range(len(claim)):
            if self.truncate_evidence:
                sentences = list(map(self.truncate, [ f" evidence_token {e}" for e in evidence[i]]))
            else:
                sentences = [ f" evidence_token {e}" for e in evidence[i]]
            input_str = "[CLS] claim_token " + claim[i] + "".join(sentences) + " [SEP]"

            input_strs.append(input_str)
        inputs =  self.tokenizer.batch_encode_plus(
            input_strs,
            **self.tokenizer_config
        )
        return inputs
    
    def retrieve_evidence(self, claim, tag=None):
        if self.use_ner:
            batch_hits = self.first_stage_retriever.batch_search(queries=claim, tags=tag, qids=[str(i) for i in range(len(claim))], k=self.num_docs_to_use)
        else:
            batch_hits = self.first_stage_retriever.batch_search(queries=claim, qids=[str(i) for i in range(len(claim))], k=self.num_docs_to_use)

        mapping = []
        sentences = []
        # print('CLAIMS: ')
        # for i, c in enumerate(claim):
        #     print(f'{i}: {c}')
        # print('========================================')
        for hits in batch_hits.values():
            claim_map = []
            claim_sents = []
            for hit in hits:
                doc = json.loads(self.first_stage_retriever.doc(hit.docid).raw())
                sents_added = 0
                for sentid, sent in enumerate(doc['sentences']):
                    if sents_added >= self.max_sent_per_doc: break
                    sents_added += 1
                    if sent != "":
                        claim_map.append((hit.docid, sentid))
                        claim_sents.append(sent)
            sentences.append(claim_sents)
            mapping.append(claim_map)
        # print('Sentences: ')
        # for i,s in enumerate(sentences):
        #     print(f'{i}: {s}')
        # print("==========================================")


        claim, evidence = self.tokenize_retriever_batch(claim, sentences)
        mask = pad_sequence(
            [torch.ones(len(inst)).unsqueeze(1) for inst in mapping], batch_first=True, padding_value=0.0
        ).squeeze(-1).type(torch.bool)

        with torch.no_grad():
            batch_topk = self.second_stage_retriever.topk(batch={
                'claim': {k:v.to(self.device) for k, v in claim.items()},
                'evidence': {k:v.to(self.device) for k,v in evidence.items()},
                'doc_sent_id_mapping': mapping,
                'mask': mask.to(self.device)
            }, k=self.max_evidence_to_retrieve)

        batch_sentences = []
        for topk in batch_topk:
            sentences = []
            for docid, sentid in topk:
                doc = json.loads(self.first_stage_retriever.doc(docid).raw())
                sentences.append(doc['sentences'][sentid])
            batch_sentences.append(sentences)
        # print('RETRIEVED EVIDENCE')
        # for i, sent in enumerate(batch_sentences):
        #     print(f'{i}: {sent}')
        # print("==================================================")
        return batch_sentences, batch_topk

    def __call__(self, batch):
        processed_batch = {}
        claim = [
            inst['claim'] for inst in batch
        ]
        tag = [
            inst['tags'] for inst in batch
        ]
        if self.use_gold_as_evidence:
            sentences = [
                inst['sentences'] for inst in batch
            ]
            tokenized_input = self.tokenize_batch(claim, sentences)
           
        else:
            sentences, batch_topk = self.retrieve_evidence(claim, tag)
            tokenized_input = self.tokenize_batch(claim, sentences)
            processed_batch['retrieved_evidence'] = batch_topk
            
        processed_batch['tokenized_claim_evidence'] = tokenized_input

        if 'label' in batch[0]:
            if type(batch[0]['label']) == int:
                processed_batch['label'] = torch.tensor([inst['label'] for inst in batch])
            else:
                processed_batch['label'] = torch.cat(
                    list(map(lambda inst: torch.Tensor(inst['label']).unsqueeze(0), batch)), dim=0
                )

        if 'gold_evidence_dict' in batch[0]:
            processed_batch['gold_evidence_dict'] = [
                instance['gold_evidence_dict'] for instance in batch
            ]
        if 'doc_sent_id_mapping' in batch[0]:
            processed_batch['doc_sent_id_mapping'] = [
                instance['doc_sent_id_mapping'] for instance in batch
            ]
        if 'gold_evidence_original' in batch[0]:
            processed_batch['gold_evidence_original'] = [
                instance['gold_evidence_original'] for instance in batch
            ]

        return processed_batch

class RetrieverDataset(Dataset):
    '''
        dataloader returns
            claim: list of len bsz
            evidence: list of len num_passages
                        each element: tuple of len bsz
                    i.e. [(e11, e12, e13), (e21,e22,e23)]
                    eij: ith evidence of jth example
            evidence_label: tensor of shape (bsz, num_passages)
    '''
    def __init__(self, params):
        self.type = params.dataset_type
        self.num_positives = params.num_positives
        self.num_negatives = params.num_negatives
        self.num_docs_to_retrieve = params.num_docs_to_retrieve
        self.max_sent_per_doc = params.max_sent_per_doc

        if self.type == 'train':
            with open('./data/fever/train.jsonl', 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        elif self.type == 'val':
            with open('./data/fever_ner/shared_task_dev.jsonl', 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        else:
            raise ValueError(f'Wrong dset type: {self.type}')
    
        self.searcher = LuceneSearcher('indexes/wiki-pages')
    
    def __getitem__(self, index):
        '''
        
        '''
        instance = self.data[index]

        claim = instance['claim']

        if self.type == 'train':
            gold_evidence = self.get_gold_evidence(instance['evidence'])
            evidence = self.add_negatives(claim, gold_evidence)
            return {
                'claim': claim, 
                **evidence
            }
        elif self.type == 'val':
            evidence = self.retrieve_evidence(claim)
            gold_evidence = self.get_gold_map(instance['evidence'])
            return {
                'claim': claim,
                **evidence,
                'gold_evidence': gold_evidence
            }
        
    def get_gold_map(self, evidence):
        gold_evidence = []
        fetch = lambda x: (x[2], x[3])
        for doc_data in evidence:
            for sent_data in doc_data:
                docid, sentid = fetch(sent_data)
                if docid != None: 
                    gold_evidence.append((docid, sentid))
        return gold_evidence


    def get_gold_evidence(self, evidence_data):
        '''
            given list of gold evidence
            returns
                list: tuple (docid, sentid) 
                tensor: 1 if relevant else 0
                list: all sentences of all gold documents
                      
                
        '''
        fetch = lambda x: (x[2], x[3])

        gold_evidence = {}
        for doc_data in evidence_data:
            for sent_data in doc_data:
                docid, sentid = fetch(sent_data)
                if docid is not None:
                    if docid not in gold_evidence:
                        gold_evidence[docid] = []
                    gold_evidence[docid].append(sentid)
        
        mapping = []
        relevance = []
        sentences = []

        added_docs = 0
        for docid, sentids in gold_evidence.items():
            if docid == None: break
            if added_docs >= self.num_positives: break
            added_docs += 1
            doc = json.loads(self.searcher.doc(unicodedata.normalize('NFC', docid)).raw())
            added_sents = 0
            for i, sent in enumerate(doc['sentences']):
                if sent == '': continue
                if added_sents >= self.max_sent_per_doc: break
                added_sents += 1
                mapping.append((docid, i))
                relevance.append(i in sentids)
                sentences.append(sent)
                
        return {
            'doc_sent_id_mapping': mapping,
            'relevance': relevance,
            'evidence': sentences
        }
    
    def add_negatives(self, claim, gold_evidence):
        used_docids = set([mapping[0] for mapping in gold_evidence['doc_sent_id_mapping']])

        negatives = []
        hits = self.searcher.search(claim, self.num_negatives)
        for hit in hits:
            if hit.docid not in used_docids:
                doc = json.loads(self.searcher.doc(hit.docid).raw())
                added_sents = 0 
                for i, sent in enumerate(doc['sentences']):
                    if sent == '': continue
                    if added_sents >= self.max_sent_per_doc: break
                    added_sents += 1
                    gold_evidence['doc_sent_id_mapping'].append((hit.docid, i))
                    negatives.append(sent)

        gold_evidence['relevance'].extend([False for i in range(len(negatives))])
        gold_evidence['evidence'].extend(negatives)

        return gold_evidence

    def retrieve_evidence(self, claim):
        hits = self.searcher.search(claim, self.num_docs_to_retrieve)
        mapping = []
        sentences = []
        for hit in hits:
            doc = json.loads(self.searcher.doc(hit.docid).raw())
            added_sents = 0
            for sentid, sent in enumerate(doc['sentences']):
                if sent == '': continue
                if added_sents >= self.max_sent_per_doc: break
                added_sents += 1
                mapping.append((hit.docid, sentid))
                sentences.append(sent)
        assert len(mapping) == len(sentences)
        return {
            'doc_sent_id_mapping': mapping,
            'evidence': sentences
        }

    def __len__(self):
        return len(self.data)
    
class RetrieverCollate():
    def __init__(self):
        self.claim_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        self.evidence_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

        self.tokenizer_config = {
            'padding': 'max_length',
            'truncation': True,
            'return_attention_mask': True,
            'return_tensors': 'pt',
            'max_length': 64
        }
    
    def tokenize_batch(self, claim, evidence):
        tokenized_claim = self.claim_tokenizer.batch_encode_plus(
                claim,
                **self.tokenizer_config
            )
        tokenized_evidence = self.evidence_tokenizer.batch_encode_plus(
                list(chain.from_iterable(evidence)),
                **self.tokenizer_config
            )
        return tokenized_claim, tokenized_evidence
        

    def __call__(self, batch):
        # 'doc_sent_id_mapping': mapping,
        #     'relevance': relevance,
        #     'sentences':

        processed_batch = {}

        claims = [
            instance['claim'] for instance in batch
        ]
        evidence = [
            instance['evidence'] for instance in batch
        ]
        processed_batch['claim'], processed_batch['evidence'] = self.tokenize_batch(claims, evidence)

        # mapping: list of list of tuples (docid, sentid)
        processed_batch['doc_sent_id_mapping'] = [
            instance['doc_sent_id_mapping'] for instance in batch
        ]

        if 'gold_evidence' in batch[0]:
            processed_batch['gold_evidence'] = [
                instance['gold_evidence'] for instance in batch
            ]

        # mask 
        processed_batch['mask'] = pad_sequence(
            map(lambda instance: torch.ones(len(instance['doc_sent_id_mapping'])).unsqueeze(1), batch), batch_first=True, padding_value=0.0
        ).squeeze(-1).type(torch.bool)

        # relevance
        if 'relevance' in batch[0]:
            processed_batch['relevance'] = pad_sequence(
                map(lambda instance: torch.tensor(instance['relevance'], dtype=torch.float32).unsqueeze(1), batch), batch_first=True, padding_value=False
            ).squeeze(-1)

        return processed_batch

if __name__ == '__main__':
    from pprint import pprint
    # @dataclass
    # class DataParams:
    #     dataset_type: str = 'train'
    #     num_docs_to_retrieve: int = 4
    #     num_positives: int = 10
    #     num_negatives: int = 1
    #     max_sent_per_doc: int = 25

    # params = DataParams()
    # # fever_train = FeverDataset(params)
    # # for i in range(10):
    # #     print(fever_train[i], end='\n\n')

    # retriever_train = RetrieverDataset(params)

    # from torch.utils.data import DataLoader
    # loader = DataLoader(retriever_train, batch_size=4, collate_fn=RetrieverCollate())
    # # for batch in loader:
    # #     for key, val in batch.items():
    # #         print(key, type(val))
    # #         if key in ['claim', 'evidence']:
    # #             print(key, batch[key]['attention_mask'].shape)
    # #             print(key, batch[key]['input_ids'].shape)
    # #         elif type(val) == torch.Tensor:
    # #             print(val.shape)
    # #         elif type(val) == list:
    # #             print(val[0], len(val))
    # #     break

    # params.dataset_type = 'val'
    # retriever_val = RetrieverDataset(params)
    # from torch.utils.data import DataLoader
    # i = 0
    # loader = DataLoader(retriever_val, batch_size=4, collate_fn=RetrieverCollate())
    # for batch in loader:
    #     for key, val in batch.items():
    #         print(key, type(val))
    #         if key in ['claim', 'evidence']:
    #             print(key, batch[key]['attention_mask'].shape)
    #             print(key, batch[key]['input_ids'].shape)
    #         elif type(val) == torch.Tensor:
    #             print(val.shape)
    #         elif type(val) == list:
    #             print(val[0], len(val))
    #     i += 1
    #     if i > 4:
    #         break

    @dataclass
    class NLILoaderParams:
        dataset_type: str = 'val'
        num_docs_to_use: int = 3
        max_sent_per_doc: int = 8
        max_evidence_length: int = 64
        use_gold_as_evidence: bool = False
        dpr_model_path: str = './runs/bm25_dpr_gpt2/retriever/dpr_final.pth'
        max_evidence_to_retrieve: int = 5
        truncate_evidence: bool = True
        device: str = 'cuda:6'
        use_ner: bool = True

    params = NLILoaderParams()
    from torch.utils.data import DataLoader
    i = 0
    loader = DataLoader(FeverDataset(params), batch_size=4, collate_fn=NLICollate(params)) 
    for batch in loader:
        pprint(batch['retrieved_evidence'])
        pprint(batch['gold_evidence_original'])
        if i > 2:
            break
        i += 1   
    