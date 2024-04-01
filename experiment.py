import numpy as np
import os
import torch
import torch.optim as optim
import transformers
from itertools import chain
from loader import FeverDataset, NLICollate, RetrieverDataset, RetrieverCollate
from dpr import DPR
from nli import NLI
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Experiment():
    def __init__(self, loader_params, training_params, retriever_exp=False) -> None:
        self.training_params = training_params
        self.device = self.training_params.device
        self.retriever_exp = retriever_exp

        if retriever_exp:
            collate_fn = RetrieverCollate()
            self.train_loader = DataLoader(RetrieverDataset(loader_params), batch_size=training_params.bsz, collate_fn=collate_fn, shuffle=True)
            loader_params.dataset_type='val'
            self.val_loader = DataLoader(RetrieverDataset(loader_params), batch_size=training_params.bsz, collate_fn=collate_fn, shuffle=False)

            self.dpr = DPR().to(self.device)
            self.optimizer = optim.Adam(self.dpr.parameters(), lr=training_params.lr)
            self.update_log = self.update_retriever_log

            
        else:
            collate_fn = NLICollate(loader_params)
            self.train_loader = DataLoader(FeverDataset(loader_params), batch_size=training_params.bsz, collate_fn=collate_fn, shuffle=True)
            loader_params.dataset_type='val'
            self.val_loader = DataLoader(FeverDataset(loader_params), batch_size=training_params.bsz, collate_fn=collate_fn, shuffle=False)

            self.nli = NLI().to(self.device)
            self.optimizer = optim.AdamW(self.nli.parameters(), lr=training_params.lr)
            self.update_log = self.update_nli_log
        
        
          
        
            # self.scheduler = LinearLR(self.optimizer, start_factor=0.5, total_iters=)

        self.writer = SummaryWriter(os.path.join('runs', training_params.exp_name, 'retriever_logs' if retriever_exp else 'nli_logs'))

    def train(self):
        with tqdm(total=self.training_params.num_epochs, desc='Epochs Done: ') as pbar:
            for epoch in range(self.training_params.num_epochs):
                epoch_loss = 0
                if self.retriever_exp: self.dpr.train()
                else: self.nli.train()
                with tqdm(total=len(self.train_loader), desc='Batches Done: ', leave=False) as inner_pbar:
                    for i, batch in enumerate(self.train_loader):
                        self.optimizer.zero_grad()
                        self._to_device(batch)
                        if self.retriever_exp:
                            loss = self.dpr(batch)
                        else:
                            loss = self.nli(batch)
                        loss.backward()
                        self.optimizer.step()
                        self.writer.add_scalar('train_loss', loss.item(), epoch*len(self.train_loader) + i)
                        epoch_loss += loss.item()
                        inner_pbar.update(1)
                epoch_loss /= len(self.train_loader)

                if epoch % self.training_params.es_step == 0:
                    metric = self.update_log(i, epoch)
                
                pbar.set_postfix({'train loss': epoch_loss, **metric})
                pbar.update(1)
        
        if self.retriever_exp:
            self.save(self.dpr, os.path.join('runs', self.training_params.exp_name, 'retriever', f'dpr_final.pth'))
        else:
            self.save(self.nli, os.path.join('runs', self.training_params.exp_name, 'nli', f'nli_final.pth'))
        
    def update_nli_log(self, batch_num, epoch):
        micro_f1 = self.validate_nli()
        self.writer.add_scalar('val_micro_f1', micro_f1, epoch*len(self.train_loader) + batch_num)
        self.save(self.nli, os.path.join('runs', self.training_params.exp_name, 'nli', f'nli_{epoch}.pth'))
        return {'micro_f1': micro_f1}

    def update_retriever_log(self, batch_num, epoch):
        mrr = self.validate_retriever()
        self.writer.add_scalar('val_mrr', mrr, epoch*len(self.train_loader) + batch_num)
        self.save(self.dpr, os.path.join('runs', self.training_params.exp_name, 'retriever', f'dpr_{epoch}.pth'))
        return {'MRR': mrr}
    
    def validate_nli(self):
        true = []
        predicted = []
        self.nli.eval()
        with tqdm(total=len(self.val_loader), desc='Validation Progress', leave=False) as pbar:
            with torch.no_grad():
                for batch in self.val_loader:
                    true.extend(batch['label'].tolist())
                    self._to_device(batch)
                    preds = self.nli.predict(batch)
                    predicted.extend(preds.tolist())
        micro_f1 = f1_score(true, predicted, average='micro')
        return micro_f1      
    
    def validate_retriever(self):
        total_rr = 0
        relevant_docs = 0
        self.dpr.eval()
        with tqdm(total=len(self.val_loader), desc='Validation Progress', leave=False) as pbar:
            with torch.no_grad():
                for batch in self.val_loader:
                    self._to_device(batch)
                    sim_score = self.dpr.predict(batch)
                    rr, num_docs = self.mrr_from_sim_score_list(sim_score, batch)
                    total_rr += rr
                    relevant_docs += num_docs
                    pbar.set_postfix({'running MRR': total_rr / relevant_docs})
                    pbar.update(1)
        mrr = total_rr / relevant_docs
        return mrr.item()
    
    def mrr_from_sim_score_list(self, sim_score, batch, reduce='sum'):
        num_sentences = batch['mask'].sum(dim=1).type(torch.int).tolist()
        rr = 0
        num_docs = 0
        for inst in range(sim_score.shape[0]):
            gold_evidence = batch['gold_evidence'][inst]
            evidence = batch['doc_sent_id_mapping'][inst]
            scores = sim_score[inst][:num_sentences[inst]]
            rank_ids = torch.argsort(scores, descending=True)
            # print(evidence)
            # print(gold_evidence)
            # print(rank_ids)
            # print(num_sentences)
            # print(scores)
            relevant = [
                evidence[i] in gold_evidence for i in rank_ids
            ]
            rr += (np.array(relevant) / np.arange(1, num_sentences[inst]+1)).sum()
            num_docs += sum(relevant)
        
        if reduce=='sum':
            return rr, num_docs
        elif reduce == 'mean':
            mrr = rr / num_docs
            return mrr
        
    def mrr_from_sim_score_tensor(self, sim_score, relevance, reduce='sum'):
        ranks = sim_score.argsort(descending=True, dim=1)
        bsz, max_sents = ranks.shape
        for i in range(bsz):
            relevant_idx = relevance[i].argwhere().T[0]
            ranks[i].apply_(lambda x: x in relevant_idx).bool()

        reciprocal_rank = ranks / torch.arange(1, max_sents+1).unsqueeze(0)

        if reduce=='sum':
            mrr = reciprocal_rank.sum()
        elif reduce == 'mean':
            mrr = reciprocal_rank.sum() / ranks.sum()
        
        return mrr
    
    def _to_device(self, batch):
        for key in batch:
            if type(batch[key]) == torch.Tensor:
                batch[key] = batch[key].to(self.device)
            if type(batch[key]) == transformers.tokenization_utils_base.BatchEncoding:
                batch[key]['input_ids'] = batch[key]['input_ids'].to(self.device)
                batch[key]['attention_mask'] = batch[key]['attention_mask'].to(self.device)

    def save(self, model, path):
        torch.save(model.state_dict(), path)
