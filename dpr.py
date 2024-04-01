import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

class DPR(nn.Module):
    def __init__(self):
        super(DPR, self).__init__()
        emb_dim = 128

        self.claim_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
        self.project_claim = nn.Linear(768, emb_dim)
        self.evidence_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        self.project_evidence = nn.Linear(768, emb_dim)
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, batch):
        sim_score = self._forward(batch)
        loss = self.loss(sim_score, batch['relevance'])
        loss.masked_fill_(
            mask=~batch['mask'],
            value=0.0
        )
        loss = loss.sum() / batch['mask'].sum()
        return loss

    def predict(self, batch):
        sim_score = self._forward(batch)
        sim_score = self.sigmoid(sim_score)
        return sim_score

    def _forward(self, batch):
        '''
            claim: tokenizer input_ids of shape bsz x seq_len
            evidence: tokenizer input_ids of shape (bsz x num_passages) x seq_len
        '''
        claim = batch['claim']
        evidence = batch['evidence']
        mask = batch['mask']
        num_sentences = mask.sum(dim=1).type(torch.int).squeeze().tolist()
        
        encoded_claim = self.project_claim(
            self.claim_encoder(**claim).pooler_output
        ) # bsz x 768
        encoded_evidence = self.project_evidence(
            self.evidence_encoder(**evidence).pooler_output
        ) # (bsz x num_passages) x 768

        # reshape for dot product
        encoded_claim = encoded_claim.unsqueeze(1) # bsz x 1 x 768

        evidence_per_claim = []
        left_idx = 0
        for i in num_sentences:
            evidence_per_claim.append(encoded_evidence[left_idx:left_idx+i])
            left_idx += i
        evidence_per_claim = pad_sequence(evidence_per_claim, batch_first=True) # bsz x max_sent x 768

        sim_score = torch.bmm(encoded_claim, evidence_per_claim.transpose(1, 2)).squeeze(1)
        sim_score = sim_score.masked_fill_(
            mask=~mask.squeeze(),
            value=float('-inf')
        )
        return sim_score
    
    def topk(self, batch, k=10):
        sim_score = self.predict(batch)
        num_sentences = batch['mask'].sum(dim=1).type(torch.int).tolist()
        topk = []
        for inst in range(sim_score.shape[0]):
            evidence = batch['doc_sent_id_mapping'][inst]
            scores = sim_score[inst][:num_sentences[inst]]
            rank_ids = torch.argsort(scores, descending=True)
            relevant = [
                evidence[i] for i in rank_ids
            ][:k]
            topk.append(relevant)
        return topk
