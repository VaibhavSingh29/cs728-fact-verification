import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2ForSequenceClassification

class NLI(nn.Module):
    def __init__(self) -> None:
        super(NLI, self).__init__()

        self.model = GPT2ForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown", num_labels=3, ignore_mismatched_sizes=True)
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        logits = self._forward(batch['tokenized_claim_evidence'])
        loss = self.loss(logits, batch['label'])
        return loss
    
    def predict(self, batch):
        probs = self.sigmoid(self.model(**batch['tokenized_claim_evidence']).logits)
        return probs.argmax(dim=1)

    def _forward(self, tokenized_input):
        logits = self.model(**tokenized_input).logits
        return logits

