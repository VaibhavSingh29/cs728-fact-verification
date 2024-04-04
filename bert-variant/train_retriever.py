import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm import tqdm

class EntailmentDataset(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length=128):
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            input,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_tensor
        }


class Retriever(nn.Module):
    def __init__(self, MODEL_NAME, return_dense_vector=False):
        super(Retriever, self).__init__()
        
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.return_dense_vector = return_dense_vector
        
    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.pooler_output
        if self.return_dense_vector:
            return cls_embedding
        logits = self.linear(cls_embedding)
        return self.sigmoid(logits)



def main():
    MODEL_NAME = 'bert-base-uncased'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    instances_df = pd.read_pickle('save/retriever_training_samples.pkl')
    training_instances = instances_df['training_instances'].tolist()
    inputs, labels = zip(*training_instances)
    
    dataset = EntailmentDataset(inputs=list(inputs), labels=list(labels), tokenizer=tokenizer)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    
    model = Retriever(MODEL_NAME=MODEL_NAME)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.BCELoss()
    model.train()
    
    for epoch in tqdm(range(2), desc='Epoch'):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc='Batch'):
            
            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            labels = labels.unsqueeze(1)
            loss = loss_fn(outputs, labels.float())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        
        print(f"Loss after {epoch + 1} Epochs: {epoch_loss/len(train_loader)}")
        
    model_path = 'save/retrieval_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to disk")
    
    # loaded_model = Retriever(MODEL_NAME=MODEL_NAME, return_dense_vector=True)
    # loaded_model.load_state_dict(torch.load(model_path))
    # print(loaded_model)
    

if __name__ == '__main__':
    main()