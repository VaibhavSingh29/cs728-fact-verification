import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU

class NLIDataset(Dataset):
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


def main():
    set_seed()
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    data_path = 'save/nli_data.pkl'
    val_data_path = 'save/val_nli_data.pkl'
    data = pd.read_pickle(data_path)
    val_data = pd.read_pickle(val_data_path)
    label_mapping = {
        'SUPPORTS': 0,
        'REFUTES': 1,
        'NOT ENOUGH INFO': 2
    }
    data['labels'] = data['labels'].map(label_mapping)
    val_data['labels'] = val_data['labels'].map(label_mapping)
    
    inputs = data['inputs'].tolist()
    labels = data['labels'].tolist()
    
    val_inputs = val_data['inputs'].tolist()
    val_labels = val_data['labels'].tolist()
    
    dataset = NLIDataset(inputs=inputs, labels=labels, tokenizer=tokenizer)
    val_dataset = NLIDataset(inputs=val_inputs, labels=val_labels, tokenizer=tokenizer)
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    train_losses = []
    val_losses = []
        
    for epoch in tqdm(range(5), desc='Epoch'):
        epoch_loss = 0
        val_epoch_loss = 0
        model.train()
        for batch in tqdm(train_loader, desc='Batch'):
            
            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        
        train_losses.append(epoch_loss/len(train_loader))
        
        model.eval()
    
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Batch'):
                
                input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                val_epoch_loss += loss.item()
                
        
        val_losses.append(val_epoch_loss/len(val_loader))
    
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('save/training_and_validation_losses.png', format='png')

    
    model_path = 'save/nli_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to disk")

if __name__ == '__main__':
    main()