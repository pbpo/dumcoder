import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from tenarytransformers import TernaryTransformer
import os
import pyarrow.parquet as pq
import numpy as  np
from torch.utils.data import DataLoader
class ParquetDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, tokenizer):
        self.file_list = file_list
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
     file_path = self.file_list[index]
     table = pq.read_table(file_path)
     data = table.to_pandas()
    
     input_code = data["files"].iloc[0]
     output_code = data["files"].iloc[0]  
    
     if isinstance(input_code, bytes):
        input_code = input_code.decode('utf-8')  
     elif isinstance(input_code, np.ndarray):
        input_code = str(input_code)  
    
     if isinstance(output_code, bytes):
        output_code = output_code.decode('utf-8')  
     elif isinstance(output_code, np.ndarray):
        output_code = str(output_code)  
    
     input_tensor = self.tokenizer.encode(input_code)
     output_tensor = self.tokenizer.encode(output_code)
    
     return input_tensor, output_tensor
def save_model_and_optimizer(model, optimizer, epoch, file_name="model_checkpoint.pth"):
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
   
    torch.save(checkpoint, file_name)
    print(f"Model and optimizer saved to {file_name} at epoch {epoch}")

def load_model_and_optimizer(model, optimizer, file_name="model_checkpoint.pth"):
  
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model and optimizer loaded from {file_name} at epoch {epoch}")
    return epoch
     
def train_with_parquet(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_tensor, output_tensor = batch
        input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)
        
        optimizer.zero_grad()
        output = model(input_tensor, output_tensor[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), output_tensor[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

class CodeTokenizer:
    def __init__(self, vocab, max_length):
        self.vocab = vocab
        self.max_length = max_length
        
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}
        
        self.special_tokens = {
            "<PAD>": len(vocab),
            "<BOS>": len(vocab) + 1,
            "<EOS>": len(vocab) + 2,
            "<UNK>": len(vocab) + 3,
        }
        
        self.token_to_id.update(self.special_tokens)
        self.id_to_token.update({i: token for token, i in self.special_tokens.items()})
    
    def tokenize(self, code):
     if isinstance(code, bytes):
        code = code.decode('utf-8')  # 바이트 형태의 데이터를 문자열로 디코딩
     tokens = re.findall(r"\w+|[^\w\s]", code)
     return tokens
    
    def encode(self, code):
        tokens = self.tokenize(code)
        token_ids = [self.token_to_id.get(token, self.special_tokens["<UNK>"]) for token in tokens]
        
        bos_id = self.special_tokens["<BOS>"]
        eos_id = self.special_tokens["<EOS>"]
        pad_id = self.special_tokens["<PAD>"]
        
        token_ids = [bos_id] + token_ids[:self.max_length - 2] + [eos_id]
        token_ids += [pad_id] * (self.max_length - len(token_ids))
        
        return torch.tensor(token_ids)
    
    def decode(self, token_ids):
        tokens = [self.id_to_token[int(i)] for i in token_ids if int(i) in self.id_to_token]
        code = "".join(tokens)
        return code

class CodingAssistantDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
     file_path = self.file_list[index]
     table = pq.read_table(file_path)
     data = table.to_pandas()
    
     input_code = data["files"].iloc[0]
     output_code = data["files"].iloc[0]  
    
     if isinstance(input_code, bytes):
        input_code = input_code.decode('utf-8')  
     elif isinstance(input_code, np.ndarray):
        input_code = input_code.item()  
    
     if isinstance(output_code, bytes):
        output_code = output_code.decode('utf-8')  
     elif isinstance(output_code, np.ndarray):
        output_code = output_code.item()  
     input_tensor = self.tokenizer.encode(input_code)
     output_tensor = self.tokenizer.encode(output_code)
    
     return input_tensor, output_tensor

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_tensor, output_tensor = batch
        input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)
        
        optimizer.zero_grad()
        output = model(input_tensor, output_tensor[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), output_tensor[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")
    
    return avg_loss

def main():
    file_list = [f"train-{i:05d}-of-00064.parquet" for i in range(63)]
    print("File list created.")
    
    vocab = ["def", "return", "if", "else", "for", "in", "while", "print", "input", "range",
             "+", "-", "*", "/", "=", ":", ",", "(", ")", "[", "]", "{", "}", "<", ">", "sum", ";"]
    max_length = 100
    tokenizer = CodeTokenizer(vocab, max_length)
    print("Tokenizer initialized.")
    
    dataset = ParquetDataset(file_list, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16)
    print("Dataset and DataLoader created.")
    
    input_vocab_size = len(tokenizer.token_to_id)
    output_vocab_size = len(tokenizer.token_to_id)
    d_model = 256
    num_heads = 8
    d_ff = 512
    num_enc_layers = 3
    num_dec_layers = 3
    dropout = 0.1
    
    model = TernaryTransformer(
        num_enc_layers, num_dec_layers, d_model, num_heads, d_ff,
        input_vocab_size, output_vocab_size, max_length, dropout
    )
    print("Model initialized.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print("Optimizer and Criterion set.")
    
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    model.to(device)
    print(f"Model moved to {device}.")
    
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        train_loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    save_model_and_optimizer(model, optimizer, epoch, "final_model_checkpoint.pth")

    
    print("Training completed.")

if __name__ == "__main__":
    main()