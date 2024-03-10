import os
import re
import requests
import multiprocessing
from functools import partial
import base64
from github import Github
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tenarytransformers import TernaryTransformer

class GoTokenizer:
    def __init__(self, vocab, max_length):
        self.vocab = vocab
        self.max_length = max_length
        
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}
        
        self.special_tokens = {
            "[PAD]": len(vocab),
            "[CLS]": len(vocab) + 1,
            "[SEP]": len(vocab) + 2,
            "[UNK]": len(vocab) + 3,
        }
        
        self.token_to_id.update(self.special_tokens)
        self.id_to_token.update({i: token for token, i in self.special_tokens.items()})
    
    def tokenize(self, code):
        tokens = re.findall(r"\w+|[^\w\s]", code)
        return tokens
    
    def encode(self, code):
        tokens = self.tokenize(code)
        token_ids = [self.token_to_id.get(token, self.special_tokens["[UNK]"]) for token in tokens]
        
        cls_id = self.special_tokens["[CLS]"]
        sep_id = self.special_tokens["[SEP]"]
        pad_id = self.special_tokens["[PAD]"]
        
        token_ids = [cls_id] + token_ids[:self.max_length - 2] + [sep_id]
        token_ids += [pad_id] * (self.max_length - len(token_ids))
        
        return token_ids
    
    def decode(self, token_ids):
        tokens = [self.id_to_token[i] for i in token_ids if i in self.id_to_token]
        code = " ".join(tokens)
        return code

class GoDataProcessor:
    def __init__(self, repo_owner, repo_name, access_token):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.access_token = access_token
        self.g = Github(access_token)
        self.repo = self.g.get_repo(f"{repo_owner}/{repo_name}")
    
    def process_file(self, file_content, go_files):
        if file_content.name.endswith(".go"):
            go_files.append(file_content)
            print(f"Found {len(go_files)} Go files...", end="\r")
    
    def get_go_files(self):
        go_files = multiprocessing.Manager().list()
        contents = self.repo.get_contents("")
        
        with multiprocessing.Pool() as pool:
            while contents:
                file_contents = []
                while contents and len(file_contents) < 16:
                    file_content = contents.pop(0)
                    if file_content.type == "dir":
                        contents.extend(self.repo.get_contents(file_content.path))
                    else:
                        file_contents.append(file_content)
                
                pool.map(partial(self.process_file, go_files=go_files), file_contents)
        
        return list(go_files)
    
    def process_code(self, file_content, go_code):
        try:
            code = base64.b64decode(file_content.content).decode("utf-8")
            go_code.append(code)
            print(f"Processed {len(go_code)} files...", end="\r")
        except Exception as e:
            print(f"Error processing file {file_content.path}: {str(e)}")
    
    def get_go_code(self, go_files):
        go_code = multiprocessing.Manager().list()
        
        with multiprocessing.Pool() as pool:
            pool.map(partial(self.process_code, go_code=go_code), go_files)
        
        return list(go_code)

class GoDataset(Dataset):
    def __init__(self, code_list, tokenizer):
        self.code_list = code_list
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.code_list)
    
    def __getitem__(self, index):
        code = self.code_list[index]
        tokens = self.tokenizer.encode(code)
        return torch.tensor(tokens)

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def train(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.dataloader, desc="Training", unit="batch")
        for batch in progress_bar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch, batch)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), batch.view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        return total_loss / len(self.dataloader)
    
    def save_checkpoint(self, epoch, checkpoint_path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, checkpoint_path)

def main():
    access_token = "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"
    repo_owner = "golang"
    repo_name = "go"
    
    data_processor = GoDataProcessor(repo_owner, repo_name, access_token)
    go_files = data_processor.get_go_files()
    go_code = data_processor.get_go_code(go_files)
    
    vocab = ["func", "var", "const", "type", "struct", "interface", "map", "if", "else", "for", "switch", "case", "default", "goto", "chan", "range", "import", "package", "return", "(", ")", "{", "}", "[", "]", ".", ",", ";", ":", "&", "*", "+", "-", "/", "%", "=", "!", "<", ">", "==", "!=", "<=", ">=", "&&", "||", ":="]
    max_length = 512
    tokenizer = GoTokenizer(vocab, max_length)
    
    dataset = GoDataset(go_code, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    device = torch.device("rocm" if torch.is_rocm_available() else "cpu")
    model = TernaryTransformer(
        num_enc_layers=6, num_dec_layers=6, d_model=512, num_heads=8, d_ff=2048,
        input_vocab_size=len(tokenizer.token_to_id), output_vocab_size=len(tokenizer.token_to_id),
        max_seq_len=max_length, dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, dataloader, optimizer, criterion, device)
    
    checkpoint_dir = "go_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = trainer.train()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch+1}.pt")
        trainer.save_checkpoint(epoch+1, checkpoint_path)
    
    print("Training completed!")

if __name__ == "__main__":
    main()