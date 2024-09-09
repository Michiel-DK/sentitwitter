import torch
import datasets
import pytorch_lightning as pl

from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    
    """Class to import dataset from Hugging Face and split into train/val"""
    
    def __init__(self, model_name="distilbert/distilbert-base-uncased", batch_size=32, val_size=0.2, random_state=42, stratify=True):
        super().__init__()

        self.batch_size = batch_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        tweets_dataset = load_dataset("m-newhauser/senator-tweets")
        
        if self.stratify:
            tweets_dataset_split = tweets_dataset['train'].train_test_split(test_size=0.01, train_size=0.01, seed=self.random_state, stratify_by_column='labels')
        else:
            tweets_dataset_split = tweets_dataset['train'].train_test_split(test_size=0.01, train_size=0.01, seed=self.random_state)

        self.train_data = tweets_dataset_split['train']
        self.val_data = tweets_dataset_split['test']


    def tokenize_data(self, example):
        return self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            
            ### REMOVE COLUMNS FOR SPEED?
            ###dataset = dataset.remove_columns(['A', 'B', 'C', 'D', 'E', 'F'])
            
            self.train_data.set_format(
                type="torch", columns=['date', 'id', 'username', 'text', 'party', 'labels', 'embeddings', 'input_ids', 'attention_mask']
            )

            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(
                type="torch", columns=['date', 'id', 'username', 'text', 'party', 'labels', 'embeddings', 'input_ids', 'attention_mask']
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )
        
if __name__ == '__main__':
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)