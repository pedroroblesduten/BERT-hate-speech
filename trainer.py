import torch
import torch.nn as nn
from load_data import LoadData
import os
from tqdm import tqdm
import numpy as np
from utils import plot_losses
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments


class TrainerTransformer():
    def __init__(self,
                 huggingface_model_name,
                 model_name,
                 device,
                 epochs,
                 save_model_path,
                 save_losses_path,
                 save_weights_interval,
                 load_data_config):



        self.huggingface_model_name = huggingface_model_name
        self.model_name = model_name
        self.device = device
        self.epochs = epochs
        self.save_model_path = save_model_path
        self.save_losses_path = save_losses_path
        self.save_weights_interval = save_weights_interval
        
        self.load_data_config = load_data_config
        self.loader = LoadData(**self.load_data_config.__dict__)

        self.best_val_loss = 1e9

    def initialize_model(self, to_gpu=True):
        model = BertForSequenceClassification.from_pretrained(self.huggingface_model_name, num_labels=self.num_labels)
        if to_gpu:
            model.to(self.device)
        
        return model

    def save_models(self, model, model_name, epoch, val_loss, path):
        if val_loss < self.best_val_loss:
            name = f"BestVal_{model_name}"
            save_path = os.path.join(path, name)
            torch.save(model.state_dict(), save_path)

        if epoch % self.save_weights_interval == 0:
            name = f"epoch_{epoch}_" + model_name
            save_path = os.path.join(path, name)
            torch.save(model.state_dict(), save_path)

    def process_input(self, batch):
        texts = batch['text']
        labels = batch['labels']
        ids = batch['message_id']

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_name)

        # Tokenize the texts
        tokenized_inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"  # Return PyTorch tensors
        )

        # Move data to the same device as the model
        tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
        labels = labels.to(self.device)

        return tokenized_inputs, labels, ids

    def train(self):
        print('----------------------------')
        print('   STARTING BERT TRAINING   ')
        print('----------------------------')

        # Loading data
        train_dataloader, val_dataloader, _ = self.loader.create_dataloaders()

        # Init model
        model = self.initialize_model()

        # Training configurations
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-5)

        print('- training transformer model -')

        all_train_loss = []
        all_val_loss = []

        for epoch in range(self.epochs):
            epoch_train_losses = []
            epoch_val_losses = []
            model.train()
            with tqdm(
                total=len(train_dataloader) + len(val_dataloader),
                    desc=f"EPOCH {epoch+1} WITH DATASET PARTITION {dataset_partition + 1}",
                ) as bar:
                for batch in train_dataloader:

                    text_tokens, labels, ids = self.process_input(batch)
                    logits = model(text_tokens)

                    # Descending the gradient and updating the linear layer
                    optimizer.zero_grad()
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    # Saving the batch loss
                    loss_numpy = loss.detach().cpu().numpy()
                    epoch_train_loss.append(loss_numpy)
                    bar.update(1)

                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        text_tokens, labels, ids = self.process_input(batch)

                        # Forward pass through the model
                        logits = model(text_tokens)

                        # Getting the loss
                        loss = criterion(logits, labels)

                        # Saving the batch loss
                        loss_numpy = loss.detach().cpu().numpy()
                        epoch_val_loss.append(loss_numpy)
                        bar.update(1)

            all_train_loss.append(np.mean(epoch_train_losses))
            all_val_loss.append(np.mean(epoch_val_losses))

            # Plotting the losses
            plot_losses(all_train_loss, all_val_loss, self.save_losses_path, self.model_name)

            # Saving model checkpoints
            self.save_models(model,
                             self.model_name,
                             epoch,
                             np.mean(epoch_val_losses),
                             self.save_model_path,
            )
            print("-")
