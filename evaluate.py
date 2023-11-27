import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from load_data import LoadData

class BERTClassificationEvaluation:
    def __init__(
        self,
        huggingface_model_name,
        model_name,
        device,
        save_model_path,
        save_results_path,
        classes,
        num_labels,
        load_data_config
    ):
        self.huggingface_model_name = huggingface_model_name
        self.model_name = model_name
        self.device = device
        self.save_model_path = save_model_path
        self.save_results_path = save_results_path
        self.classes = classes
        self.num_labels = num_labels
        self.load_data_config = load_data_config
        self.loader = LoadData(**self.load_data_config.__dict__)

        self.best_val_loss = 1e9
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)

    def process_input(self, batch):
        texts = batch['text']
        labels = batch['labels']

        tokenized_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokenized_inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
        labels = labels.to(self.device)

        return tokenized_inputs, labels

    def initialize_model(self, to_gpu=True):
        model = BertForSequenceClassification.from_pretrained(self.huggingface_model_name, num_labels=self.num_labels)
        if to_gpu:
            model.to(self.device)

        return model

    def load_model(self, model):
        load_path = os.path.join(self.save_model_path, f"BestVal_{self.model_name}")
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"No model file found at specified path: {load_path}")

        model.load_state_dict(torch.load(load_path, map_location=self.device))
        model.to(self.device)
        return model


    def calculate_metrics_table(self, binary_results, true_labels):
        # Ensure binary_results and true_labels have the same shape
        if binary_results.shape != true_labels.shape:
            raise ValueError("Shapes of binary_results and true_labels must match.")

        num_classes = binary_results.shape[1]

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for class_idx in range(num_classes):
            class_binary_results = binary_results[:, class_idx].cpu().numpy()
            class_true_labels = true_labels[:, class_idx].cpu().numpy()

            accuracy = accuracy_score(class_true_labels, class_binary_results)
            precision = precision_score(
                class_true_labels, class_binary_results, zero_division=0
            )
            recall = recall_score(
                class_true_labels, class_binary_results, zero_division=0
            )
            f1 = f1_score(class_true_labels, class_binary_results, zero_division=0)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        metrics_dict = {
            "Class": self.classes,
            "Accuracy": accuracy_scores,
            "Precision": precision_scores,
            "Recall": recall_scores,
            "F1 Score": f1_scores,
        }

        metrics_df = pd.DataFrame(metrics_dict)
        return metrics_df

    def find_optimal_threshold_per_class(self, model):
        model.to(self.device)
        model.eval()

        num_classes = len(self.classes)
        thresholds = np.arange(0, 1.01, 0.01)  # Array of thresholds from 0 to 1 with step 0.01

        # Dictionaries to store predictions and true labels
        predictions = {thresh: [[] for _ in range(num_classes)] for thresh in thresholds}
        true_labels_dict = [[] for _ in range(num_classes)]

        # Processing the dataset once
        _, val_dataloader, _ = self.loader.create_dataloaders()
        for batch in val_dataloader:
            text_tokens, labels = self.process_input(batch)
            labels = labels.to(self.device).float()

            with torch.no_grad():
                output = model(**text_tokens)
                logits = output.logits
                probs = torch.sigmoid(logits)

                for class_idx in range(num_classes):
                    for thresh in thresholds:
                        predicted_binary = (probs[:, class_idx] >= thresh).float()
                        predictions[thresh][class_idx].extend(predicted_binary.cpu().numpy())
                    true_labels_dict[class_idx].extend(labels[:, class_idx].cpu().numpy())

        # Finding the best threshold for each class
        best_thresholds = [0.5] * num_classes
        best_f1s = [0.0] * num_classes

        for class_idx in range(num_classes):
            for thresh in thresholds:
                f1 = f1_score(true_labels_dict[class_idx], predictions[thresh][class_idx], zero_division=0)

                if f1 > best_f1s[class_idx]:
                    best_f1s[class_idx] = f1
                    best_thresholds[class_idx] = thresh

        return best_thresholds


    def evaluate_model_on_test(self, model, test_loader, best=True):
        if best:
            model = self.load_model(model)

        thresholds = self.find_optimal_threshold_per_class(model)

        # Running on test set
        all_binary_results, all_true_labels = [], []

        with torch.no_grad():
            with tqdm(total=len(test_loader), desc="Evaluating on Test Set") as bar:
                for batch in test_loader:
                    text_tokens, test_labels = self.process_input(batch)

                    # Forward pass through the model
                    output = model(**text_tokens)
                    logits = output.logits
                    test_probs = torch.sigmoid(logits)

                    # Converting the logits to a binary tensor using class-specific thresholds
                    binary_result = torch.zeros_like(test_probs)
                    for i in range(len(thresholds)):
                        binary_result[:, i] = (test_probs[:, i] >= thresholds[i]).float()

                    # Append binary results and true labels for this batch
                    all_binary_results.append(binary_result)
                    all_true_labels.append(test_labels)

                    bar.update(1)

        # Concatenate results across batches
        all_binary_results = torch.cat(all_binary_results, dim=0)
        all_true_labels = torch.cat(all_true_labels, dim=0)

        # Calculate metrics table
        metrics_table = self.calculate_metrics_table(
            all_binary_results, all_true_labels
        )
        print(metrics_table)
        print("-- FINISHED EVALUATION --")

        return metrics_table



    def run(self, best=False):
        model = self.initialize_model()

        _, _, test_loader = self.loader.create_dataloaders()

        print("-- Evaluating BERT Model --")
        metrics = self.evaluate_model_on_test(model, test_loader, best=best)

        # Save metrics
        save_path = os.path.join(self.save_results_path, f'{self.model_name}_metrics.csv')
        metrics.to_csv(save_path)

        return metrics

