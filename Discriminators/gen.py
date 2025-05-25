# -*- coding: utf-8 -*-
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, pipeline
import torch
import numpy as np
from evaluate import load  # Use 'evaluate' library instead
import csv
import argparse
import os
import multiprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def train_model(args):
    # Load dataset for training
    df = pd.read_csv(args.data_path)
    df.columns = ['text', 'label']
    df['label'] = df['label'].astype(int)

    # Split dataset into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # Load tokenizer and model
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Create datasets
    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False
    )

    # Define evaluation metrics
    accuracy_metric = load('accuracy')
    precision_metric = load('precision')
    recall_metric = load('recall')
    f1_metric = load('f1')

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)
        precision = precision_metric.compute(predictions=preds, references=p.label_ids, average='binary')
        recall = recall_metric.compute(predictions=preds, references=p.label_ids, average='binary')
        f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average='binary')
        return {
            'accuracy': accuracy['accuracy'],
            'precision': precision['precision'],
            'recall': recall['recall'],
            'f1': f1['f1']
        }

    # Initialize Trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    print('Model and tokenizer saved successfully.')

def test_model(args):
    # Load the saved model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(args.model_save_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_save_path)

    # Use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using {'GPU' if device == 0 else 'CPU'} for inference.")

    classifier = pipeline(
        'text-classification',
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Predict results for the input files
    results = []
    for filename in os.listdir(args.input_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(args.input_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # Perform prediction with truncation
                prediction = classifier(content, truncation=True, max_length=512)[0]
                label = prediction['label']
                score = prediction['score']

                # Map labels to 'AI' or 'Human'
                mapped_label = 'AI' if label == 'LABEL_1' else 'Human' if label == 'LABEL_0' else 'Unknown'

                file_number = os.path.splitext(filename)[0]
                results.append([file_number, mapped_label, score])

    # Save predictions to a CSV file
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Number', 'Label', 'Confidence Score'])
        writer.writerows(results)

    print('Prediction results saved successfully.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a RoBERTa model and use it for predictions on new files.'
    )
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training dataset CSV.')
    parser.add_argument('--model_save_path', type=str, required=True, help='Directory to save the trained model.')
    parser.add_argument('--input_directory', type=str, required=False, help='Directory containing text files for prediction.')
    parser.add_argument('--results_file', type=str, required=False, help='CSV file to save the prediction results.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test.')

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        if not args.input_directory or not args.results_file:
            print("Error: --input_directory and --results_file are required in test mode.")
            exit(1)
        test_model(args)
