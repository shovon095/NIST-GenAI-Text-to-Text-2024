# -*- coding: utf-8 -*-
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import os
import csv
import argparse

def main(args):
    # Load the trained model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(args.model_save_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_save_path)

    # Use the trained model for prediction on new files
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=False)

    # Directory containing text files for prediction
    files_directory = args.input_directory

    # Predict and save results
    results = []
    for filename in os.listdir(files_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(files_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                prediction = classifier(content)[0]
                label = 'AI' if prediction['label'] == 'LABEL_1' else 'Human'
                score = prediction['score']  # Confidence score
                file_number = filename.split('.')[0]
                results.append([file_number, label, score])

    # Save results to CSV
    with open(args.results_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Number', 'Label', 'Confidence Score'])
        writer.writerows(results)

    print('Results saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a trained RoBERTa model and use it for predictions on new files.')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to the saved trained model.')
    parser.add_argument('--input_directory', type=str, required=True, help='Directory containing the text files for prediction.')
    parser.add_argument('--results_file', type=str, required=True, help='Path to save the prediction results as a CSV file.')

    args = parser.parse_args()

    main(args)
