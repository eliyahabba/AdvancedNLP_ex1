import argparse

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Define model names
MODEL_NAMES = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']


def get_data(train_samples, val_samples, test_samples):
    dataset = load_dataset("sst2")
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    if train_samples != -1:
        train_dataset = train_dataset.select(range(train_samples))
    if val_samples != -1:
        val_dataset = val_dataset.select(range(val_samples))
    if test_samples != -1:
        test_dataset = test_dataset.select(range(test_samples))
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    val_dataset = val_dataset.remove_columns(['label'])
    test_dataset = test_dataset.remove_columns(['label'])
    train_dataset = train_dataset.remove_columns(['label'])

    return train_dataset, val_dataset, test_dataset


# define helper functions
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length")


def fine_tune_model(model_name, train_dataset, val_dataset, seed, device, batch_size=32):
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set up tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters())
    num_epochs = 3
    num_training_steps = len(train_dataset) // (batch_size * num_epochs)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Set up data loaders
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

    val_results = []
    # Train model
    # loop = tqdm(zip(val_loader, val_qas_ids_loader), leave=True)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch in tqdm(train_dataloader):
            # move to device
            inputs = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**inputs)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # Log metrics to Weights&Biases
            wandb.log({'train/loss': loss.item(), 'epoch': epoch, 'seed': seed})
            wandb.log({'train/learning_rate': lr_scheduler.get_last_lr()[0], 'epoch': epoch, 'seed': seed})
            wandb.log({'train/batch_size': batch_size, 'epoch': epoch, 'seed': seed})

        # Evaluate model
        val_accuracy = evaluate_model(model, val_dataloader, epoch, seed, device)
        val_results.append(val_accuracy)

    # return the last validation accuracy
    return val_results[-1]


def evaluate_model(model, val_dataloader, epoch, seed, device):
    model.eval()
    val_accuracy = 0
    val_loss = 0
    for batch in val_dataloader:
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            val_loss += outputs.loss
            predictions = torch.argmax(logits, dim=-1)
            val_accuracy += np.sum(predictions.cpu().numpy() == batch['labels'].cpu().numpy()) / len(
                batch['labels'].cpu().numpy())
    val_accuracy /= len(val_dataloader)
    # Log metrics to Weights&Biases
    wandb.log({'eval/accuracy': val_accuracy, 'epoch': epoch, 'seed': seed})
    wandb.log({'eval/loss': val_loss, 'epoch': epoch, 'seed': seed})
    return val_accuracy


def summarize_results(results_df) -> pd.DataFrame:
    """
    Computes the mean and standard deviation of the model accuracies and formats them for saving to the res.txt file.

    Args:
        model_scores (List[dict]): A list of dictionaries containing the mean and standard deviation of the model

    Returns:
        str: A formatted string containing the mean and standard deviation of the model accuracies.
    """
    results = "RESULTS:\n"
    # Compute mean and standard deviation of each model's accuracy (across all seeds)
    results_df['mean'] = results_df.groupby('model_name')['accuracy'].transform('mean')
    results_df['std'] = results_df.groupby('model_name')['accuracy'].transform('std')

    # Add mean and standard deviation to results df
    # Format results
    for model in results_df.to_dict('records'):
        print(f"{model['model_name']}: {model['accuracy']:.3f} ± {model['std']:.3f}\n")
        results += f"{model['model_name']}: {model['accuracy']:.3f}\n"
    return results_df


def predict(model, test_loader, device, output_path):
    """
    Generate predictions for the test set using the given model.

    Args:
    - model (torch.nn.Module): The trained model to use for prediction.
    - test_loader (torch.utils.data.DataLoader): The data loader for the test set.
    - device (torch.device): The device on which to perform the prediction (cpu or gpu).
    - output_path (str): The path to the output file to write the predictions to.

    Returns:
    - None
    """
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc='Predicting on test set'):
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)[0]
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())

    # Write the predictions to the output file
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")


def select_best_model(results_df: pd.DataFrame):
    # select the best model (The model with the best mean, and if there are multiple models with the same mean, select the one with the best accuracy)
    # get all the rows with the best mean and select the one with the best accuracy
    best_model = results_df.loc[results_df['mean'] == results_df['mean'].max()]
    best_model = best_model.loc[best_model['accuracy'] == best_model['accuracy'].max()]
    return best_model


def save_results(summary_results, predictions):
    pass


def tokenize_data(train_dataset, val_dataset, test_dataset, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer), batched=True)
    val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True)
    test_dataset = test_dataset.map(
        lambda examples: tokenizer(examples["sentence"], truncation=True, max_length=tokenizer.model_max_length),
        batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return train_dataset, val_dataset, test_dataset


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune large language models for sentiment analysis')
    parser.add_argument('--num_seeds', type=int, help='number of seeds to use for each model', default=1)
    parser.add_argument('--train_samples', type=int, help='number of samples to use for training or -1 for all',
                        default=1)
    parser.add_argument('--val_samples', type=int, help='number of samples to use for validation or -1 for all',
                        default=1)
    parser.add_argument('--test_samples', type=int, help='number of samples to predict or -1 for all',
                        default=-1)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size to use for training')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up Weights & Biases
    # Set up Weights&Biases logging
    # wandb.login()
    # Load SST-2 dataset
    train_dataset, val_dataset, test_dataset = get_data(args.train_samples, args.val_samples, args.test_samples)

    # Fine-tune each model with several seeds and compute mean and std of validation accuracy
    seeds = list(range(args.num_seeds))
    results = []
    for model_name in MODEL_NAMES:
        # load the tokenizer and tokenize the data

        tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset = tokenize_data(train_dataset,
                                                                                               val_dataset,
                                                                                               test_dataset,
                                                                                               model_name)

        for seed in seeds:
            run_name = f"{model_name}-seed-{seed}--batch_size-{args.batch_size}"
            wandb.init(project='sentiment_analysis', entity='eliyahabba',
                       name=run_name, config={'model_name': model_name, 'seed': seed, 'batch_size': args.batch_size},
                       reinit=True)
            val_accuracy_result = fine_tune_model(model_name, tokenized_train_dataset, tokenized_val_dataset, seed,
                                                  device, args.batch_size)
            result = {'model_name': model_name, 'seed': seed, 'accuracy': val_accuracy_result}
            results.append(result)
    results_df = pd.DataFrame(results)
    # Compute mean and std of validation accuracy for each model
    summary_results = summarize_results(results_df)

    # Select best model
    best_model = select_best_model(summary_results)

    # Predict on test set with best model
    # _ = fine_tune_model(best_model_name, train_dataset, val_dataset, best_seed)
    # predictions = predict(model, tokenized_test_dataset, tokenizer, device, 'test_predictions.txt')

    # Save results to files
    # save_results(summary_results, predictions)


if __name__ == '__main__':
    main()
