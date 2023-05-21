import argparse
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import transformers
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments

import wandb

# Define model names
MODEL_NAMES = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']

# use the default batch size and number of epochs from transformers training arguments
DEFAULT_BATCH_SIZE = int(TrainingArguments.per_device_train_batch_size)
DEFAULT_NUM_EPOCHS = int(TrainingArguments.num_train_epochs)


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
    train_dataset = train_dataset.rename_column('label', 'labels')
    val_dataset = val_dataset.rename_column('label', 'labels')
    test_dataset = test_dataset.rename_column('label', 'labels')

    return train_dataset, val_dataset, test_dataset


# define helper functions
def tokenize_function_with_padding(examples, tokenizer):
    return tokenizer(examples["sentence"], truncation=True, padding='max_length')


def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def fine_tune_model(model_name, train_dataset, val_dataset, seed, num_epochs=DEFAULT_NUM_EPOCHS,
                    batch_size=DEFAULT_BATCH_SIZE):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",
                                      seed=seed, num_train_epochs=num_epochs,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # return model and the accuracy on the validation set
    return trainer, trainer.evaluate()


def summarize_results(results_metadata: list) -> pd.DataFrame:
    """
    Computes the mean and standard deviation of the model accuracies and formats them for saving to the res.txt file.

    Args:
        model_scores (List[dict]): A list of dictionaries containing the mean and standard deviation of the model

    Returns:
        str: A formatted string containing the mean and standard deviation of the model accuracies.
    """
    # Compute mean and standard deviation of each model's accuracy (across all seeds)
    results_df = pd.DataFrame(results_metadata)
    results_df['mean'] = results_df.groupby('model_name')['accuracy'].transform('mean')
    results_df['std'] = results_df.groupby('model_name')['accuracy'].transform('std')

    # Save the results to a file with name res.txt, with the following format:
    # <model name>,<mean accuracy> +- <accuracy std>
    models_results = []
    for model in results_df.groupby('model_name').first().reset_index().to_dict('records'):
        models_results.append(f"{model['model_name']},{model['mean']:.2f} +- {model['std']:.2f}")
    with open('res.txt', 'w') as f:
        f.write('\n'.join(models_results))
    with open('res.txt', 'a') as f:
        f.write('---\n')

    return results_df


def predict(test_dataset, best_model_metadata: dict, best_model_trainer: Trainer):
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
    # Predict on test set with best model
    tokenized_test_dataset = tokenize_test_dataset(test_dataset, best_model_metadata['model_name'])
    # test_trainer = best_model['trainer']
    # change the batch size of the trainer to 1
    best_model_trainer.args.per_device_eval_batch_size = 1
    # best_model_trainer.compute_loss = False
    predict_start_time = time.time()
    predictions = best_model_trainer.predict(tokenized_test_dataset)
    predict_time = time.time() - predict_start_time
    return predictions, predict_time


def select_best_model(results: List[dict], batch_size: int) -> Tuple[dict, Trainer]:
    # convert the results to a dataframe (with columns: model_name, seed, accuracy)
    # remove the models from the results
    results_metadata = [{key: val for key, val in result.items() if key != 'trainer'} for result in results]

    # Compute mean and std of validation accuracy for each model
    results_df = summarize_results(results_metadata)

    # select the best model (The model with the best mean, and if there are multiple models with the same mean, select the one with the best accuracy)
    # get all the rows with the best mean and select the one with the best accuracy
    best_models = results_df.loc[results_df['mean'] == results_df['mean'].max()]
    best_models = best_models.loc[best_models['accuracy'] == best_models['accuracy'].max()]
    best_model = best_models.iloc[0]  # get the first row
    best_model_metadata = {'model_name': best_model['model_name'], 'seed': best_model['seed']}
    best_model_trainer = get_best_trainer(best_model_metadata, results)
    # get the trainer and the metadata of the best model
    return best_model_metadata, best_model_trainer


def get_best_trainer(best_model_metadata: dict, results):
    best_model = None
    for result in results:
        if result['model_name'] == best_model_metadata['model_name'] and result['seed'] == best_model_metadata['seed']:
            best_model = result
            break
    if best_model is None:
        raise Exception('No model was selected')
    return best_model['trainer']


def get_best_trainer_old(best_model_metadata: dict, batch_size: int):
    model_name = best_model_metadata['model_name']
    seed = best_model_metadata['seed']
    run_name = f"{model_name}-seed-{seed}--batch_size-{batch_size}--wo-ls"
    # load the trainer

    model = AutoModelForSequenceClassification.from_pretrained(run_name)
    trainer = Trainer(model_init=model)
    return trainer


def save_results(sentences, predictions):
    # save the predictions to a file with the name predictions.txt
    with open('predictions.txt', 'w') as f:
        # save with the format of: "<input sentence>###<predicted label 0 or 1>"
        for sentence, prediction_array in zip(sentences, predictions.predictions):
            prediction_label = np.argmax(prediction_array, axis=-1)
            f.write(f"{sentence}###{prediction_label}\n")


def tokenize_data(train_dataset, val_dataset, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function_with_padding(examples, tokenizer), batched=True,
        load_from_cache_file=False)
    val_dataset = val_dataset.map(
        lambda examples: tokenize_function_with_padding(examples, tokenizer), batched=True,
        load_from_cache_file=False)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_dataset = train_dataset.remove_columns(['sentence', 'idx'])
    val_dataset = val_dataset.remove_columns(['sentence', 'idx'])
    return train_dataset, val_dataset


def tokenize_test_dataset(test_dataset, model_name):
    # change all the labels to 0
    test_dataset = test_dataset.map(lambda examples: {'labels': 0})
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    test_dataset = test_dataset.map(
        lambda examples: tokenizer(examples["sentence"], truncation=True), load_from_cache_file=False)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset = test_dataset.remove_columns(['sentence', 'idx'])
    return test_dataset


def train_model(model_name, tokenized_train_dataset, tokenized_val_dataset, seed, num_epochs=DEFAULT_NUM_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE):
    run_name = f"{model_name}-seed-{seed}--batch_size-{batch_size}"
    wandb.init(project='sentiment_analysis', entity='eliyahabba',
               name=run_name, config={'model_name': model_name, 'seed': seed, 'batch_size': batch_size},
               reinit=True)

    trainer, val_accuracy_result = fine_tune_model(model_name=model_name,
                                                   train_dataset=tokenized_train_dataset,
                                                   val_dataset=tokenized_val_dataset,
                                                   seed=seed, num_epochs=num_epochs,
                                                   batch_size=batch_size)
    result = {'model_name': model_name, 'seed': seed, 'accuracy': val_accuracy_result['eval_accuracy'],
              'trainer': trainer}
    wandb.finish()
    # # save the model
    # trainer.save_model(f"{run_name}")
    # # torch.save(trainer.state, f"trainer_state_{run_name}-model.pt")  # Save the Trainer's state
    return result


def train_model_on_many_seeds(model_name, train_dataset, val_dataset, seeds, num_epochs=DEFAULT_NUM_EPOCHS,
                              batch_size=DEFAULT_BATCH_SIZE):
    tokenized_train_dataset, tokenized_val_dataset = tokenize_data(train_dataset, val_dataset, model_name)
    model_results = []
    for seed in seeds:
        transformers.set_seed(seed)
        result = train_model(model_name, tokenized_train_dataset, tokenized_val_dataset, seed, num_epochs, batch_size)
        model_results.append(result)
    return model_results


def main(args):
    # Load SST-2 dataset
    train_dataset, val_dataset, test_dataset = get_data(args.train_samples, args.val_samples, args.test_samples)

    # Fine-tune each model with several seeds and compute mean and std of validation accuracy
    seeds = list(range(args.num_seeds))
    results = []
    start_train_time = time.time()
    for model_name in MODEL_NAMES:
        # load the tokenizer and tokenize the data
        results.extend(train_model_on_many_seeds(model_name, train_dataset, val_dataset, seeds, args.epochs,
                                                 args.batch_size))
    end_train_time = time.time()
    # Select best model
    best_model_metadata, best_model_trainer = select_best_model(results, args.batch_size)
    # Predict on test set with best model
    predict_start_time = time.time()
    predictions, predict_time = predict(test_dataset, best_model_metadata, best_model_trainer)
    predict_end_time = time.time()

    with open('res.txt', 'a') as f:
        f.write(f"train time,{(end_train_time - start_train_time):.2f}\n")
        f.write(f"predict time,{(predict_end_time - predict_start_time):.2f}\n")
    # Save results to files
    save_results(test_dataset['sentence'], predictions)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune large language models for sentiment analysis')
    parser.add_argument('num_seeds', type=int, help='number of seeds to use for each model', default=3)
    parser.add_argument('train_samples', type=int, help='number of samples to use for training or -1 for all',
                        default=-1)
    parser.add_argument('val_samples', type=int, help='number of samples to use for validation or -1 for all',
                        default=-1)
    parser.add_argument('test_samples', type=int, help='number of samples to predict or -1 for all',
                        default=-1)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size to use for training')
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='number of epochs to train each model')
    args = parser.parse_args()
    main(args)
