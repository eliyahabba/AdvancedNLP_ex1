# Advanced NLP Exercise 1: Fine Tuning

This is the code base for ANLP HUJI course exercise 1, fine tuning pretrained models to perform sentiment analysis on the SST2 dataset.

# Install
``` pip install -r requirements.txt ```

# Fine-Tune and Predict on Test Set
Run:

``` python ex1.py <number of seeds> <number of training samples> <number of validation samples> <number of prediction samples> ```

Generated files are res.txt, showing model's performance, training time and prediction time, and prediction.txt, containing prediction results for all test samples.
