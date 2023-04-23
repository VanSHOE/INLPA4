# ELMo

This repository contains code for training and evaluating ELMo model on 2 datasets: `multinli` and `sst`.

1. `multinli` (Multi-Genre Natural Language Inference): This dataset is a collection of sentence pairs that have been manually labeled for three types of relationship: entailment, contradiction, and neutral. Each sentence pair is drawn from a different genre, such as fiction, telephone conversations, and government reports. The dataset is intended to evaluate the ability of models to perform natural language inference across a variety of domains and writing styles. The original version of the dataset contains about 400,000 sentence pairs for training and 10,000 sentence pairs for testing.

2. `sst` (Stanford Sentiment Treebank): This dataset consists of movie reviews and their associated sentiment labels, which are on a five-point scale ranging from very negative to very positive. The dataset also includes a parse tree for each sentence, which allows models to incorporate structural information about the sentence in their predictions. The dataset contains about 11,000 movie reviews for training and 2,500 movie reviews for testing.

Both of these datasets have become standard benchmarks for evaluating natural language processing models, and many state-of-the-art models have been trained and evaluated on them. The ELMo model is one such model that has shown strong performance on both datasets.

## Files

There are two files for the ELMo model:

1. `sst.py`: This file trains the ELMo model on the sst dataset and saves the trained model to disk.
2. `nli.py`: This file trains the ELMo model on the nli dataset and saves the trained model to disk.
   Both of these files use the same architecture for the ELMo model but train it on different datasets. The trained models can later be used for downstream tasks such as sentiment analysis and natural language inference.

## Usage

To train and save ELMo model on sst dataset, run the `sst.py` script with the following command:

`python sst.py`

To train and save ELMo model on nli dataset, run the `nli.py` script with the following command:

`python nli.py`

To use pretrained models, just ensure that the models are in the same directory as the scripts and the scripts will automatically load the models otherwise they will train the models from scratch.

Big Files: https://1drv.ms/f/s!AlS9diCw3ZVTqS4HspsCAn_O28kM?e=BKQiwm
