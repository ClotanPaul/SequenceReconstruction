# Deep Learning Project: Transformer-Based Sequence Reconstruction

## Description
This project implements a custom transformer architecture for sequence reconstruction tasks. The main objective of the model is to accurately reconstruct sequences from scrambled input. Built with TensorFlow and Keras, the project was executed on Kaggle using a **GPU P100 accelerator** for efficient training and evaluation. It features a robust pipeline for training and evaluation, including early stopping and a custom learning rate scheduler to optimize performance.

## Project Setup
To set up and run the project, execute the following commands in the provided notebook:

```bash
!pip install datasets
!pip install --upgrade keras
```
These commands install all necessary dependencies and ensure compatibility with the provided code.
The project was developed and executed in Kaggle using a P100 GPU, which significantly reduced training time and allowed for more efficient experimentation.

## Dataset
The Dataset was taken from HuggingFace and contains a large (3.5M+ sentence) knowledge base of generic sentences. The HuggingFace page of the dataset can be found at <a href="https://huggingface.co/datasets/community-datasets/generics_kb" target="_blank">this</a> 
In the nodebook, the dataset is installed using the command:
```bash
ds = load_dataset('generics_kb', trust_remote_code=True)['train']
```
