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
Additionally, a **requirements.txt** file was added to the project in order to be able to replicate the results outside of the kaggle environment.

## Dataset
The Dataset was taken from HuggingFace and contains a large (3.5M+ sentence) knowledge base of generic sentences. The HuggingFace page of the dataset can be found <a href="https://huggingface.co/datasets/community-datasets/generics_kb" target="_blank">here</a>.
In the nodebook, the dataset is installed using the command:
```bash
ds = load_dataset('generics_kb', trust_remote_code=True)['train']
```
## Evaluation Metric
This project uses a similarity-based metric to evaluate the predictions. The metric is calculated as follows:

1. Determine the **longest common sequence** shared by the original string and the predicted string.
2. Compute the ratio of the length of this sequence to the **maximum length** of either the original or predicted string.

### Formula:
```text
Score = Length of Longest Matching Sequence(original, predicted) / Max(Length(original), Length(predicted))
```

## Model Description
The project implements a custom transformer-based model designed for sequence reconstruction tasks. This architecture is inspired by the Transformer model, leveraging self-attention mechanisms to capture relationships between tokens in a sequence. It consists of an encoder-decoder structure.

### Hyperparameters
The model's performance is fine-tuned using the following hyperparameters:
- **Embedding Dimension**: 128  
  Represents the size of the token embeddings used as input to the model.
  
- **Latent Dimension**: 600  
  Defines the size of the internal dense layers within the transformer blocks.

- **Number of Heads in Multi-Head Attention**: 14  
  Specifies the number of parallel attention mechanisms used for capturing relationships between tokens.

- **Number of Layers**: 5  
  Indicates the depth of the model, i.e., the number of stacked transformer blocks.

- **Dropout Rate**: 0.15  
  Used to regularize the model and prevent overfitting during training.

### Optimizer
The model is trained using the **AdamW optimizer**, which combines the Adam optimization algorithm with weight decay regularization. This optimizer ensures:
1. **Efficient convergence**: Adaptive learning rates help improve training stability.
2. **Regularization**: Weight decay reduces overfitting by penalizing large weights in the model.

Additionally, a **custom learning rate scheduler** is used to optimize training by stabilizing convergence during the initial steps and gradually reducing the learning rate in later epochs for fine-tuning.

### Additional Components
- **Early Stopping**:  
  Training halts if the model's performance does not improve after 5 consecutive epochs, ensuring efficiency and preventing overfitting.

This configuration ensures a balance between model complexity and training stability, allowing the model to effectively learn sequence reconstruction tasks.

## Results
The final model achieved an average score of approximately **0.54** using the defined evaluation metric. This performance significantly exceeds the estimated baseline for a random sequence reconstruction model, which is around **0.19** with a standard deviation of **0.06**.

The weights of the trained model are provided in the repository.

