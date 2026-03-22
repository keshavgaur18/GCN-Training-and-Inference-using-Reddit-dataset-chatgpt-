# Reddit GCN 🌐

A Graph Convolutional Network (GCN) built with PyTorch Geometric to classify posts on the massive **Reddit dataset**. This repository demonstrates scalable graph machine learning using mini-batching techniques over large-scale graph structures.

## 🚀 Overview

The Reddit dataset is a massive graph containing over **230,000 nodes** and **114 million edges**. Performing full-batch training and inference on a graph of this size is highly memory-intensive (requiring upwards of 58GB of RAM!). 

To solve this, the code implements PyTorch Geometric's `NeighborLoader` to perform local neighborhood sampling. This batches the computation, completely avoiding Out-Of-Memory (OOM) errors and allowing the GCN to run seamlessly on standard hardware.

## 📁 Repository Structure

- `reddit_gcn_chatgpt.py`: The main training script. It automatically downloads the dataset, initializes the GCN, trains the model using batched neighborhood sampling (`[25, 10]`), evaluations validation/test accuracy, and saves the final weights.
- `inference.py`: The evaluation script. Loads the pre-trained weights (`gcn_reddit_model.pt`) and performs batched inference over the full graph to generate predictions cleanly.
- `.gitignore`: Prevents massive dataset directories (`data/`) and model weight files (`*.pt`) from being accidentally pushed to GitHub.

## 🛠️ Requirements

To run this code, you need Python and the following key packages:
- `torch`
- `torch_geometric`
- `tqdm`

You can install the primary dependencies via:
```bash
pip install torch
pip install torch_geometric
pip install tqdm
```

## 🧠 Usage

### 1. Training the Model
Run the following command to download the dataset and begin training:
```bash
python reddit_gcn_chatgpt.py
```
This script will:
1. Fetch the Reddit dataset (this may take a moment to download ~2GB).
2. Train a 2-layer GCN for 10 epochs.
3. Output the validation accuracy for each epoch and final test accuracy.
4. Save the trained weights to `gcn_reddit_model.pt`.

### 2. Running Inference
To generate predictions using your pre-trained model on the dataset:
```bash
python inference.py
```
This script properly loads the graph and processes nodes batch-by-batch to quickly output predictions.

## 📈 Model Architecture
- **Layer 1:** `GCNConv` (In: `602` features → Out: `128` hidden channels)
- **Activation:** `ReLU`
- **Layer 2:** `GCNConv` (In: `128` hidden channels → Out: `41` subreddit classes)
- **Optimizer:** Adam (Learning Rate: `0.01`)

---
*Built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).*
