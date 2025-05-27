"""
Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

"""
This file includes functions for training.
"""

import io

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

import torch


def train_model(model, features, edges, labels, test_nodes, loss_func, optimizer, trn_labels, epochs, edge_weights=None, contrastive=True):
    """
    Trains the given model using supervised or contrastive PU loss.

    Args:
        model (nn.Module): The model to be trained.
        features (Tensor): Node feature matrix of shape [N, D].
        edges (Tensor): Edge index tensor of shape [2, E].
        labels (Tensor): Ground truth labels (used for test evaluation).
        test_nodes (Tensor): Indices of nodes used for testing.
        loss_func (nn.Module): Loss function module (e.g., Hope_loss_with_contrastive).
        optimizer (Optimizer): Optimizer for parameter updates.
        trn_labels (Tensor): Training labels (0 for unlabeled).
        epochs (int): Number of training epochs.
        edge_weights (list[Tensor], optional): List of edge weight tensors (one per view).
        contrastive (bool): Whether to include contrastive loss term.

    Returns:
        int: Best epoch number (with lowest training loss).
        DataFrame: Training log with columns [epoch, trn_loss, test_f1, test_acc].
        float: Best training loss value.
        nn.Module: Model loaded with the best parameters.
    """
    logs = []
    saved_model, best_epoch, best_f1 = io.BytesIO(), -1, -1
    torch.save(model.state_dict(), saved_model)
    best_loss = np.inf

    for epoch in range(epochs + 1):
        model.train()
        if edge_weights == None:
            out = model(features, edges)
        else:
            out = model(features, edges, edge_weights)

        if contrastive:
            embeddings = model.embedding(features, edges, edge_weights)
            loss = loss_func(out, trn_labels, embeddings)
        else:
            loss = loss_func(out, trn_labels)

        if epoch > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_f1, test_acc = evaluate_model(model, features, edges, labels, test_nodes, edge_weights)

        logs.append((epoch, np.round(loss.item(), 4), np.round(test_f1, 4), np.round(test_acc, 4)))
        if loss.item() < best_loss:
            best_epoch = epoch
            best_loss = loss.item()
            saved_model.seek(0)
            torch.save(model.state_dict(), saved_model)

    saved_model.seek(0)
    model.load_state_dict(torch.load(saved_model))

    columns = ['epoch', 'trn_loss', 'test_f1', 'test_acc']
    print(f'Epochs: {logs[best_epoch][0]}, Loss: {logs[best_epoch][1]}, '
          f'Test F1: {logs[best_epoch][2]}, Test Accuracy: {logs[best_epoch][3]}')
    return best_epoch, pd.DataFrame(logs, columns=columns), best_loss, model

def evaluate_model(model, features, edges, labels, test_nodes, edge_weights=None):
    """
    Evaluates the model on a test node subset using F1 and accuracy.

    Args:
        model (nn.Module): The model to evaluate.
        features (Tensor): Node feature matrix of shape [N, D].
        edges (Tensor): Edge index tensor of shape [2, E].
        labels (Tensor): Ground truth labels for all nodes [N].
        test_nodes (Tensor): Indices of test nodes.
        edge_weights (list[Tensor], optional): Edge weights per view, if applicable.

    Returns:
        float: Macro F1 score on the test set.
        float: Accuracy on the test set.
    """
    model.eval()
    with torch.no_grad():
        if edge_weights == None:
            out = model(features, edges).cpu()
        else:
            out = model(features, edges, edge_weights).cpu()
        out_labels = torch.argmax(out, dim=1)

    test_f1 = f1_score(labels[test_nodes], out_labels[test_nodes], average='macro')
    test_acc = accuracy_score(labels[test_nodes], out_labels[test_nodes])
    return test_f1, test_acc
