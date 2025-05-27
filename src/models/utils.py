"""
Accurate Graph-based Multi-Positive Unlabeled Learning via Disentangled Multi-view Feature Propagation
Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Hoyoung Yoon (crazy8597@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyunpark@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
"""

"""
This file includes utility functions.
"""

import numpy as np
import torch


def compute_edge_weights(features, edge_index, normalize=True):
    """
    Computes edge weights based on the cosine similarity of connected node features.

    Args:
        features (Tensor): Node features of shape [N, D].
        edge_index (Tensor): Edge indices of shape [2, E].
        normalize (bool): Whether to normalize features using z-score.

    Returns:
        Tensor: Edge weights of shape [E].
    """
    src, tgt = edge_index

    if normalize:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        features = (features - mean) / (std + 1e-6)

    edge_weights = torch.sum(features[src] * features[tgt], dim=1)

    return edge_weights

def normalize_edge_weights(*edge_weights):
    """
    Normalizes multiple edge weight tensors using softmax over the view axis.

    Args:
        *edge_weights (Tensor): One or more edge weight tensors of shape [E].

    Returns:
        List[Tensor]: Softmax-normalized edge weights for each view.
    """
    stacked_weights = torch.stack(edge_weights, dim=0)
    normalized_edge_weights = torch.softmax(stacked_weights, dim=0)
    normalized_edge_weights = [weight for weight in normalized_edge_weights]

    return normalized_edge_weights

def compute_prior(trn_labels, num_class=2):
    """
    Computes class prior probabilities from labeled training data.

    Args:
        trn_labels (array-like): Training labels of shape [N].
        num_class (int): Number of classes.

    Returns:
        List[float]: Class prior probabilities.
    """
    prior = []
    count = np.bincount(trn_labels, minlength=int(num_class))
    for i in range(len(count)):
        prior.append(count[i] / len(trn_labels))
    return prior

def estimate_prior(trn_nodes, predictions, num_class, trn_labels):
    """
    Estimates class prior by using model predictions for unlabeled nodes.

    Args:
        trn_nodes (array-like): Indices of training nodes.
        predictions (Tensor): Model output logits [N, C].
        num_class (int): Number of classes.
        trn_labels (Tensor): PU-style training labels [N].

    Returns:
        Tuple[List[float], Tensor]:
            - Estimated class priors.
            - Soft pseudo-labels for all nodes [N].
    """
    expected_trn_labels = torch.argmax(predictions, dim=1).float()
    expected_trn_labels[trn_labels != 0] = trn_labels[trn_labels != 0]

    prior_labels = expected_trn_labels.cpu().numpy()
    prior_labels = torch.from_numpy(np.delete(prior_labels, trn_nodes))
    prior = compute_prior(prior_labels, num_class=num_class)

    # expected_trn_labels = torch.argmax(predictions, dim=1)
    return prior, expected_trn_labels
