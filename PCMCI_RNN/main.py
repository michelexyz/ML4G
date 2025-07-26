import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Tuple, Set


from data_manager import TimeSeriesDataManager


from PCMCI_RNN import PCMCI_RNN




###############################################################################
# main.py (example usage)
###############################################################################

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(0)
    T, N = 200, 3
    data = np.random.randn(T, N)

    # Suppose variable 0 is influenced by variable 1 (temporal effect)
    for t in range(1, T):
        data[t, 0] += 0.5 * data[t-1, 1]

    pcmci = PCMCI_RNN(
        data=data,
        lookback=5,
        rnn_type='RNN',
        hidden_dim=8,
        significance_threshold=0.05,  # example threshold
        device='cpu'
    )

    # -- PHASE 1: Discover parents (with batched training)
    discovered_parents = pcmci.discover_parents(
        max_condition_set_size=2,
        epochs=10,
        batch_size=16
    )

    # Inspect adjacency after all iterations
    final_adjacency = discovered_parents
    print("\nFinal adjacency (parents) discovered:")
    for y in range(N):
        print(f"Variable {y} has parents: {sorted(list(final_adjacency[y]))}")

    # We can also check adjacency and R^2 matrices:
    for i, (A, R) in enumerate(zip(pcmci.adjacency_history, pcmci.r2_history)):
        print(f"\nIteration {i} adjacency matrix:\n{A}")
        print(f"Iteration {i} R^2 matrix:\n{R}")

    # -- PHASE 2: MCI step
    final_r2_dict = pcmci.mci_step(epochs=10, batch_size=16)
    print("\nMCI step: final R^2 for each (Y, X) link among discovered parents:")
    for (y, x), score in final_r2_dict.items():
        print(f"R^2(Y={y} | X={x}): {score:.3f}")
