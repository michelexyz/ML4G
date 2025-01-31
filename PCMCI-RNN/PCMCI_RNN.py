import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Tuple, Set


from data_manager import TimeSeriesDataManager
from models import SimpleRNN, train_model, compute_r2

class PCMCI_RNN:
    """
    A class that implements a PCMCI-like causal discovery approach with:
      - RNN-based conditional dependencies
      - Non-split training for residual generation
      - Split training for significance test
      - NxN R^2 matrix recorded each iteration
      - Batching for training
    """

    def __init__(
        self,
        data: np.ndarray,
        lookback: int = 5,
        rnn_type: str = 'RNN',
        hidden_dim: int = 8,
        significance_threshold: float = 0.1,
        device: str = 'cpu',
        split_percentage: float = 0.7
    ):
        """
        Args:
            data: shape (T, N)
            lookback: how many time steps the RNN sees
            rnn_type: 'RNN' or 'LSTM'
            hidden_dim: size of RNN hidden dimension
            significance_threshold: cutoff for discarding links
            device: 'cpu' or 'cuda'
        """
        self.data = data
        self.num_timesteps, self.num_vars = data.shape
        self.lookback = lookback
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.significance_threshold = significance_threshold
        self.device = device

        # For demonstration, we fix a train/val split here
        self.split_idx = int(split_percentage * self.num_timesteps)

       
        self.candidate_parents = {
            var: set(range(self.num_vars))  # not excluding var
            for var in range(self.num_vars)
        }

        # Manager for slicing time-series to RNN inputs
        self.manager = TimeSeriesDataManager(data, lookback=lookback, device=device)

        # We keep track of last iteration's R^2 for each link (X->Y)
        # Initialize to 0.0
        self.r2_scores = {
            (y, x): 0.0
            for y in range(self.num_vars)
            for x in range(self.num_vars)
        }

        # We store adjacency plus an NxN R^2 matrix each iteration
        self.adjacency_history = []
        self.r2_history = []

    # --------------------------------------------------------------------------
    # PHASE 1: PARENT DISCOVERY
    # --------------------------------------------------------------------------

    def discover_parents(
        self, 
        max_condition_set_size: int = 3, 
        epochs: int = 20, 
        batch_size: int = 32
    ):
        """
        Builds the adjacency set for each variable using:
          For cond_size in [0 .. max_condition_set_size]:
            For each variable Y:
              For each candidate X in candidate_parents[Y]:
                1) Identify cond_set (top cond_size parents by last iteration's R^2, excluding X)
                2) Build model cond_set->Y on full data => residual(Y)
                3) Then X->(residual Y) on train/val => R^2
                4) If R^2 < threshold => remove X from candidate_parents[Y]

        After each cond_size pass, records adjacency matrix + NxN R^2 matrix.
        """
        for cond_size in range(max_condition_set_size + 1):
            print(f"\n=== Condition set size = {cond_size} ===")
            for y in range(self.num_vars):
                # We'll check each candidate parent X
                to_remove = []
                
                # Sort candidate parents by last iteration's R^2 (descending)
                current_candidates = list(self.candidate_parents[y])
                current_candidates.sort(key=lambda x: self.r2_scores[(y, x)], reverse=True)

                if len(current_candidates) < 1 or len(current_candidates) <= cond_size:
                    print(f"Skipping variable {y} at cond_size {cond_size}, not enough candidates")
                    continue

                for x in current_candidates:
                    # Build the cond_set from the best scorers, excluding 'x'
                    cond_set = [p for p in current_candidates if p != x]
                    
                    cond_set = cond_set[:cond_size]

                    

                    if cond_size > 0:
                        # (1) Build cond_set->Y model on full data => residual(Y)
                        Y_res = self._condition_through_residuals(
                            y, cond_set, epochs=epochs, batch_size=batch_size
                        )
                        # (2) Test link x->Y_res on train/val => R^2
                        r2_val = self._test_link_on_split(
                            x, Y_res, epochs=epochs, batch_size=batch_size
                        )
                    else: 
                        # No conditioning, just test X->Y
                        r2_val = self._test_link_on_split(
                            x, target_var=y, epochs=epochs, batch_size=batch_size
                        )
                    self.r2_scores[(y, x)] = r2_val  # store for next iteration

                    if r2_val < self.significance_threshold:
                        to_remove.append(x)
                
                # Remove the non-significant parents
                for x_bad in to_remove:
                    self.candidate_parents[y].discard(x_bad)

            # After we've gone through all variables/candidates at this cond_size,
            # record adjacency + NxN R^2
            self._record_iteration_state()

        return self.candidate_parents

    # --------------------------------------------------------------------------
    # PHASE 2: MCI STEP
    # --------------------------------------------------------------------------

    def mci_step(self, epochs: int = 20, batch_size: int = 32) -> Dict[Tuple[int, int], float]:
        """
        
        """
        final_r2 = {}
        for y in range(self.num_vars):
            parents_of_y = list(self.candidate_parents[y])  # includes Y if not removed
            # remove x from parents_of_y

            if x in parents_of_y:
                
                parents_of_y.remove(x)
                

        
            if parents_of_y > 0:
                # Build (parents_of_y)->Y => residual
                # remove x from parents_of_y if present

                
                Y_res = self._condition_through_residuals(
                    y, parents_of_y, epochs=epochs, batch_size=batch_size
                )
                
            for x in range(self.num_vars):

                parents_of_x = list(self.candidate_parents[x])

                if x in parents_of_y:
                    # Skip Y itself
                    continue
                # TODO
               






            parents_of_y = list(self.candidate_parents[y])  # includes Y if not removed

            if len(parents_of_y)> 0:

                # Build (parents_of_y)->Y => residual
                Y_res = self._condition_through_residuals(
                    y, parents_of_y, epochs=epochs, batch_size=batch_size
                )
            # Now for each x in parents_of_y, test x->Y_res
            for x in parents_of_y:
                r2_val = self._test_link_on_split(
                    x, Y_res, epochs=epochs, batch_size=batch_size
                )
                final_r2[(y, x)] = r2_val
        return final_r2

    # --------------------------------------------------------------------------
    # UTILS
    # --------------------------------------------------------------------------

    def _condition_through_residuals(
        self, 
        target_var: int,
        cond_set: List[int],
        epochs: int = 20,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Trains cond_set->Y on the FULL dataset (no train/val split),
        obtains predictions, returns residual = Y - Y_pred.
        Residual is returned as a length T array (with zeros for the first 
        'lookback' timesteps that can't be predicted).
        """

        if len(cond_set) < 1:
            raise ValueError("Conditioning set must have at least one variable")
        # Build dataset for full range
        X_full, Y_full = self.manager.create_rnn_dataset(
            target_var_idx=target_var,
            predictor_vars=cond_set,
            start_idx=0,
            end_idx=self.num_timesteps
        )

        model = SimpleRNN(
            input_dim=len(cond_set), 
            hidden_dim=self.hidden_dim,
            rnn_type=self.rnn_type
        ).to(self.device)

        # Train on entire dataset
        train_model(model, X_full, Y_full, epochs=epochs, lr=1e-3, batch_size=batch_size)

        # Get predictions
        model.eval()
        with torch.no_grad():
            Y_hat = model(X_full).cpu().numpy()

        # Convert to full length T array by padding the first 'lookback' steps with zeros
        residual = np.zeros((self.num_timesteps,))
        valid_range = range(self.lookback, self.num_timesteps)
        Y_true = Y_full.cpu().numpy()
        for i, t in enumerate(valid_range):
            residual[t] = Y_true[i] - Y_hat[i]

        return residual

    def _test_link_on_split(
        self,
        candidate_var: int,
        conditioned_series: np.ndarray = None,
        target_var: int = None,
        epochs: int = 20,
        batch_size: int = 32,
        
    ) -> float:
        """
        Tests X->Y ,given that Y could have been residualized and passed as (conditioned_series).
        Splits data into [0 : split_idx] for training, [split_idx : end] for validation.
        Returns R^2 on validation.
        """
        if target_var is None and conditioned_series is None:
            raise ValueError("Must provide conditioned_series or target_var")
        if target_var is None:
            target_var = 0
            
        # Build training set
        X_train, Y_train = self.manager.create_rnn_dataset(
            target_var_idx=target_var,  # We'll store the residual in 'target_series' as if it's a single "variable"
            predictor_vars=[candidate_var],
            start_idx=0,
            end_idx=self.split_idx,
            override_target=conditioned_series
        )
        # Build validation set
        X_val, Y_val = self.manager.create_rnn_dataset(
            target_var_idx=0,
            predictor_vars=[candidate_var],
            start_idx=self.split_idx,
            end_idx=self.num_timesteps,
            override_target=conditioned_series
        )

        if X_train.shape[0] < 1 or X_val.shape[0] < 1:
            return 0.0  # Not enough data, return minimal R^2

        model = SimpleRNN(
            input_dim=1,  # only candidate_var as input
            hidden_dim=self.hidden_dim,
            rnn_type=self.rnn_type
        ).to(self.device)

        train_model(model, X_train, Y_train, epochs=epochs, lr=1e-3, batch_size=batch_size)

        model.eval()
        with torch.no_grad():
            Y_val_hat = model(X_val).cpu().numpy()
            r2_val = compute_r2(Y_val.cpu().numpy(), Y_val_hat, Y_train.mean().item())
        return r2_val

    def _record_iteration_state(self):
        """
        Builds an NxN adjacency matrix for the current iteration
        AND an NxN R^2 matrix.
        
        We store both in self.adjacency_history and self.r2_history.
        """
        A = np.zeros((self.num_vars, self.num_vars), dtype=int)
        R2_mat = np.zeros((self.num_vars, self.num_vars), dtype=float)
        
        for y in range(self.num_vars):
            for x in range(self.num_vars):
                # Adjacency is 1 if x remains a candidate parent of y
                if x in self.candidate_parents[y]:
                    A[y, x] = 1
                    R2_mat[y, x] = self.r2_scores[(y, x)]
                else:
                    A[y, x] = 0
                    R2_mat[y, x] = 0.0
        
        self.adjacency_history.append(A)
        self.r2_history.append(R2_mat)


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
        data[t, 1] += 0.5 * data[t-1, 2]

    pcmci = PCMCI_RNN(
        data=data,
        lookback=5,
        rnn_type='LSTM',
        hidden_dim=100,
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
