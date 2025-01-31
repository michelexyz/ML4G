
import numpy as np
import torch

from typing import List, Tuple, Optional
class TimeSeriesDataManager:
    """
    Handles slicing time-series into (X, Y) shapes for RNNs, including lookback windows.
    """

    def __init__(
        self, 
        data: np.ndarray, 
        lookback: int = 5, 
        device: str = 'cpu'
    ):
        """
        Args:
            data: shape (T, N). T=timesteps, N=number of variables
            lookback: number of time steps for the RNN input
            device: 'cpu' or 'cuda'
        """
        self.data = data
        self.lookback = lookback
        self.num_timesteps, self.num_vars = data.shape
        self.device = device
    
    def create_rnn_dataset(
        self, 
        target_var_idx: Optional[int], 
        predictor_vars: List[int],
        start_idx: int = 0,
        end_idx: int = None,
        override_target: np.ndarray = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build (X, Y) for training an RNN to predict target_var_idx from 'conditioning_vars'
        over [start_idx : end_idx].

        If 'override_target' is provided, it is used in place of 'self.data[:, target_var_idx]'.
        """
        if target_var_idx is None and override_target is None:
            raise ValueError("Either target_var_idx or override_target must be provided.")
        if end_idx is None:
            end_idx = self.num_timesteps
        
        X_list, Y_list = [], []
        for t in range(start_idx + self.lookback, end_idx):
            # shape: (lookback, #cond_vars)
            X_t = self.data[t-self.lookback:t, predictor_vars]
            X_list.append(X_t)
            if override_target is not None:
                Y_list.append(override_target[t])
            else:
                Y_list.append(self.data[t, target_var_idx])
        
        X_arr = np.stack(X_list, axis=0)  # (num_samples, lookback, len(cond_vars))
        Y_arr = np.array(Y_list)         # (num_samples,)

        X_torch = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
        Y_torch = torch.tensor(Y_arr, dtype=torch.float32).to(self.device)
        return X_torch, Y_torch