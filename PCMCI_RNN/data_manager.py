
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
        predictor_vars: Optional[int],
        start_idx: int = 0,
        end_idx: int = None,
        override_predictors: np.ndarray = None,
        override_target: np.ndarray = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build (X, Y) for training an RNN to predict 'target_var_idx' using 'predictor_vars'.
        over [start_idx : end_idx].

        If 'override_target' is provided, it is used in place of 'self.data[:, target_var_idx]'.
        """
        if target_var_idx is None and override_target is None:
            raise ValueError("Either target_var_idx or override_target must be provided.")
        if predictor_vars is None and override_predictors is None:
            raise ValueError("Either predictor_vars or override_predictors must be provided.")
        if end_idx is None:
            end_idx = self.num_timesteps
        
        X_list, Y_list = [], []
        for t in range(start_idx + self.lookback, end_idx):
            
            if override_predictors is not None:
                X_t = override_predictors[t-self.lookback:t]
            else:
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
    
    def create_rnn_dataset_split(
        self,
        target_var_idx: Optional[int],
        predictor_vars: Optional[int],
        train_split: float = 0.8,
        override_predictors: np.ndarray = None,
        override_target: np.ndarray = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the (X, Y) datasets for training an RNN with a sliding window approach and split them into 
        training and validation sets based on the given train_split percentage.

        Parameters:
            target_var_idx (Optional[int]): The column index for the target variable. Must be provided unless override_target is.
            predictor_vars (Optional[int]): The column index for predictor variables. Must be provided unless override_predictors is.
            train_split (float): The fraction (between 0 and 1) of the dataset to be used for training.
            override_predictors (np.ndarray): If provided, used instead of extracting predictor_vars from self.data.
            override_target (np.ndarray): If provided, used instead of extracting target_var_idx from self.data.
            
        Returns:
            A tuple: (X_train, Y_train, X_val, Y_val)
              - X_train: Tensor of shape (train_samples, lookback, num_predictor_features)
              - Y_train: Tensor of shape (train_samples,)
              - X_val:   Tensor of shape (val_samples, lookback, num_predictor_features)
              - Y_val:   Tensor of shape (val_samples,)
        """
        # Ensure that we have a target specification.
        if target_var_idx is None and override_target is None:
            raise ValueError("Either target_var_idx or override_target must be provided.")
        # Ensure that we have predictors specified.
        if predictor_vars is None and override_predictors is None:
            raise ValueError("Either predictor_vars or override_predictors must be provided.")

        # Get predictors as a torch tensor.
        if override_predictors is not None:
            predictors = torch.tensor(override_predictors, dtype=torch.float32, device=self.device)
        else:
            # Use predictor_vars to select columns from self.data.
            predictors = torch.tensor(self.data[:, predictor_vars], dtype=torch.float32, device=self.device)
        # predictors shape: (num_timesteps, num_features)

        if predictors.dim() == 1:
            predictors = predictors.unsqueeze(1)
        # Use torch.unfold to extract sliding windows along the time dimension.
        # This creates a tensor of shape: (num_timesteps - lookback + 1, num_features, lookback)
        X_full = predictors.unfold(dimension=0, size=self.lookback, step=1)
        
        X_full = X_full.permute(0, 2, 1)# This creates a tensor of shape: (num_timesteps - lookback + 1, lookback, num_features)

        # We need exactly (num_timesteps - lookback) windows so that each window has a corresponding target.
        total_windows = self.num_timesteps - self.lookback
        X_full = X_full[:total_windows]

        # Get target values as a torch tensor.
        if override_target is not None:
            target_data = torch.tensor(override_target, dtype=torch.float32, device=self.device)
        else:
            target_data = torch.tensor(self.data[:, target_var_idx], dtype=torch.float32, device=self.device)
        # For each window starting at index i, the corresponding target is at time index i + lookback.
        Y_full = target_data[self.lookback: self.lookback + total_windows]

        # Compute the number of training samples from the available windows.
        train_samples = int(train_split * total_windows)

        # Split into training and validation sets.
        X_train = X_full[:train_samples]
        Y_train = Y_full[:train_samples]
        X_val = X_full[train_samples:]
        Y_val = Y_full[train_samples:]

        return X_train, Y_train, X_val, Y_val