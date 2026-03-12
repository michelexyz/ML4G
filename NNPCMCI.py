import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product, combinations as combi_sub

###############################################################################
# 1) A Tiny RNN Model for Multi-Dimensional Inputs
###############################################################################
class SmallRNN(nn.Module):
    """
    A small RNN with a single hidden layer and linear output, intended
    for univariate or multivariate time-series inputs.

    Args:
        input_size (int): number of features in the input.
        hidden_size (int): number of hidden units.
        output_size (int): dimension of output (1 for univariate regression).
    """
    def __init__(self, input_size=1, hidden_size=8, output_size=1):
        super(SmallRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the RNN.
        Args:
            x (torch.Tensor): shape (batch_size, seq_len, input_size)
        Returns:
            out (torch.Tensor): shape (batch_size, output_size)
        """
        rnn_out, hidden = self.rnn(x)
        # Take the last time-step's hidden state
        out = self.fc(rnn_out[:, -1, :])
        return out


###############################################################################
# 2) PCMCI-like Algorithm Class
###############################################################################
class NNPCMCI:
    """
    A class that implements a neural-network-based version of PCMCI/PC,
    using a small RNN to compute single-direction R² for partial correlation tests.

    The constructor stores the data, model class, and hyperparameters.
    Call .run() to execute the discovery procedure and retrieve an adjacency matrix.

    Args:
        data (np.ndarray): shape (T, N) time series data
        model_cls (type): a neural network class (e.g., SmallRNN)
        alpha (float): threshold for independence (on partial correlation)
        seq_len (int): how many time steps to feed into the RNN
        max_cond_set_size (int): maximum size of conditioning sets
        hidden_size (int): hidden dimension for the RNN
        epochs (int): number of training epochs
        lr (float): learning rate
        verbose (bool): whether to print debugging info
    """

    def __init__(self,
                 data: np.ndarray,
                 model_cls=SmallRNN,
                 alpha=0.05,
                 seq_len=5,
                 max_cond_set_size=2,
                 hidden_size=8,
                 epochs=10,
                 lr=1e-3,
                 verbose=True):
        self.data = data
        self.model_cls = model_cls  # e.g., SmallRNN
        self.alpha = alpha
        self.seq_len = seq_len
        self.max_cond_set_size = max_cond_set_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

    def run(self):
        """
        Execute the PCMCI-like procedure to learn an adjacency matrix.

        Returns:
            adj_matrix (np.ndarray): shape (N, N), where 1 = edge, 0 = no edge.
        """
        N = self.data.shape[1]
        adj_matrix = np.ones((N, N), dtype=int)  # fully connected initially

        for cond_size in range(1, self.max_cond_set_size + 1):
            # Check edges i -> j for i, j in product
            for i, j in product(range(N), repeat=2):
                

                if adj_matrix[i, j] == 0:
                    # already removed
                    continue

                # Gather i's ingoing neighbors
                neighbors_i = np.where(adj_matrix[:,i] == 1)[0].tolist()
                neighbors_j = np.where(adj_matrix[:,j] == 1)[0].tolist()
                if j in neighbors_i:
                    neighbors_i.remove(j)  # exclude j from neighbors
                if i in neighbors_j:
                    neighbors_j.remove(i)
                # get the common neighbors
                common_neighbors = set(neighbors_i).intersection(neighbors_j)

    
                remove_edge = self._check_cond_subsets(
                    i, j, list(common_neighbors), cond_size, adj_matrix
                )
                if remove_edge:
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0  # symmetrical removal

        return adj_matrix

    def _check_cond_subsets(self, i, j, neighbors_i, cond_size, adj_matrix):
        """
        Check all subsets of neighbors_i of size cond_size; if any indicates
        independence, remove the edge.

        Returns:
            remove_edge (bool): True if edge i->j should be removed.
        """
        all_cond_subsets = list(combi_sub(neighbors_i, cond_size))
        for cond_subset in all_cond_subsets:
            if self.verbose:
                print(f"Testing {i} -> {j} conditioned on {cond_subset}")

            if len(cond_subset) == 0:
                # Unconditional test: R²(Y_j <- X_i)
                target_series = self.data[:, j]  # Y_j
                predictor_series = self.data[:, i]  # X_i
                r2_val = self._compute_r2_pair(target_series, predictor_series)
                if self.verbose:
                    print(f"R²({j} <- {i}) = {r2_val:.3f}")

                if r2_val < self.alpha:
                    return True  # remove edge immediately
            else:
                # Condition on Z
                X_data = self.data[:, i]
                Y_data = self.data[:, j]
                Z_data = self.data[:, cond_subset]  # shape (T, cond_size)
                r_xy_given_z = self._partial_corr_r2(X_data, Y_data, Z_data)
                if self.verbose:
                    print(f"Partial R²({j} <- {i} | {cond_subset}) = {r_xy_given_z:.3f}")

                if abs(r_xy_given_z) < self.alpha:
                    return True
        return False

    #########################
    # Helper / Utility Methods
    #########################

    def _compute_r2_pair(self, target_series, predictor_series):
        """
        Compute single-direction R² (target <- predictor) using an RNN model.
        """
        train_loader, val_loader = self._create_data_loaders_multivar(
            predictor_series, target_series, seq_len=self.seq_len, split=0.8
        )
        p = predictor_series.shape[1] if len(predictor_series.shape) > 1 else 1
        train_mean = np.mean(target_series)
        r2_val = self._train_and_val_r2(train_mean, train_loader, val_loader, input_size=p)
        return r2_val

    def _partial_corr_r2(self, X_data, Y_data, Z_data=None):
        """
        Compute partial correlation in one direction, i.e. R²(Y <- X | Z).

        If Z_data is not None, we compute:
            r²_yx - r²_xz * r²_yz
            --------------------------------
            sqrt((1 - r²_xz^2)(1 - r²_yz^2))
        with single-direction R² in each case.

        Returns a float for the partial correlation measure.
        """
        # R²(Y <- X)
        r2_yx = self._compute_r2_pair(Y_data, X_data)

        if Z_data is None:
            return r2_yx

        # R²(X <- Z), R²(Y <- Z)
        r2_xz = self._compute_r2_pair(X_data, Z_data)
        r2_yz = self._compute_r2_pair(Y_data, Z_data)

        denom = np.sqrt((1 - r2_xz**2)*(1 - r2_yz**2))
        if denom < 1e-12:
            return 0.0
        return (r2_yx - r2_xz*r2_yz)/denom

    def _create_data_loaders_multivar(self, X_series, Y_series,
                                      seq_len=5, batch_size=16, split=0.8):
        """
        Build train and validation DataLoaders for multiple-input single-output.
        """
        if len(X_series.shape) == 1:
            X_series = X_series.reshape(-1, 1)
        if len(Y_series.shape) > 1 and Y_series.shape[1] == 1:
            Y_series = Y_series.reshape(-1)

        T = len(X_series)
        X, Y = [], []

        # Build sequences
        for i in range(seq_len, T):
            x_window = X_series[i-seq_len:i]  # (seq_len, p)
            y_value  = Y_series[i]           # scalar
            X.append(x_window)
            Y.append(y_value)

        X = np.array(X)  # shape: (num_samples, seq_len, p)
        Y = np.array(Y)  # shape: (num_samples,)

        split_index = int(len(X)*split)
        X_train, Y_train = X[:split_index], Y[:split_index]
        X_val,   Y_val   = X[split_index:], Y[split_index:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
        Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32).unsqueeze(-1)

        train_ds = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
        val_ds   = torch.utils.data.TensorDataset(X_val_t,   Y_val_t)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def _train_and_val_r2(self, train_mean, train_loader, val_loader, input_size=1):
        """
        Train a new instance of self.model_cls and compute R² on the validation set.
        """
        model = self.model_cls(input_size=input_size,
                               hidden_size=self.hidden_size,
                               output_size=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Train loop
        for epoch in range(self.epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        # Validation R²
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x).squeeze()
                all_preds.append(preds.cpu().numpy())
                all_true.append(batch_y.squeeze().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_true  = np.concatenate(all_true)

        # R^2 = 1 - SSR/SST
        sst = np.sum((all_true - train_mean)**2)
        ssr = np.sum((all_preds - all_true)**2)
        if sst < 1e-12:
            return 0.0

        r2  = 1 - ssr/sst
        if ssr > sst:
            if self.verbose:
                print("Warning: SSR > SST. Model is worse than mean prediction.")
            r2 = 0.0
        return r2


###############################################################################
# 3) Synthetic Data Generation (example)
###############################################################################
def generate_synthetic_data(T=200):
    """
    Generate synthetic time-series data for 3 variables: X, Y, Z with specific dependencies:
        X_t = 0.5 * X_{t-1} + 0.3 * Y_{t-1} + eX_t
        Y_t = 0.2 * X_{t-1} + 0.6 * Y_{t-1} + eY_t
        Z_t = 0.7 * Z_{t-1} + eZ_t
    where eX, eY, eZ are i.i.d. Gaussian noise.

    Returns:
        data: np.array of shape (T, 3) with columns [X, Y, Z]
    """
    X = np.zeros(T)
    Y = np.zeros(T)
    Z = np.zeros(T)

    # Initialize random noise
    eX = 0.1 * np.random.randn(T)
    eY = 0.1 * np.random.randn(T)
    eZ = 0.1 * np.random.randn(T)

    # Initialize first time step randomly
    X[0] = np.random.randn()
    Y[0] = np.random.randn()
    Z[0] = np.random.randn()

    for t in range(1, T):
        X[t] = 0.5 * X[t-1] + 0.3 * Y[t-1] + eX[t]
        Y[t] = 0.2 * X[t-1] + 0.6 * Y[t-1] + eY[t]
        Z[t] = 0.7 * Z[t-1] + eZ[t]

    data = np.column_stack((X, Y, Z))
    return data


###############################################################################
# 4) Demo usage
###############################################################################
if __name__ == "__main__":
    # Example: generate synthetic data with 3 variables
    np.random.seed(42)
    data = generate_synthetic_data(T=300)

    # Instantiate the NNPCMCI class
    nn_pcmci = NNPCMCI(
        data,
        model_cls=SmallRNN,    # The RNN model class we'll use
        alpha=0.001,           # threshold for significance
        seq_len=5,             # length of RNN input sequence
        max_cond_set_size=1,   # condition on up to 1 neighbor
        hidden_size=8,
        epochs=10,
        lr=1e-3,
        verbose=True
    )

    # Run the PC-like algorithm
    adjacency = nn_pcmci.run()
    print("Learned adjacency matrix (0=none, 1=edge):")
    print(adjacency)
