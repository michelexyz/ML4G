import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product

from itertools import combinations as combi_sub

import numpy as np

###############################################################################
# 1) A Tiny RNN Model for Multi-Dimensional Inputs
###############################################################################
class SmallRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=8, output_size=1):
        super(SmallRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        rnn_out, hidden = self.rnn(x)
        # Take the last time-step's hidden state
        out = self.fc(rnn_out[:, -1, :])
        return out


###############################################################################
# 2) Data Loader: Multiple Input Features, Single Target
###############################################################################
def create_data_loaders_multivar(X_series, Y_series, seq_len=5,
                                 batch_size=16, split=0.8):
    """
    X_series: np.array of shape (T, p) or (T,) if p=1
              The predictor data at each time step.
    Y_series: np.array of shape (T,) or (T,1)
              The scalar target data at each time step.
    seq_len : how many time steps we feed into the RNN.
    batch_size : ...
    split : fraction of data to use for training (vs. validation).

    Returns train_loader, val_loader.
    """

    # Ensure X_series has shape (T, p), even if p=1
    if len(X_series.shape) == 1:
        X_series = X_series.reshape(-1, 1)
    # Ensure Y_series has shape (T,)
    if len(Y_series.shape) > 1 and Y_series.shape[1] == 1:
        Y_series = Y_series.reshape(-1)

    T = len(X_series)
    X, Y = [], []

    # Build sequences of length seq_len
    for i in range(seq_len, T):
        x_window = X_series[i-seq_len:i]     # shape (seq_len, p)
        y_value  = Y_series[i]              # scalar
        X.append(x_window)
        Y.append(y_value)

    X = np.array(X)  # shape (num_samples, seq_len, p)
    Y = np.array(Y)  # shape (num_samples,)

    # Train/Val split
    split_index = int(len(X) * split)
    X_train, Y_train = X[:split_index], Y[:split_index]
    X_val,   Y_val   = X[split_index:],   Y[split_index:]

    # Convert to torch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1) 
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32).unsqueeze(-1)

    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    val_dataset   = torch.utils.data.TensorDataset(X_val_t,   Y_val_t)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    return train_loader, val_loader

###############################################################################
# 3) Train and Get R² (Single Direction)
###############################################################################
def train_and_val_r2(train_mean, train_loader, val_loader, input_size=1, hidden_size=8,
                     epochs=10, lr=1e-3):
    """
    Trains a SmallRNN to predict a 1D target from multi-dim inputs, returning R² on validation set.
    """
    model = SmallRNN(input_size=input_size, hidden_size=hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train loop (basic)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)  # shape (batch_size, 1)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    # Compute R² on validation
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
    r2  = 1 - ssr/sst if sst > 1e-12 else 0.0

    if ssr > sst:
        print("Warning: SSR > SST. Model is worse than mean prediction.")
        return 0.0
    return r2

###############################################################################
# 4) Partial Correlation Surrogate (One-Directional R²)
###############################################################################
def partial_corr_r2(X_data, Y_data, Z_data=None,
                                     seq_len=5, batch_size=16, verbose=True):
    """
    X_data: shape (T,) or (T, pX) - target we want to predict
    Y_data: shape (T,) or (T, pY) - predictor
    Z_data: shape (T, pZ) or None (pZ could be the # of conditioning vars)
    seq_len: how many time steps for RNN input
    alpha  : threshold for deciding independence

    Returns: r_xy_given_z (float)
    """

    # 1. Compute r2_xy = R^2( Y <- X )
    r2_yx = compute_r2_pair(Y_data, X_data, seq_len=seq_len, batch_size=batch_size)

    # If no Z provided, interpret this as zero conditioning
    if Z_data is None:
        return r2_yx  # might just return the unconditional measure

    # 2. Compute r2_xz = R^2( X <- Z )
    r2_xz = compute_r2_pair(X_data, Z_data, seq_len=seq_len, batch_size=batch_size)

    # 3. Compute r2_yz = R^2( Y <- Z )
    r2_yz = compute_r2_pair(Y_data, Z_data, seq_len=seq_len, batch_size=batch_size)

    # 4. Partial correlation formula
    denom = np.sqrt((1 - r2_xz**2)*(1 - r2_yz**2))
    if denom < 1e-12:
        r_xy_given_z = 0.0
    else:
        r_xy_given_z = (r2_yx - r2_xz*r2_yz)/denom

    if verbose:
        print(f"R²(X <- Z) = {r2_xz:.3f}, R²(Y <- Z) = {r2_yz:.3f}")
        print(f"Partial R²(X <- Y | Z) = {r_xy_given_z:.3f}")

    return r_xy_given_z

###############################################################################
# Helper to train a small RNN with multi-dim input -> 1D target
###############################################################################
def compute_r2_pair(target_series, predictor_series, seq_len=5, batch_size=16,
                    hidden_size=8, epochs=10, lr=1e-3):
    """
    target_series: shape (T,) or (T, 1)
    predictor_series: shape (T, p) for p predictor features

    Returns R²(X <- Y) for the predictor -> target mapping.
    """
    # Build data loaders for (predictor -> target) mapping
    train_loader, val_loader = create_data_loaders_multivar(
        predictor_series, target_series,
        seq_len=seq_len, batch_size=batch_size, split=0.8
    )

    # Train, get R²
    p = predictor_series.shape[1] if len(predictor_series.shape) > 1 else 1

    train_mean = np.mean(target_series)

    assert len(target_series.shape) == 1, "Target should be 1D"
    r2_score = train_and_val_r2(train_mean, train_loader, val_loader,
                                input_size=p,
                                hidden_size=hidden_size,
                                epochs=epochs, lr=lr)
    return r2_score

###############################################################################
# 5) A Simplified PC Algorithm Sketch
###############################################################################
def NN_PCMCI(data, alpha=0.05, seq_len=5, max_cond_set_size=2, verbose=True):
    """
    data: np.array of shape (T, N)
    alpha: threshold for partial correlation
    seq_len: how many time steps for RNN
    max_cond_set_size: maximum size of conditioning sets
    Returns adjacency matrix (N x N) with 1 = edge, 0 = no edge.
    Additionally, you could store directional info by comparing R^2(X<-Y) vs R^2(Y<-X).
    """
    N = data.shape[1]
    variables = range(N)
    # Start with a fully-connected graph
    adj_matrix = np.ones((N, N), dtype=int)

    for cond_size in range(1, max_cond_set_size+1):
        
        for (i, j) in product(variables, repeat=2):
            if verbose:
                print(f"Checking edge {i} -> {j} with cond_size={cond_size}")
            if adj_matrix[i, j] == 0:
                continue

            # Gather possible conditioning subsets of the neighbors
            neighbors_i = np.where(adj_matrix[i] == 1)[0].tolist() #TODO CHECK CORRECTNESS
            if j in neighbors_i:
                neighbors_i.remove(j)  # exclude j from i's neighbors

            remove_edge = check_cond_subsets(
                neighbors_i, cond_size, data, i, j,
                alpha=alpha, seq_len=seq_len, verbose=verbose
            )
            

            if remove_edge:
                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0

    # (Optional) Orientation step: decide direction for each remaining edge
    # For each i--j that remains, compare R^2(X<-Y) vs. R^2(Y<-X), or run the
    # standard PC orientation rules for colliders, etc.

    return adj_matrix

def check_cond_subsets(neighbors, cond_size, data, i, j, alpha=0.05, seq_len=5, verbose=True):
                       
    all_cond_subsets = list(combi_sub(neighbors, cond_size))
    
    remove_edge = False
    for cond_subset in all_cond_subsets:

        if verbose:
            print(f"Checking subset {cond_subset}")
        
        if len(cond_subset) == 0:
            # unconditioned partial correlation => r_xy
            # single direction: r2_xy
            X_data = data[:, i]
            Y_data = data[:, j]
            r_xy = compute_r2_pair(Y_data, X_data, seq_len=seq_len)
            if verbose:
                print(f"R²(Y <- X) = {r_xy:.3f}")
            if r_xy < alpha:
                remove_edge = True
                break
        else:
            # condition on Z
            Z_data = data[:, cond_subset]  # shape (T, cond_size)
            # partial corr in one direction
            X_data = data[:, i]
            Y_data = data[:, j]
            r_xy_given_z = partial_corr_r2(
                X_data, Y_data, Z_data,
                seq_len=seq_len, verbose=verbose
            )
            if abs(r_xy_given_z) < alpha:
                remove_edge = True
                break

    return remove_edge

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
# Demo usage
###############################################################################
if __name__ == "__main__":
    # # For reproducibility
    # np.random.seed(42)

    # # Mock data: (T, N) -> T=200 time steps, N=5 variables
    # T, N = 200, 5
    # data = np.random.randn(T, N)

    # # Run the PC algorithm with single-direction partial correlation
    # adjacency = NN_PCMCI(
    #     data,
    #     alpha=0.05,
    #     seq_len=5,
    #     max_cond_set_size=1
    # )

    # # Print the resulting adjacency matrix (undirected for now).
    # # 1 => edge, 0 => no edge
    # print("Learned adjacency matrix:")
    # print(adjacency)

    # 1) Generate synthetic data
    data = generate_synthetic_data(T=300)  # shape = (300, 3)

    # 2) Run your single-direction PC-like algorithm
    adjacency = NN_PCMCI(
        data,           # shape (T, N=3)
        alpha=0.001,     # threshold or significance
        seq_len=5,      # how many time steps for RNN
        max_cond_set_size=1,
        verbose=True
    )

    print("Learned adjacency matrix:")
    print(adjacency)


