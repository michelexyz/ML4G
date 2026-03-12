import numpy as np
from sklearn.linear_model import LinearRegression

class LinearLinkFitter:
    """
    Fits a linear model for each variable i, using all parents j at each lag l
    indicated in a 3D graph. Returns a (N, N, L) coefficient array with np.nan 
    for non-links.
    """
    def __init__(self, data, graph_3d):
        """
        Args:
            data: shape (T, N) time-series. data[t, i] = X_{i,t}.
            graph_3d: shape (N, N, L). e.g. graph[i, j, l] = '-->' if j->i at lag l,
                      or '' if no link.
        """
        self.data = data
        self.graph = graph_3d
        
        self.T, self.N = data.shape
        # The third dimension is the maximum lag
        self.L = graph_3d.shape[2]

        # Just a quick check
        if graph_3d.shape[0] != self.N or graph_3d.shape[1] != self.N:
            raise ValueError("graph_3d's first two dims must match data's N.")

    def fit_links(self):
        """
        Fits one linear model per variable i, using all indicated parents j,l.
        Returns:
            coefs: shape (N, N, L) array of floats, with np.nan where no link is present.
        """
        coefs = np.full((self.N, self.N, self.L), np.nan, dtype=float)

        # For each child variable i:
        for i in range(self.N):
            # 1) Find all (j, l) such that graph[i,j,l] is a non-empty link
            parent_lags = []
            for j in range(self.N):
                for l in range(self.L):
                    if self.graph[ j,i, l] != '':
                        parent_lags.append((j, l))

            if not parent_lags:
                # no parents => skip
                continue

            # 2) Build design matrix X and target y
            # We want to predict data[t, i] from data[t-l, j] for (j,l) in parent_lags.
            max_lag = max(l for _, l in parent_lags)
            # We can only form Y[t] for t >= max_lag
            T_eff = self.T - max_lag  # number of valid rows

            # Initialize design matrix:
            # shape = (T_eff, num_parents), plus we will feed it into scikit-learn
            X_mat = np.zeros((T_eff, len(parent_lags)), dtype=float)
            y_vec = np.zeros(T_eff, dtype=float)

            for row_idx, t in enumerate(range(max_lag, self.T)):
                # y = X_{i, t}
                y_vec[row_idx] = self.data[t, i]
                # fill columns
                for col_idx, (j, l) in enumerate(parent_lags):
                    X_mat[row_idx, col_idx] = self.data[t - l, j]

            # 3) Fit linear regression
            lr = LinearRegression(fit_intercept=True)
            lr.fit(X_mat, y_vec)
            betas = lr.coef_  # shape = (num_parents,)

            # 4) Store the fitted betas in coefs[i, j, l]
            for (col_idx, (j, l)) in enumerate(parent_lags):
                coefs[j, i, l] = betas[col_idx]

        return coefs
