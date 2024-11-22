from pathlib import Path
import torch
import torch.nn as nn
import math


def calculate_r2(y_preds, y_test):
    """Calculate R^2 score."""
    y_mean = torch.mean(y_test)
    # v = torch.sum((y_test - y_mean) ** 2)
    # u = torch.sum((y_test - y_preds) ** 2)
    u = ((y_test - y_preds) ** 2).mean()
    v = ((y_test - y_test.mean()) ** 2).mean()

    r2 = 1 - u / v
    return r2

def corrected_r2(y_preds, y_test, n, m):
    """Calculate corrected R^2 score."""
    # n= 1000
    u = ((y_test - y_preds) ** 2).mean()
    if m<n:
        r2_adj = 1 - u / (1 + m / (n - m - 1))
    else:
        r2_adj = (m / n) * (m - n - 1) / (2 * m - n - 1) * ((m - 1) / (m - n - 1) - u)
    return r2_adj

def gwash_paper_fun(X, y):
    # Get the dimensions of X_tilda
    n, m = X.shape

    # Standardize X and y
    X_tilda = col_normalization(X)
    y_tilda = col_normalization(y)

    # Calculate correlation score u
    u = (X_tilda.T @ y_tilda) / torch.sqrt(torch.tensor(n - 1.0))

    # Calculate s^2, empirical second moment of the correlation scores
    s_squared = (1 / m) * torch.norm(u, 2) ** 2

    # Calculate mu_2_hat
    # S_tilda
    S_tilda = (1 / (n - 1)) * (X_tilda.T @ X_tilda)

    # Using eigenvalues to speed up
    mu_2_hat = (1 / m) * torch.sum(S_tilda ** 2) - (m - 1) / (n - 1)

    # Calculate GWASH estimator (19)
    h_gwash = m * (s_squared - 1) / (n * mu_2_hat)

    return h_gwash

def create_directory(directory_path=None):
    if directory_path is None:
        raise ValueError("Directory path is not mentioned.")
        return None
    else:
        path = Path(directory_path)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"Directory {directory_path} created.")
        else:
            print(f"Directory {directory_path} already exists.")

def split_data(data, train_size=0.8):
    """Split data into training, validation, and test sets."""
    rows = data.shape[0]
    train = data[:int(rows * train_size), :]
    test = data[int(rows * train_size):, :]
    return train, test

def get_features_and_labels(data):
    """Extract features and labels from the dataset."""
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def col_normalization(x):
    mean = torch.mean(x, dim=0, keepdim=True)
    std = torch.std(x, dim=0, keepdim=True)

    x = (x - mean) / std
    return x

def init_weights(m):
    for layer in m.children():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=math.sqrt(1.0 / layer.in_features))
            layer.bias.data.fill_(0.01)
    return m

