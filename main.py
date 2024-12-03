import os

from torch.utils.data import DataLoader, TensorDataset
import torch
from experiment_configs import create_experiment_configs
from dataset_creation import make_linear_df
from experiment_configs import ACTIVATIONS
from utils import create_directory, get_features_and_labels, gwash_paper_fun, split_data, calculate_r2, corrected_r2
from model_archs import FittingNN
import json
import concurrent.futures
import json

RESULTS_DIR = "results_temp6"
# temp5 config
# FVE = [0.0,0.2,0.4,0.6,0.8,1.0]
# COLS = [100,1500]
# LAYERS = [3,]
# HIDDEN_LAYERS_FACTORS = [2]
# ACTIVATIONS = ["leaky_relu"]
# OUTPUT_SIZE = 1
# SCALING_CONSTANT = [3.0]

# temp4 config
# FVE = [0.0,0.2,0.4,0.6,0.8,1.0]
# COLS = [100,1500]
# LAYERS = [1,3,5]
# HIDDEN_LAYERS_FACTORS = [1,2]
# ACTIVATIONS = ["leaky_relu"]
# OUTPUT_SIZE = 1
# SCALING_CONSTANT = [1.0,3.0]

device = torch.device(os.getenv("DEVICE","cpu"))
EPOCHS = 2000
BATCH_SIZE = 1024
ROWS = 100000
SAMPLES=2
# COLS = 4000
# LAYERS = 3
# # HIDDEN_LAYERS = [int(COLS/(2**(i+1))) for i in range(LAYERS)]
# HIDDEN_LAYERS = [100,50,25]

def fit_model(model, X_train, y_train):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss().to(device)

    for epoch in range(EPOCHS):
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(X).squeeze(-1)
            y_pred = y_pred.to(device)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}: {loss.item()}")
    return model


def model_predictions(model, X_test, y_test):
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    with torch.no_grad():
        model.eval()
        y_preds = model(X_test).squeeze(-1)
        # print(f"Loss: {torch.nn.MSELoss()(y_preds, y_test)}")
    return y_preds


def gwash_layer_values(model, X_test):
    with torch.no_grad():
        X_test = X_test.to(device)
        model.eval()
        X_dashes = model.get_xdash(X_test)
        y_preds = model(X_test)
        gwash_layer_results = {}
        for i, X_dash in enumerate(X_dashes):
            gwash = gwash_paper_fun(X_dash, y_preds)
            gwash_layer_results[f"Layer {i}"] = f"{gwash}"
        return gwash_layer_results


def run_experiment(config):
    results = []
    for sample in range(SAMPLES):
        res = {}
        COLS = config["cols"]
        HIDDEN_LAYERS = config["hidden_layers"]
        fve_val = config["fve"]
        ACTIVATION = config["activation"]
        scaling_constant = config["scaling_constant"]

        df = make_linear_df(rows=ROWS,cols=COLS, fve=fve_val, hidden_layers=HIDDEN_LAYERS, activation=ACTIVATION,scaling_constant=scaling_constant)

        train, test = split_data(df)

        X_train, y_train = get_features_and_labels(train)
        X_test, y_test = get_features_and_labels(test)

        # if COLS > ROWS:
            # model = MPINV model
        betas = torch.matmul(torch.linalg.pinv(X_train), y_train)
        y_preds_pinv = torch.matmul(X_test, betas)
        res["r2_pinv"] = calculate_r2(y_test, y_preds_pinv).item()
        res["corrected_r2_pinv"] = corrected_r2(y_test, y_preds_pinv, ROWS, COLS).item()

        # else:
        model = FittingNN(input_size=COLS, hidden_layers=HIDDEN_LAYERS, activation="tanh", output_size=1).to(device)
        model = fit_model(model, X_train, y_train)
        y_preds = model_predictions(model, X_test, y_test).detach().cpu()
        res["gwash_layer_values"] = gwash_layer_values(model, X_test)

        res["r2"] = calculate_r2(y_test, y_preds).item()
        res["corrected_r2"] = corrected_r2(y_test, y_preds, ROWS, COLS).item()
        res["gwash"] = gwash_paper_fun(X_test, y_test).item()

        # print("R2 score: ", calculate_r2(y_test, y_preds))
        # print("Corrected R2 score: ", corrected_r2(y_test, y_preds, ROWS, COLS))
        # print(f"GWASH for X_test : {gwash_paper_fun(X_test, y_test)}")
        # print(gwash_layer_values(model, X_test))
        results.append(res)
    return results

def run_experiment_parallel(config):
    results = run_experiment(config)
    return {"config": config, "results": results}



if __name__ == "__main__":
    configs = create_experiment_configs()
    all_results = []
    print(f"Running {len(configs)} experiments")

    create_directory(RESULTS_DIR)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(run_experiment_parallel, configs)):
            print(f"Running experiment {i+1}/{len(configs)}")
            all_results.append(result)
            with open(f"{RESULTS_DIR}/results.json", "w") as f:
                json.dump(all_results, f)
# for i, config in enumerate(configs):
#     print(f"Running experiment {i+1}/{len(configs)}")
#     results = run_experiment(config)
#     results_json = {"config": config, "results": results}
#     all_results.append(results_json)
#     # print(results_json)
#     with open("results_temp2/results.json", "w") as f:
#         json.dump(all_results, f)

# Save json file
