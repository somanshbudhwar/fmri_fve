# Plotting results

import matplotlib.pyplot as plt
import numpy as np
import json

from experiment_configs import SCALING_CONSTANT, HIDDEN_LAYERS_FACTORS

FVE_VALS = [0.0,0.2, 0.4, 0.6, 0.8, 1.0]

FOLDER = "results_temp5"
# Read in the results
raw = json.load(open(f"{FOLDER}/results.json"),parse_float=float)
ROWS = 1000
COLS = 100
SCALING_CONSTANT = 3.0
LAYERS = 3
HIDDEN_LAYERS_FACTORS = 2
ACTIVATION = "leaky_relu"

def plot_scores():
    filter_raw = []
    # Filter out the results that have the 100 cols
    scores = {}
    for r in raw:
        if r["config"]["cols"] == COLS:
            if r["config"]["activation"] == ACTIVATION:
                if r["config"]["hidden_layers_factor"] == HIDDEN_LAYERS_FACTORS:
                    if r["config"]["scaling_constant"] == SCALING_CONSTANT:
                        if r["config"]["layers"] == LAYERS:
                            for fve in FVE_VALS:
                                if r["config"]["fve"] == fve:
                                    print(fve,[res["r2"] for res in r["results"]])
                                    if scores.get("r2") is None:
                                        scores["r2"] = [[res["r2"] for res in r["results"]]]
                                        scores["corrected_r2"] = [[res["corrected_r2"] for res in r["results"]]]
                                        scores["gwash"] = [[res["gwash"] for res in r["results"]]]
                                        scores["r2_pinv"] = [[res["r2_pinv"] for res in r["results"]]]
                                        scores["corrected_r2_pinv"] = [[res["corrected_r2_pinv"] for res in r["results"]]]
                                        scores["gwash_layer_values_0"] = [[float(res["gwash_layer_values"]["Layer 0"])  for res in r["results"]]]
                                        if r["results"][0].get("gwash_layer_values") is not None:
                                            for i in range(1,len(r["results"][0]["gwash_layer_values"])):
                                                scores[f"gwash_layer_values_{i}"] = [[float(res["gwash_layer_values"][f"Layer {i}"]) for res in r["results"]]]
                                    else:
                                        scores["r2"].append([res["r2"] for res in r["results"]])
                                        scores["corrected_r2"].append([res["corrected_r2"] for res in r["results"]])
                                        scores["gwash"].append([res["gwash"] for res in r["results"]])
                                        scores["r2_pinv"].append([res["r2_pinv"] for res in r["results"]])
                                        scores["corrected_r2_pinv"].append([res["corrected_r2_pinv"] for res in r["results"]])
                                        scores["gwash_layer_values_0"].append([float(res["gwash_layer_values"]["Layer 0"]) for res in r["results"]])
                                        if r["results"][0].get("gwash_layer_values") is not None:
                                            for i in range(1,len(r["results"][0]["gwash_layer_values"])):
                                                # for res in r["results"]:
                                                #     print("&"*100)
                                                #     print(res["gwash_layer_values"])
                                                scores[f"gwash_layer_values_{i}"].append([float(res["gwash_layer_values"][f"Layer {i}"]) for res in r["results"]])
                                                pass

    # print(scores)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), constrained_layout=True)
    plt.title(f"Cols: {COLS} Layers: {LAYERS} Hidden Layers Factor: {HIDDEN_LAYERS_FACTORS} Scaling Constant: {SCALING_CONSTANT}")
    identx = [0.0, 1.0]
    identy = [0.0, 1.0]

    axes.set_xlim([0, 1.0])
    axes.set_ylim([0., 1.0])
    axes.plot(identx, identy, alpha=0.6, label='_Hidden label', color='tab:gray')
    # print("#"*100)
    # print(scores["r2"])
    axes.plot(FVE_VALS, np.array(scores["r2"]), alpha=0.1, label='_Hidden label')
    avg = np.array(scores["r2"]).mean(1)
    axes.plot(FVE_VALS, avg, marker="o", alpha=0.6, label="R2", color='tab:pink')

    axes.plot(FVE_VALS, np.array(scores["corrected_r2"]), alpha=0.1, label='_Hidden label')
    avg = np.array(scores["corrected_r2"]).mean(1)
    axes.plot(FVE_VALS, avg, marker="o", alpha=0.6, label="Corrected R2", color='tab:blue')

    axes.plot(FVE_VALS, np.array(scores["gwash"]), alpha=0.1, label='_Hidden label')
    avg = np.array(scores["gwash"]).mean(1)
    axes.plot(FVE_VALS, avg, marker="o", alpha=0.6, label="GWASH", color='tab:red')

    axes.plot(FVE_VALS, np.array(scores["r2_pinv"]), alpha=0.1, label='_Hidden label')
    avg = np.array(scores["r2_pinv"]).mean(1)
    axes.plot(FVE_VALS, avg, marker="o", alpha=0.6, label="R2 PINV", color='tab:green')

    axes.plot(FVE_VALS, np.array(scores["corrected_r2_pinv"]), alpha=0.1, label='_Hidden label')
    avg = np.array(scores["corrected_r2_pinv"]).mean(1)
    axes.plot(FVE_VALS, avg, marker="o", alpha=0.6, label="Corrected R2 PINV", color='tab:orange')

    # Flatten the arrays before plotting
    if scores.get("gwash_layer_values_0") is not None:
        flattened_values = np.array(scores["gwash_layer_values_0"])
        axes.plot(FVE_VALS, flattened_values, alpha=0.1, label='_Hidden label')
        avg = np.array(scores["gwash_layer_values_0"]).mean(1)
        axes.plot(FVE_VALS, avg, marker="o", alpha=0.6, label="GWASH Layer 0", color='tab:purple')

    if scores.get("gwash_layer_values_1") is not None:
        flattened_values = np.array(scores["gwash_layer_values_1"])
        axes.plot(FVE_VALS, flattened_values, alpha=0.1, label='_Hidden label')
        avg = np.array(scores["gwash_layer_values_1"]).mean(1)
        axes.plot(FVE_VALS, avg, marker="o", alpha=0.6, label="GWASH Layer 1", color='tab:cyan')

    if scores.get("gwash_layer_values_2") is not None:
        flattened_values = np.array(scores["gwash_layer_values_2"])
        axes.plot(FVE_VALS, flattened_values, alpha=0.1, label='_Hidden label')
        avg = np.array(scores["gwash_layer_values_2"]).mean(1)
        axes.plot(FVE_VALS, avg, marker="o", alpha=0.6, label="GWASH Layer 2", color='tab:olive')

    axes.legend(loc="upper left")

    plt.show()


plot_scores()