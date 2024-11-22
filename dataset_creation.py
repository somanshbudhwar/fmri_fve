import torch
import math
from model_archs import GeneratorNN
from torch import nn
from utils import col_normalization, init_weights

mps_device = torch.device("cpu")




def make_linear_df(rows=1000,cols=100, fve=1.0, hidden_layers=(10,), activation="tanh",
            output_size=1,scaling_constant=1.0):
    x = torch.normal(0, 1, size=(rows, cols)).to(mps_device)

    model = GeneratorNN(input_size=cols, hidden_layers=hidden_layers,activation=activation,
                        output_size=output_size, scaling_constant=scaling_constant).to(mps_device)

    model = model.apply(init_weights)
    x_dash = model.get_xdash(x)
    x_dash = col_normalization(x_dash)
    # print(x_dash.mean(), x_dash.var())

    beta=torch.normal(0,math.sqrt(fve/hidden_layers[-1]),size=(hidden_layers[-1], 1)).to(mps_device)

    x_beta = torch.matmul(x_dash, beta)
    x_beta = math.sqrt(fve) * (x_beta - x_beta.mean()) / x_beta.std()
    e0 = torch.normal(0, math.sqrt(1 - fve), size=(rows, 1)).to(mps_device)
    y = x_beta + e0

    df = torch.cat((x, y), axis=-1)

    return df.detach()#,x.detach(),x_dash.detach(),x_beta.detach(),e0.detach(),y.detach()
