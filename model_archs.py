import torch
import torch.nn as nn
import math

LEAKY_ALPHA = 0.01
mps_device = torch.device("cpu")


def col_normalization(x):
    mean = torch.mean(x, dim=0, keepdim=True)
    std = torch.std(x, dim=0, keepdim=True)

    x = (x - mean) / std
    return x


class GeneratorNN(nn.Module):
    def __init__(self, input_size=100, hidden_layers=(50,25,12,6,3),activation="linear", output_size=1, scaling_constant=1.0):
        super(GeneratorNN, self).__init__()
        self.layers = len(hidden_layers)
        self.scaling_constant = scaling_constant
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky-relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Identity()

        if self.layers > 0:
            self.stack1 = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
            )

        if self.layers > 1:
            self.stack2 = nn.Sequential(
                nn.Linear(hidden_layers[0], hidden_layers[1]),
            )
        if self.layers > 2:
            self.stack3 = nn.Sequential(
                nn.Linear(hidden_layers[1], hidden_layers[2]), )
        if self.layers > 3:
            self.stack4 = nn.Sequential(
                nn.Linear(hidden_layers[2], hidden_layers[3]),
            )
        if self.layers > 4:
            self.stack5 = nn.Sequential(
                nn.Linear(hidden_layers[3], hidden_layers[4]),
            )
        if self.layers > 5:
            self.stack6 = nn.Sequential(
                nn.Linear(hidden_layers[4], hidden_layers[5]),
            )

        self.final = nn.Sequential(
            nn.Linear(hidden_layers[-1], output_size)
        )

    def norm_and_act(self, x):
        x = col_normalization(x)
        x = x * self.scaling_constant
        x = self.activation(x)

        return x

    def get_xdash(self, x):
        if self.layers > 0:
            x = self.stack1(x)
            x = self.norm_and_act(x)
        if self.layers > 1:
            x = self.stack2(x)
            x = self.norm_and_act(x)
        if self.layers > 2:
            x = self.stack3(x)
            x = self.norm_and_act(x)
        if self.layers > 3:
            x = self.stack4(x)
            x = self.norm_and_act(x)
        if self.layers > 4:
            x = self.stack5(x)
            x = self.norm_and_act(x)
        if self.layers > 5:
            x = self.stack6(x)
            x = self.norm_and_act(x)
        if self.layers > 6:
            x = self.stack7(x)
            x = self.norm_and_act(x)

        return x

    def forward(self, x):
        x = self.get_xdash(x)
        x = self.final(x)
        return x


class FittingNN(nn.Module):
    def __init__(self, input_size=100, hidden_layers=(50,25,12,6,3), activation="linear", output_size=1,):
        super(FittingNN, self).__init__()
        self.layers = len(hidden_layers)
        if self.layers > 0:
            self.stack1 = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
            )

        if self.layers > 1:
            self.stack2 = nn.Sequential(
                nn.Linear(hidden_layers[0], hidden_layers[1]),
            )
        if self.layers > 2:
            self.stack3 = nn.Sequential(
                nn.Linear(hidden_layers[1], hidden_layers[2]), )
        if self.layers > 3:
            self.stack4 = nn.Sequential(
                nn.Linear(hidden_layers[2], hidden_layers[3]),
            )
        if self.layers > 4:
            self.stack5 = nn.Sequential(
                nn.Linear(hidden_layers[3], hidden_layers[4]),
            )
        if self.layers > 5:
            self.stack6 = nn.Sequential(
                nn.Linear(hidden_layers[4], hidden_layers[5]),
            )

        self.final = nn.Sequential(
            nn.Linear(hidden_layers[-1], output_size)
        )
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky-relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Identity()


    def norm_and_act(self, x):
        x = nn.BatchNorm1d(x.shape[-1]).to(mps_device)(x)
        x = self.activation(x)

        return x

    def get_xdash(self, x):
        if self.layers > 0:
            x = self.stack1(x)
            x = self.norm_and_act(x)
        if self.layers > 1:
            x = self.stack2(x)
            x = self.norm_and_act(x)
        if self.layers > 2:
            x = self.stack3(x)
            x = self.norm_and_act(x)
        if self.layers > 3:
            x = self.stack4(x)
            x = self.norm_and_act(x)
        if self.layers > 4:
            x = self.stack5(x)
            x = self.norm_and_act(x)
        if self.layers > 5:
            x = self.stack6(x)
            x = self.norm_and_act(x)
        if self.layers > 6:
            x = self.stack7(x)
            x = self.norm_and_act(x)

        return x

    def forward(self, x):
        x = self.get_xdash(x)
        x = self.final(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        h1, h2, h3 = int(input_dim/2), int(input_dim/4), int(input_dim/8)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LeakyReLU(negative_slope=LEAKY_ALPHA),
            nn.Linear(h1, h2),
            nn.LeakyReLU(negative_slope=LEAKY_ALPHA),
            nn.Linear(h2, h3),
            nn.LeakyReLU(negative_slope=LEAKY_ALPHA),
            nn.Linear(h3, latent_dim),
            nn.LeakyReLU(negative_slope=LEAKY_ALPHA),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3),
            nn.LeakyReLU(negative_slope=LEAKY_ALPHA),
            nn.Linear(h3, h2),
            nn.LeakyReLU(negative_slope=LEAKY_ALPHA),
            nn.Linear(h2, h1),
            nn.LeakyReLU(negative_slope=LEAKY_ALPHA),
            nn.Linear(h1, input_dim),
            nn.LeakyReLU(negative_slope=LEAKY_ALPHA),
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded, latent